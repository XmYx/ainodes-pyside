import os
import torch
from tqdm import tqdm

from backend.singleton import singleton
from backend.torch_gc import torch_gc

gs = singleton

#normal merge

def merge_models(model_0, model_1, alpha, output, device):

    print(f'Start Merge Model {model_0} with {model_1} at an alpha of {str(alpha)} on Device {device}')

    model_0 = torch.load(model_0, map_location=device)
    model_1 = torch.load(model_1, map_location=device)
    theta_0 = model_0["state_dict"]
    theta_1 = model_1["state_dict"]

    print('Start Stage 1')
    for key in tqdm(theta_0.keys(), desc="Stage 1/2"):
        if "model" in key and key in theta_1:
            theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

    print('Start Stage 2')
    for key in tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:
            theta_0[key] = theta_1[key]

    print("Saving...")

    output_file = os.path.join(gs.system.custom_models_dir,f'{output}-{str(alpha)[2:] + "0"}.ckpt')
    torch.save({"state_dict": theta_0}, output_file)

    print("Done!")

    torch_gc()


# EBL merge


# taken from the original SD util.py
def traverse_state_dict(sd_src, sd_dst, state=None, verbose=True, func_not_in_source=None, func_size_different=None,
                        func_matching=None, func_non_tensor=None, func_not_in_dest=None, **kwargs):
    if state == None:
        state = {}

    src_keys = sd_src.keys()
    dst_keys = sd_dst.keys()

    for k in dst_keys:
        if not k in sd_src:
            dst_item = sd_dst[k]
            if torch.is_tensor(dst_item):
                dst_size = dst_item.size()
            else:
                dst_size = ''

            if verbose:
                print(f'Source is missing {k}, type: {type(sd_dst[k])}, size: {dst_size}')

            if func_not_in_source:
                r = func_not_in_source(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                if r != None:
                    sd_dst[k] = r
                else:
                    del sd_dst[k]
        else:
            src_item = sd_src[k]
            dst_item = sd_dst[k]

            if torch.is_tensor(src_item) and torch.is_tensor(dst_item):
                src_size = src_item.size()
                dst_size = dst_item.size()

                if src_size != dst_size:
                    if verbose:
                        print(f'{k} differs in size: src: {src_size}, dst: {dst_size}')

                    if func_size_different:
                        r = func_size_different(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                        if r != None:
                            sd_dst[k] = r
                        else:
                            del sd_dst[k]
                else:
                    if verbose:
                        print(f'{k} matches')

                    if func_matching:
                        r = func_matching(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                        if r != None:
                            sd_dst[k] = r
                        else:
                            del sd_dst[k]
            else:
                if verbose:
                    print(f'{k} is not a torch Tensor')

                if func_non_tensor:
                    r = func_non_tensor(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                    if r != None:
                        sd_dst[k] = r
                    else:
                        del sd_dst[k]

    for k in src_keys:
        if k not in sd_dst:
            src_item = sd_src[k]
            if torch.is_tensor(src_item):
                src_size = src_item.size()
            else:
                src_size = ''

            if verbose:
                print(f'Destination is missing {k}, type: {type(sd_src[k])}, size: {src_size}')

            if func_not_in_dest:
                r = func_not_in_dest(sd_src, sd_dst, k, state, verbose=verbose, **kwargs)
                if r != None:
                    sd_dst[k] = r

    return sd_dst, state


# taken from smirkingface github
def merge_not_in_source(sd_src, sd_dst, k, state, verbose=True, alpha_new=0.5, **kwargs):
    dst_item = sd_dst[k]

    if len(dst_item.shape) == 1:
        if verbose:
            print(f'{k}: Bias, merging new weight')
        return alpha_new*dst_item
    else:
        if verbose:
            print(f'{k}: Weight, merging new weight with dirac')
        tmp = alpha_new*dst_item
        torch.nn.init.dirac_(dst_item)
        return (1-alpha_new)*dst_item + tmp

def merge_size_different(sd_src, sd_dst, k, state, verbose=True, alpha=0.5, alpha_new=0.5, **kwargs):
    src_size = sd_src[k].size()
    dst_size = sd_dst[k].size()

    # Assumes weight is convolutional
    # TODO: Support different kernel sizes per dimension
    delta = (dst_size[3] - src_size[3])//2
    pd = torch.nn.ConstantPad2d(delta,0)
    mask = pd(torch.ones_like(sd_src[k]))

    if verbose:
        print(f'{k}: Merging')
    return (1-alpha) * pd(sd_src[k]) + mask*alpha*sd_dst[k] + alpha_new*(1-mask)*sd_dst[k]


def merge_matching(sd_src, sd_dst, k, state, verbose=True, alpha=0.5, **kwargs):
    if verbose:
        print(f'{k}: Merging')
    return (1-alpha) * sd_src[k] + alpha*sd_dst[k]

def merge_checkpoint(src_filename, dst_filename, output_filename, alpha=0.5, alpha_new=0.5, verbose=True):
    checkpoint_src = torch.load(src_filename, map_location='cpu')
    checkpoint_dst = torch.load(dst_filename, map_location='cpu')

    checkpoint_dst['state_dict'],_ = traverse_state_dict(checkpoint_src['state_dict'], checkpoint_dst['state_dict'], func_not_in_source=merge_not_in_source,
                                                         func_matching=merge_matching, func_size_different=merge_size_different,
                                                         alpha=alpha, alpha_new=alpha_new, verbose=verbose)

    remove_keys = ['loops', 'optimizer_states', 'lr_schedulers', 'callbacks']
    for x in remove_keys:
        if x in checkpoint_dst:
            del checkpoint_dst[x]

    torch.save(checkpoint_dst, output_filename)
    del checkpoint_dst

def merge_ebl_model(src, dst, output, alpha=0.5, alpha_new=0.5):
    #python scripts/txt2img.py --prompt "a smirking face" --ckpt ./models/SF_EBL_1.0.ckpt --config ./configs/large_inference.yaml
    alpha_new = alpha_new
    if alpha_new == None:
        alpha_new = alpha

    merge_checkpoint(src, dst, output, alpha=alpha, alpha_new=alpha_new)
