import os
import shutil

import safetensors
import torch

import tqdm

from backend.singleton import singleton
from backend.torch_gc import torch_gc

gs = singleton

#normal merge

chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}

def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        pl_sd = safetensors.torch.load_file(checkpoint_file, device=map_location)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

def get_raw_filename(path):
    (dirname, filename) = os.path.split(path)
    (base, ext) = os.path.splitext(filename)
    return base

def merge_models(model_0, model_1, model_2, multiplier, output, device, safe_tensors=False,
                 interp_method="Weighted sum", save_as_half=False):
    print(f'Start Merge Model {model_0} with {model_1} at an alpha of {str(multiplier)} on Device {device}')
    def weighted_sum(theta0, theta1, alpha):
        return ((1 - alpha) * theta0) + (alpha * theta1)

    def get_difference(theta1, theta2):
        return theta1 - theta2

    def add_difference(theta0, theta1_2_diff, alpha):
        return theta0 + (alpha * theta1_2_diff)

    result_is_inpainting_model = False

    theta_funcs = {
        "Weighted sum": (None, weighted_sum),
        "Add difference": (get_difference, add_difference),
    }
    theta_func1, theta_func2 = theta_funcs[interp_method]


    if theta_func1 and not model_2:
        print("Failed: Interpolation method requires a tertiary model.")

    theta_1 = read_state_dict(model_1, map_location=device)

    if theta_func1:
        print(f"Loading {model_2}...")
        theta_2 = read_state_dict(model_2, map_location=device)

        for key in tqdm.tqdm(theta_1.keys()):
            if 'model' in key:
                if key in theta_2:
                    t2 = theta_2.get(key, torch.zeros_like(theta_1[key]))
                    theta_1[key] = theta_func1(theta_1[key], t2)
                else:
                    theta_1[key] = torch.zeros_like(theta_1[key])
        del theta_2

    theta_0 = read_state_dict(model_0, map_location=device)



    chckpoint_dict_skip_on_merge = ["cond_stage_model.transformer.text_model.embeddings.position_ids"]

    for key in tqdm.tqdm(theta_0.keys()):
        if 'model' in key and key in theta_1:

            if key in chckpoint_dict_skip_on_merge:
                continue

            a = theta_0[key]
            b = theta_1[key]

            # this enables merging an inpainting model (A) with another one (B);
            # where normal model would have 4 channels, for latenst space, inpainting model would
            # have another 4 channels for unmasked picture's latent space, plus one channel for mask, for a total of 9
            if a.shape != b.shape and a.shape[0:1] + a.shape[2:] == b.shape[0:1] + b.shape[2:]:
                if a.shape[1] == 4 and b.shape[1] == 9:
                    raise RuntimeError("When merging inpainting model with a normal one, A must be the inpainting model.")

                assert a.shape[1] == 9 and b.shape[1] == 4, f"Bad dimensions for merged layer {key}: A={a.shape}, B={b.shape}"

                theta_0[key][:, 0:4, :, :] = theta_func2(a[:, 0:4, :, :], b, multiplier)
                result_is_inpainting_model = True
            else:
                theta_0[key] = theta_func2(a, b, multiplier)

            if save_as_half:
                theta_0[key] = theta_0[key].half()

    # I believe this part should be discarded, but I'll leave it for now until I am sure
    for key in theta_1.keys():
        if 'model' in key and key not in theta_0:

            if key in chckpoint_dict_skip_on_merge:
                continue

            theta_0[key] = theta_1[key]
            if save_as_half:
                theta_0[key] = theta_0[key].half()
    del theta_1

    print("Saving...")

    checkpoint_format = '.safetensors' if safe_tensors else '.ckpt'

    filename = \
        get_raw_filename(model_0) + '_' + str(round(1-multiplier, 2)) + '-' + \
        get_raw_filename(model_1) + '_' + str(round(multiplier, 2)) + '-' + \
        interp_method.replace(" ", "_") + \
        '-merged.' + \
        ("inpainting." if result_is_inpainting_model else "") + \
        checkpoint_format
    output_file = os.path.join(gs.system.custom_models_dir, filename)

    (base, ext) = os.path.splitext(model_0)
    source_yaml = base + '.yaml'
    (base, ext) = os.path.splitext(output_file)
    destination_yaml = base + '.yaml'
    shutil.copyfile(source_yaml, destination_yaml)


    if safe_tensors:
        safetensors.torch.save_file(theta_0, output_file, metadata={"format": "pt"})
    else:
        torch.save(theta_0, output_file)
    del theta_0
    print("Saved")


def merge_models_(model_0, model_1, alpha, output, device, safe_tensors=False):

    print(f'Start Merge Model {model_0} with {model_1} at an alpha of {str(alpha)} on Device {device}')

    if 'safetensors' in model_0:
        model_0 = safetensors.torch.load_file(model_0, device=device)
    else:
        model_0 = torch.load(model_0, map_location=device)
    if 'safetensors' in model_1:
        model_1 = safetensors.torch.load_file(model_1, device=device)
    else:
        model_1 = torch.load(model_1, map_location=device)
    #model_1 = torch.load(model_1, map_location=device)

    theta_0 = model_0["state_dict"] if "state_dict" in model_0 else model_0
    theta_1 = model_1["state_dict"] if "state_dict" in model_1 else model_1



    print('Start Stage 1')
    for key in tqdm.tqdm(theta_0.keys(), desc="Stage 1/2"):
        if "model" in key and key in theta_1:
            theta_0[key] = (1 - alpha) * theta_0[key] + alpha * theta_1[key]

    print('Start Stage 2')
    for key in tqdm.tqdm(theta_1.keys(), desc="Stage 2/2"):
        if "model" in key and key not in theta_0:
            theta_0[key] = theta_1[key]

    print("Saving...")
    output_file = os.path.join(gs.system.custom_models_dir,f'{output}-{str(alpha)[2:] + "0"}')
    if safe_tensors:
        output_file = output_file+ '.safetensors'
        safetensors.torch.save_file(theta_0, output_file)
    else:
        output_file = output_file+ '.ckpt'
        torch.save({"state_dict": theta_0}, output_file)

    print("Done!")
    del model_0
    del model_1
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
