import gc
import time

import torch
from omegaconf import OmegaConf
from torch import nn

#from backend.deforum.six.model_load import make_linear_decode
from backend.singleton import singleton
from backend.devices import choose_torch_device
from ldm.util import instantiate_from_config

gs = singleton

model_types = ['sd','inpaint', 'custom']

def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode
def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def kill_model(model, device):
    print('kill:', model)
    if device != 'cpu':
        try:
            gs.models[model].to('cpu')
        except:
            pass
    del gs.models[model]
    torch_gc()

def cleanup_models(target_type, device='cuda', target_model=None):
    print(target_type, target_model)
    if target_type == 'custom':
        print('its custom', gs.models['custom_model_name'])
        if gs.models['custom_model_name'] != target_model:
            for key in gs.models:
                print(key)
            if 'sd' in gs.models:
                print('go kill')
                kill_model(model='sd', device=device)
    else:
        for type in model_types:
            if type != target_type:
                if type in gs.models:
                    print(type)
                    kill_model(type, device)


def load_inpaint_model(device='cuda'):
    cleanup_models(target_type='sd', device=device)
    """Load and initialize the model from configuration variables passed at object creation time"""
    if "inpaint" not in gs.models:
        weights = 'models/sd-v1-5-inpaint.ckpt'
        config = 'configs/stable-diffusion/inpaint.yaml'
        embedding_path = None

        config = OmegaConf.load(config)

        gs.models["inpaint"] = instantiate_from_config(config.model)

        gs.models["inpaint"].load_state_dict(torch.load(weights)["state_dict"], strict=False)


        gs.models["inpaint"].half().to(device)

def load_custom_model(model, device='cuda'):
    cleanup_models(target_type='custom', device=device, target_model=model)
    """Load and initialize the model from configuration variables passed at object creation time"""
    if "sd" not in gs.models:
        if gs.models['custom_model_name'] != model:
            gs.models['custom_model_name'] = model
            gs.system.sdPath = model


def load_model_from_config(config, ckpt, device='cuda', verbose=False):
    config = 'configs/stable-diffusion/v1-inference-a.yaml'
    #ckpt = gs.system.sdPath
    config = OmegaConf.load(config)
    config.model.params.cond_stage_config.params.T = 0
    config.model.params.cond_stage_config.params.lr = 0.0
    config.model.params.cond_stage_config.params.aesthetic_embedding_path = (
        "None"
    )
    if "sd" not in gs.models:
        print(f"Loading model from {ckpt}")
        #if "sd" not in gs.models:
        pl_sd = torch.load(ckpt, map_location="cpu")
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        sd = pl_sd["state_dict"]
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        #model.half()
        model.half().to("cuda")

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode

        autoencoder_version = "sd-v1" #TODO this will be different for different models
        model.linear_decode = make_linear_decode(autoencoder_version, device)

        torch_gc()
        del pl_sd
        del sd
        del m, u
        return model

"""def _load_model_from_config(config, ckpt, device='cuda'):
    print(f'>> Loading model from {ckpt}')

    # for usage statistics
    device_type = choose_torch_device()
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    tic = time.time()

    # this does the work

    if "sd" not in gs.models:

        pl_sd = torch.load(ckpt, map_location='cpu')
        sd = pl_sd['state_dict']
        gs.models["sd"] = instantiate_from_config(config.model)
        m, u = gs.models["sd"].load_state_dict(sd, strict=False)

        # if self.full_precision:
        #    print(
        #        '>> Using slower but more accurate full-precision math (--full_precision)'
        #    )
        # else:
        #    print(
        #        '>> Using half precision math. Call with --full_precision to use more accurate but VRAM-intensive full precision.'
        #    )
        gs.models["sd"].half()
        gs.models["sd"].to(device)
        #model.eval()
        del m, u
        del pl_sd
        # usage statistics
        toc = time.time()
        print(
            f'>> Model loaded in', '%4.2fs' % (toc - tic)
        )
        if device_type == 'cuda':
            print(
                '>> Max VRAM used to load the model:',
                '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
                '\n>> Current VRAM usage:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
                )
        #return model"""
