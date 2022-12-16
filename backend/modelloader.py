import os
import base64
import sys
from omegaconf import OmegaConf
import torch
from ldm.util import instantiate_from_config

from backend.singleton import singleton

gs = singleton
gs.models = {}


def load_gfpgan():
    model_name = 'GFPGANv1.3'
    model_path = os.path.join(gs.system.gfpgan_model_file)
    if not os.path.isfile(model_path):
        raise Exception("GFPGAN model not found at path " + model_path)

    sys.path.append(os.path.abspath(gs.system.gfpgan_dir))
    from gfpgan import GFPGANer

    if gs.system.gfpgan_cpu or gs.system.extra_models_cpu:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
                            device=torch.device('cpu'))
    elif gs.system.extra_models_gpu:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
                            device=torch.device(f'cuda:{gs.system.extra_models_gpu}'))
    else:
        instance = GFPGANer(model_path=model_path, upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None,
                            device=torch.device(f'cuda:{gs.system.gpu}'))
    return instance


def load_realesrgan(model_name: str):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    realesrgan_models = {
        'RealESRGAN_x4plus': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
        'RealESRGAN_x4plus_anime_6B': RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32,
                                              scale=4)
    }

    model_path = ''
    if model_name == 'RealESRGAN_x4plus':
        model_path = gs.system.realesrgan_model_file
    if model_name == 'RealESRGAN_x4plus_anime_6B':
        model_path = gs.system.realesrgan_anime_model_file

    #model_path = os.path.join(gs.system.realesrgan_dir, model_path)
    if not os.path.exists(model_path):
        raise Exception("Model not found at path " + model_path)

    sys.path.append(os.path.abspath(gs.system.realesrgan_dir))
    from realesrgan import RealESRGANer

    if gs.system.realesrgan_cpu or gs.system.extra_models_cpu:
        instance = RealESRGANer(scale=2,
                                model_path=model_path,
                                model=realesrgan_models[model_name],
                                pre_pad=0,
                                half=False)  # cpu does not support half
        instance.device = torch.device('cpu')
        instance.model.to('cpu')
    elif gs.system.extra_models_gpu:
        instance = RealESRGANer(scale=2,
                                model_path=model_path,
                                model=realesrgan_models[model_name],
                                pre_pad=0,
                                half=gs.diffusion.fullPrecision,
                                device=torch.device(f'cuda:{gs.system.extra_models_gpu}'))
    else:
        instance = RealESRGANer(scale=2,
                                model_path=model_path,
                                model=realesrgan_models[model_name],
                                pre_pad=0,
                                half=gs.diffusion.fullPrecision,
                                device=torch.device(f'cuda:{str(gs.system.gpu)}'))
    instance.model.name = model_name

    return instance


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    print(m[0])
    print(u[0])

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def load_upscaler(use_gfpgan=False, use_realesrgan=False, realesrgan_model="RealESRGAN_x4plus"):
    if use_gfpgan:
        if "GFPGAN" in gs.models:
            print("GFPGAN already loaded")
        else:
            # Load GFPGAN
            if os.path.exists(gs.system.gfpgan_dir):
                try:
                    gs.models["GFPGAN"] = load_gfpgan()
                    print("Loaded GFPGAN")
                except Exception:
                    import traceback
                    print("Error loading GFPGAN:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
            else:
                if "GFPGAN" in gs.models:
                    del gs.models["GFPGAN"]

    if use_realesrgan:
        if "RealESRGAN" in gs.models and gs.models["RealESRGAN"].model.name == realesrgan_model:
            print("RealESRGAN already loaded")
        else:
            # Load RealESRGAN
            try:
                # We first remove the variable in case it has something there,
                # some errors can load the model incorrectly and leave things in memory.
                del gs.models["RealESRGAN"]
            except KeyError:
                pass

            if os.path.exists(gs.system.realesrgan_dir):
                # st.session_state is used for keeping the models in memory across multiple pages or runs.
                gs.models["RealESRGAN"] = load_realesrgan(realesrgan_model)
                print("Loaded RealESRGAN with model " + gs.models["RealESRGAN"].model.name)

    else:
        if "RealESRGAN" in gs.models:
            del gs.models["RealESRGAN"]

def load_models(continue_prev_run=False, use_gfpgan=False, use_realesrgan=False, realesrgan_model="RealESRGAN_x4plus"):
    """Load the different models. We also reuse the models that are already in memory to speed things up instead of loading them again. """

    print("Loading models.")

    # Generate random run ID
    # Used to link runs linked w/ continue_prev_run which is not yet implemented
    # Use URL and filesystem safe version just in case.
    gs.run_id = base64.urlsafe_b64encode(
        os.urandom(6)
    ).decode("ascii")

    # check what models we want to use and if the they are already loaded.
    # extern now for be able to load just those models
    load_upscaler(use_gfpgan, use_realesrgan, realesrgan_model)


    if "model" in gs.models:
        print("Model already loaded")
    else:
        config = OmegaConf.load(gs.system.sdInference)
        gs.models["model"] = load_model_from_config(config, gs.system.sd_model_file)

        gs.device = torch.device(
            f"cuda:{gs.system.gpu}") if torch.cuda.is_available() else torch.device("cpu")

        gs.models["model"].half().to(gs.device)

        print("Model loaded.")


def load_depth_model(anim_args, model_path, device):
    if "depth_model" in gs.models:
        print("Using depth model from cache")
    else:
        gs.models["depth_model"] = DepthModel(device)
        gs.models["depth_model"].load_midas(model_path)
    if anim_args.midas_weight < 1.0:
        gs.models["depth_model"].load_adabins()
