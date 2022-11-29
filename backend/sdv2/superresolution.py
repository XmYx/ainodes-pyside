import os.path
import sys
import torch
import numpy as np

from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything

from ldm_v2.models.diffusion.ddim import DDIMSampler
from ldm_v2.models.diffusion.ddpm import LatentUpscaleDiffusion, LatentUpscaleFinetuneDiffusion
from ldm_v2.util import exists, instantiate_from_config

from backend.singleton import singleton

gs = singleton

torch.set_grad_enabled(False)

def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)
    print('model_loaded')
    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('sr device', device)
    model = model.to(device)
    sampler = DDIMSampler(model)
    return sampler


def make_batch_sd(
        image,
        txt,
        device,
        num_samples=1,
):
    image = np.array(image.convert("RGB"))
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
    batch = {
        "lr": rearrange(image, 'h w c -> 1 c h w'),
        "txt": num_samples * [txt],
    }
    batch["lr"] = repeat(batch["lr"].to(device=device), "1 ... -> n ...", n=num_samples)
    return batch


def make_noise_augmentation(model, batch, noise_level=None):
    x_low = batch[model.low_scale_key]
    x_low = x_low.to(memory_format=torch.contiguous_format).float()
    x_aug, noise_level = model.low_scale_model(x_low, noise_level)
    return x_aug, noise_level


def paint(sampler, image, prompt, seed, scale, h, w, steps, num_samples=1, callback=None, eta=0., noise_level=None):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model
    seed_everything(seed)
    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, model.channels, h , w)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad(),\
            torch.autocast("cuda"):
        batch = make_batch_sd(image, txt=prompt, device=device, num_samples=num_samples)
        c = model.cond_stage_model.encode(batch["txt"])
        c_cat = list()
        if isinstance(model, LatentUpscaleFinetuneDiffusion):
            for ck in model.concat_keys:
                cc = batch[ck]
                if exists(model.reshuffle_patch_size):
                    assert isinstance(model.reshuffle_patch_size, int)
                    cc = rearrange(cc, 'b c (p1 h) (p2 w) -> b (p1 p2 c) h w',
                                   p1=model.reshuffle_patch_size, p2=model.reshuffle_patch_size)
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)
            # cond
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}
        elif isinstance(model, LatentUpscaleDiffusion):
            x_augment, noise_level = make_noise_augmentation(model, batch, noise_level)
            cond = {"c_concat": [x_augment], "c_crossattn": [c], "c_adm": noise_level}
            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [x_augment], "c_crossattn": [uc_cross], "c_adm": noise_level}
        else:
            raise NotImplementedError()

        shape = [model.channels, h, w]
        samples, intermediates = sampler.sample(
            steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
            callback=callback
        )
    with torch.no_grad():
        x_samples_ddim = model.decode_first_stage(samples)
    result = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    print(f"upscaled image shape: {result.shape}")
    return result

def t_callback(**args):
    print(args)

def run_sr(image_list , target_h, target_w, prompt, seed, num_samples, scale, steps, eta, noise_level):

    print('run_sr', locals())

    if os.path.isfile(gs.system.sdPath) and os.path.isfile(gs.system.sdv2_upscale_inference):
        print('ready to run upscale 2.0')
        sampler = initialize_model(gs.system.sdv2_upscale_inference, gs.system.sdPath)

        for image in image_list:
            if image:
                image_path = image
                image = Image.open(image)
                w, h = image.size
                print(f"loaded input image of size ({w}, {h})")
                width, height = map(lambda x: x - x % 64, (target_w, target_h))  # resize to integer multiple of 64
                image = image.resize((width, height))
                print(f"resized input image to size ({width}, {height} (w, h))")

                if not isinstance(sampler.model, LatentUpscaleDiffusion):
                    # TODO: make this work for all models
                    noise_level = None

                sampler.make_schedule(steps, ddim_eta=eta, verbose=True)

                result = paint(
                    sampler=sampler,
                    image=image,
                    prompt=prompt,
                    seed=seed if len(seed) > 0 else -1,
                    scale=scale,
                    h=height, w=width, steps=steps,
                    num_samples=num_samples,
                    callback=t_callback,
                    noise_level=noise_level,
                    eta=eta
                )

                for image in result:
                    outpath = image_path + '.enhanced.png'
                    image = image.convert("RGBA")
                    image.save(outpath)
