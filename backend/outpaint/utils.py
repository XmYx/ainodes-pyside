
from sys import path as sys_path
from os import path as os_path

sys_path.append('./stable-diffusion')


import torch
from numpy import float16, uint8
from numpy import array as np_array
from PIL import Image
from omegaconf import OmegaConf

from einops import rearrange, repeat

from itertools import islice

from ldm.util import instantiate_from_config


def load_config_and_model(config_path, ckpt, device, verbose=False):
    config = OmegaConf.load(config_path)
    print(f"Loading model from {ckpt}")
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

    model = model.half().to(device)
    model.eval()
    return [config, model]


def load_inpainting_config_and_model(config_path, model_path, device):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    model = model.half().to(device)
    return [config, model]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())



def get_prompts_data (opt):

    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        return [opt.n_samples * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            prompts_data = f.read().splitlines()
            return list(chunk(prompts_data, opt.n_samples))




#class CFGDenoiser(torch.nn.Module):
#    def __init__(self, model):
#        super().__init__()
#        self.inner_model = model
#
#    def forward(self, x, sigma, uncond, cond, cond_scale):
#        x_in = torch.cat([x] * 2)
#        sigma_in = torch.cat([sigma] * 2)
#        cond_in = torch.cat([uncond, cond])
#        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
#        return uncond + (cond - uncond) * cond_scale



def split_weighted_subprompts(text):
    remaining  = len(text)
    subprompts = []
    weights    = []
    while remaining > 0:
        if "::" in text:
            idx = text.index("::") # first occurrence from start
            # grab up to index as sub-prompt
            subprompt = text[:idx].strip()
            remaining -= idx
            # remove from main text
            text = text[idx+2:]
            # find value for weight
            if " " in text:
                idx = text.index(" ") # first occurence
            else: # no space, read to end
                idx = len(text)
            if idx != 0:
                try:
                    weight = float(text[:idx])
                except: # couldn't treat as float
                    print(f"Warning: '{text[:idx]}' is not a value, are you missing a space?")
                    idx = -1
                    weight = 1.0
            else: # no value found
                weight = 1.0
            # remove from main text
            remaining -= idx
            text = text[idx+1:]
            # append the sub-subprompt and its weight
            subprompts.append(subprompt)
            weights.append(weight)
        else: # no : found
            text = text.strip()
            if len(text) > 0: # there is still text though
                # take remainder as weight 1
                subprompts.append(text)
                weights.append(1.0)
            remaining = 0
    return subprompts, weights



def get_conditionings(model, prompts, opt):
    unconditional_conditioning = model.get_learned_conditioning(opt.n_samples * [""])
    if isinstance(prompts, tuple):
        prompts = list(prompts)

    # weighted sub-prompts
    subprompts, weights = split_weighted_subprompts(prompts[0])  # FIXME [0]
    if len(subprompts) > 1:
        conditioning = torch.zeros_like(unconditional_conditioning)
        totalWeight  = sum(weights)
        for i in range(0, len(subprompts)):
            weight = weights[i] / totalWeight
            conditioning = torch.add(
                conditioning, model.get_learned_conditioning(subprompts[i]), alpha=weight)
    else:
        conditioning = model.get_learned_conditioning(prompts)

    return [unconditional_conditioning, conditioning]




def image_path_to_torch(path, device):
    assert os_path.isfile(path)
    image = Image.open(path).convert("RGB")
    source_w, source_h = image.size
    print(f"loaded input image of size ({source_w}, {source_h}) from {path}")

    w, h = map(lambda x: x - x % 64, (source_w, source_h))  # resize to integer multiple of 32
    if source_w != w or source_h != h:
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
    image = np_array(image).astype(float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image.half().to(device)  # FIXME to(device) ??


def torch_image_to_latent(model, torch_image, n_samples=1):
    formatted_image = 2.*torch_image - 1.
    if n_samples > 1:
        formatted_image = repeat(
            formatted_image, '1 ... -> b ...', b=n_samples)
    latent_image = model.get_first_stage_encoding(
        model.encode_first_stage(formatted_image))
    return latent_image


# [1, 4, 64, 64] => [1, 3, 512, 512]
def encoded_to_torch_image(model, encoded_image):
    decoded = model.decode_first_stage(encoded_image)
    return torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
    
# [1, 4, 64, 64] => image
def encoded_to_image(model, encoded_image):
    return sampleToImage(encoded_to_torch_image(model, encoded_image)[0])


def sampleToImage (sample):
    sample = 255. * rearrange(sample.cpu().numpy(), 'c h w -> h w c')
    return Image.fromarray(sample.astype(uint8))

