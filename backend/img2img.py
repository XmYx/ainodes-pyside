from sd.kdiffusion_sampler import KDiffusionSampler
import os
import platform
import sys
import time

import cv2
import k_diffusion as K
import numpy as np
import skimage
import torch
import torch.nn as nn
from PIL import Image, ImageFilter, ImageOps
from omegaconf import OmegaConf

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config
from sd.image_processor import process_images, seed_to_int
from sd.kdiffusion_sampler import KDiffusionSampler
from sd.modelloader import load_models
from sd.singleton import singleton

# from scripts.tools.nsp.nsp_pantry import parser

gs = singleton

if "Linux" in platform.platform():
    sys.path.extend([
        '/content/src/taming-transformers',
        '/content/src/clip',
        '/content/deforum-sd-ui-colab',
        '/content/src/k-diffusion',
        '/content/src/pytorch3d-lite',
        '/content/src/AdaBins',
        '/content/src/MiDaS',
        '/content/src/soup',
        '/content/src/Real-ESRGAN'
    ])


def _get_masked_window_rgb(np_mask_grey, hardness=1.):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:, :, c] = hardened[:]
    return np_mask_rgb


# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
    if data.ndim > 2:  # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
            out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
    else:  # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
        out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

    return out_fft


def _ifft2(data):
    if data.ndim > 2:  # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
            out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
    else:  # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
        out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

    return out_ifft


def _get_gaussian_window(width, height, std=3.14, mode=0):
    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))

    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2. - 1.) * window_scale_x
    for y in range(height):
        fy = (y / height * 2. - 1.) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x ** 2 + fy ** 2) * std)
        else:
            window[:, y] = (1 / ((x ** 2 + 1.) * (fy ** 2 + 1.))) ** (
                    std / 3.14)  # hey wait a minute that's not gaussian

    return window


def get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation):
    """
    Explanation:
    Getting good results in/out-painting with stable diffusion can be challenging.
    Although there are simpler effective solutions for in-painting, out-painting can be especially challenging because
    there is no color data in the masked area to help prompt the generator. Ideally, even for in-painting we'd like
    work effectively without that data as well. Provided here is my take on a potential solution to this problem.

    By taking a fourier transform of the masked src img we get a function that tells us the presence and orientation
    of each feature scale in the unmasked src. Shaping the init/seed noise for in/outpainting to the same distribution
    of feature scales, orientations, and positions increases output coherence by helping keep features aligned. This
    technique is applicable to any continuous generation task such as audio or video, each of which can be
    conceptualized as a series of out-painting steps where the last half of the input "frame" is erased. For
    multi-channel data such as color or stereo sound the "color tone" or histogram of the seed noise can be
    matched to improve quality (using scikit-image currently) This method is quite robust and has the added benefit
    of being fast independently of the size of the out-painted area. The effects of this method include things like
    helping the generator integrate the pre-existing view distance and camera angle.

    Carefully managing color and brightness with histogram matching is also essential to achieving good coherence.

    noise_q controls the exponent in the fall-off of the distribution can be any positive number, lower values means
    higher detail (range > 0, default 1.) color_variation controls how much freedom is allowed for the colors/palette
    of the out-painted area (range 0..1, default 0.01) This code is provided as is under the Unlicense
    (https://unlicense.org/) Although you have no obligation to do so, if you found this code helpful please
    find it in your heart to credit me [parlance-zz].

    Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
    This code is part of a new branch of a discord bot I am working on integrating with diffusers
    (https://github.com/parlance-zz/g-diffuser-bot)

    """

    global DEBUG_MODE
    global TMP_ROOT_PATH

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    np_src_image = _np_src_image[:] * (1. - np_mask_rgb)
    np_mask_grey = (np.sum(np_mask_rgb, axis=2) / 3.)
    np_src_grey = (np.sum(np_src_image, axis=2) / 3.)
    all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1. - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(
        _np_src_image) * np_mask_rgb  # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black, we get better results from fft by filling the average unmasked color
    # windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) / (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    # _save_debug_img(windowed_image, "windowed_src_img")

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    # _save_debug_img(src_dist, "windowed_src_dist")

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = (np.sum(noise_rgb, axis=2) / 3.)
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1. - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (
            src_dist ** noise_q) * src_phase  # perform the actual shaping

    brightness_variation = 0.  # color_variation # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.) - brightness_variation * 2.

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(shaped_noise[img_mask, :] ** 1.,
                                                                  contrast_adjusted_np_src[ref_mask, :], multichannel=1)
    shaped_noise = _np_src_image[:] * (1. - np_mask_rgb) + shaped_noise * np_mask_rgb
    # _save_debug_img(shaped_noise, "shaped_noise")

    matched_noise = np.zeros((width, height, num_channels))
    matched_noise = shaped_noise[:]
    # matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:], _np_src_image[ref_mask,:], channel_axis=1)
    # matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb

    # _save_debug_img(matched_noise, "matched_noise")

    # todo: color_variation doesnt have to be a single number, the overall color tone of the out-painted area could be
    #  param controlled

    return np.clip(matched_noise, 0., 1.)


def resize_image(resize_mode, im, width, height):
    lanczos = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
    if resize_mode == 0:
        res = im.resize((width, height), resample=lanczos)
    elif resize_mode == 1:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=lanczos)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=lanczos)
        res = Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(resized.resize((width, fill_height), box=(0, 0, width, 0)), box=(0, 0))
            res.paste(resized.resize((width, fill_height), box=(0, resized.height, width, resized.height)),
                      box=(0, fill_height + src_h))
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(resized.resize((fill_width, height), box=(0, 0, 0, height)), box=(0, 0))
            res.paste(resized.resize((fill_width, height), box=(resized.width, 0, resized.width, height)),
                      box=(fill_width + src_w, 0))

    return res


# used OK 20.09.22
def load_sd_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd


# used OK 20.09.22
def load_model_from_config(config, ckpt, verbose=False):
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

    model.cuda()
    model.eval()
    return model


# Loads Stable Diffusion model by name
def load_sd_model(model_name: str) -> [any, any, any, any, any]:
    ckpt_path = gs.defaults.general.default_model_path
    if model_name != gs.defaults.general.default_model:
        ckpt_path = os.path.join("models", "custom", f"{model_name}.ckpt")

    if gs.defaults.general.optimized:
        config = OmegaConf.load(gs.defaults.general.optimized_config)

        sd = load_sd_from_config(ckpt_path)
        li, lo = [], []
        for key, v_ in sd.items():
            sp = key.split('.')
            if (sp[0]) == 'model':
                if 'input_blocks' in sp:
                    li.append(key)
                elif 'middle_block' in sp:
                    li.append(key)
                elif 'time_embed' in sp:
                    li.append(key)
                else:
                    lo.append(key)
        for key in li:
            sd['model1.' + key[6:]] = sd.pop(key)
        for key in lo:
            sd['model2.' + key[6:]] = sd.pop(key)

        device = torch.device(f"cuda:{gs.defaults.general.gpu}") \
            if torch.cuda.is_available() else torch.device("cpu")

        model = instantiate_from_config(config.modelUNet)
        _, _ = model.load_state_dict(sd, strict=False)
        model.cuda()
        model.eval()
        model.turbo = gs.defaults.general.optimized_turbo

        modelCS = instantiate_from_config(config.modelCondStage)
        _, _ = modelCS.load_state_dict(sd, strict=False)
        modelCS.cond_stage_model.device = device
        modelCS.eval()

        modelFS = instantiate_from_config(config.modelFirstStage)
        _, _ = modelFS.load_state_dict(sd, strict=False)
        modelFS.eval()

        del sd

        if not gs.defaults.general.no_half:
            model = model.half()
            modelCS = modelCS.half()
            modelFS = modelFS.half()

        return config, device, model, modelCS, modelFS
    else:
        config = OmegaConf.load(gs.defaults.general.default_model_config)
        model = load_model_from_config(config, ckpt_path)

        device = torch.device(f"cuda:{gs.defaults.general.gpu}") \
            if torch.cuda.is_available() else torch.device("cpu")
        model = (model if gs.defaults.general.no_half
                 else model.half()).to(device)

        return config, device, model, None, None


# used OK 20.09.22
class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised


def run_img2img_json(prompt: str = '',
                     init_info: any = None,
                     init_info_mask: any = None,
                     mask_mode: int = 0,
                     mask_blur_strength: int = 3,
                     mask_restore: bool = False,
                     ddim_steps: int = 50,
                     sampler_name: str = 'DDIM',
                     n_iter: int = 1,
                     cfg_scale: float = 7.5,
                     denoising_strength: float = 0.8,
                     seed: int = -1,
                     noise_mode: int = 0,
                     find_noise_steps: str = "",
                     height: int = 512,
                     width: int = 512,
                     resize_mode: int = 0,
                     fp=None,
                     variant_amount: float = None,
                     variant_seed: int = None,
                     ddim_eta: float = 0.0,
                     write_info_files: bool = True,
                     realesrgan_model: str = "realesrgan_x4plus_anime_6B",
                     separate_prompts: bool = False,
                     normalize_prompt_weights: bool = True,
                     save_individual_images: bool = True,
                     save_grid: bool = True,
                     group_by_prompt: bool = True,
                     save_as_jpg: bool = True,
                     use_gfpgan: bool = True,
                     use_realesrgan: bool = True,
                     loopback: bool = False,
                     random_seed_loopback: bool = False
                     ):
    load_models()
    outpath = gs.defaults.general.outdir_img2img or gs.defaults.general.outdir or "outputs/img2img-samples"
    # err = False
    # loopback = False
    # skip_save = False
    seed = seed_to_int(seed)

    batch_size = 1

    # prompt_matrix = 0
    # normalize_prompt_weights = 1 in toggles
    # loopback = 2 in toggles
    # random_seed_loopback = 3 in toggles
    # skip_save = 4 not in toggles
    # save_grid = 5 in toggles
    # sort_samples = 6 in toggles
    # write_info_files = 7 in toggles
    # write_sample_info_to_log_file = 8 in toggles
    # jpg_sample = 9 in toggles
    # use_gfpgan = 10 in toggles
    # use_realesrgan = 11 in toggles

    if sampler_name == 'PLMS':
        sampler = PLMSSampler(gs.models["model"])
    elif sampler_name == 'DDIM':
        sampler = DDIMSampler(gs.models["model"])
    elif sampler_name == 'k_dpm_2_a':
        sampler = KDiffusionSampler(gs.models["model"], 'dpm_2_ancestral')
    elif sampler_name == 'k_dpm_2':
        sampler = KDiffusionSampler(gs.models["model"], 'dpm_2')
    elif sampler_name == 'k_euler_a':
        sampler = KDiffusionSampler(gs.models["model"], 'euler_ancestral')
    elif sampler_name == 'k_euler':
        sampler = KDiffusionSampler(gs.models["model"], 'euler')
    elif sampler_name == 'k_heun':
        sampler = KDiffusionSampler(gs.models["model"], 'heun')
    elif sampler_name == 'k_lms':
        sampler = KDiffusionSampler(gs.models["model"], 'lms')
    else:
        raise Exception("Unknown sampler: " + sampler_name)

    def process_init_mask(init_mask: Image):
        if init_mask.mode == "RGBA":
            init_mask = init_mask.convert('RGBA')
            background = Image.new('RGBA', init_mask.size, (0, 0, 0))
            init_mask = Image.alpha_composite(background, init_mask)
            init_mask = init_mask.convert('RGB')
        return init_mask

    init_img = init_info
    init_mask = None
    if mask_mode == 0:
        if init_info_mask:
            init_mask = process_init_mask(init_info_mask)
    elif mask_mode == 1:
        if init_info_mask:
            init_mask = process_init_mask(init_info_mask)
            init_mask = ImageOps.invert(init_mask)
    elif mask_mode == 2:
        init_img_transparency = init_img.split()[-1].convert('L')  # .point(lambda x: 255 if x > 0 else 0, mode='1')
        init_mask = init_img_transparency
        init_mask = init_mask.convert("RGB")
        init_mask = resize_image(resize_mode, init_mask, width, height)
        init_mask = init_mask.convert("RGB")

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(denoising_strength * ddim_steps)

    if init_mask is not None and (noise_mode == 2 or noise_mode == 3) and init_img is not None:
        noise_q = 0.99
        color_variation = 0.0
        mask_blend_factor = 1.0

        np_init = (np.asarray(init_img.convert("RGB")) / 255.0).astype(np.float64)  # annoyingly complex mask fixing
        np_mask_rgb = 1. - (np.asarray(ImageOps.invert(init_mask).convert("RGB")) / 255.0).astype(np.float64)
        np_mask_rgb -= np.min(np_mask_rgb)
        np_mask_rgb /= np.max(np_mask_rgb)
        np_mask_rgb = 1. - np_mask_rgb
        np_mask_rgb_hardened = 1. - (np_mask_rgb < 0.99).astype(np.float64)
        blurred = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., multichannel=2, truncate=32.)
        blurred2 = skimage.filters.gaussian(np_mask_rgb_hardened[:], sigma=16., multichannel=2, truncate=32.)
        # np_mask_rgb_dilated = np_mask_rgb + blurred  # fixup mask todo: derive magic constants
        # np_mask_rgb = np_mask_rgb + blurred
        np_mask_rgb_dilated = np.clip((np_mask_rgb + blurred2) * 0.7071, 0., 1.)
        np_mask_rgb = np.clip((np_mask_rgb + blurred) * 0.7071, 0., 1.)

        noise_rgb = get_matched_noise(np_init, np_mask_rgb, noise_q, color_variation)
        blend_mask_rgb = np.clip(np_mask_rgb_dilated, 0., 1.) ** (mask_blend_factor)
        noised = noise_rgb[:]
        blend_mask_rgb **= (2.)
        noised = np_init[:] * (1. - blend_mask_rgb) + noised * blend_mask_rgb

        np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.
        ref_mask = np_mask_grey < 1e-3

        all_mask = np.ones((height, width), dtype=bool)
        noised[all_mask, :] = skimage.exposure.match_histograms(noised[all_mask, :] ** 1., noised[ref_mask, :],
                                                                multichannel=1)

        init_img = Image.fromarray(np.clip(noised * 255., 0., 255.).astype(np.uint8), mode="RGB")
        # st.session_state["editor_image"].image(init_img)  # debug

    def init():
        image = init_img.convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        mask_channel = None
        if init_mask:
            alpha = resize_image(resize_mode, init_mask, width // 8, height // 8)
            mask_channel = alpha.split()[-1]

        mask = None
        if mask_channel is not None:
            mask = np.array(mask_channel).astype(np.float32) / 255.0
            mask = (1 - mask)
            mask = np.tile(mask, (4, 1, 1))
            mask = mask[None].transpose(0, 1, 2, 3)
            mask = torch.from_numpy(mask).to(gs.device)

        if gs.defaults.general.optimized:
            gs.models.modelFS.to(gs.device)

        init_image = 2. * image - 1.
        init_image = init_image.to(gs.device)
        init_latent = (
            gs.models["model"] if not gs.defaults.general.optimized else gs.models.modelFS).get_first_stage_encoding(
            (gs.models["model"] if not
            gs.defaults.general.optimized else gs.models.modelFS).encode_first_stage(
                init_image))  # move to latent space

        if gs.defaults.general.optimized:
            mem = torch.cuda.memory_allocated() / 1e6
            gs.models.modelFS.to("cpu")
            while (torch.cuda.memory_allocated() / 1e6 >= mem):
                time.sleep(1)

        return init_latent, mask,

    def sample(init_data, x, conditioning, unconditional_conditioning, sampler_name):
        t_enc_steps = t_enc
        obliterate = False
        if ddim_steps == t_enc_steps:
            t_enc_steps = t_enc_steps - 1
            obliterate = True

        if sampler_name != 'DDIM':
            x0, z_mask = init_data

            sigmas = sampler.model_wrap.get_sigmas(ddim_steps)
            noise = x * sigmas[ddim_steps - t_enc_steps - 1]

            xi = x0 + noise

            # Obliterate masked image
            if z_mask is not None and obliterate:
                random = torch.randn(z_mask.shape, device=xi.device)
                xi = (z_mask * noise) + ((1 - z_mask) * xi)

            sigma_sched = sigmas[ddim_steps - t_enc_steps - 1:]
            model_wrap_cfg = CFGMaskedDenoiser(sampler.model_wrap)
            samples_ddim = K.sampling.__dict__[f'sample_{sampler.get_sampler_name()}'](model_wrap_cfg, xi, sigma_sched,
                                                                                       extra_args={'cond': conditioning,
                                                                                                   'uncond': unconditional_conditioning,
                                                                                                   'cond_scale': cfg_scale,
                                                                                                   'mask': z_mask,
                                                                                                   'x0': x0, 'xi': xi},
                                                                                       disable=False,
                                                                                       callback=None)
        else:

            x0, z_mask = init_data

            sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=0.0, verbose=False)
            z_enc = sampler.stochastic_encode(x0,
                                              torch.tensor([t_enc_steps] * batch_size).to(gs.device))

            # Obliterate masked image
            if z_mask is not None and obliterate:
                random = torch.randn(z_mask.shape, device=z_enc.device)
                z_enc = (z_mask * random) + ((1 - z_mask) * z_enc)

            # decode it
            samples_ddim = sampler.decode(z_enc, conditioning, t_enc_steps,
                                          unconditional_guidance_scale=cfg_scale,
                                          unconditional_conditioning=unconditional_conditioning,
                                          z_mask=z_mask, x0=x0)
        return samples_ddim

    if loopback:
        output_images, info = None, None
        history = []
        initial_seed = None

        do_color_correction = False
        try:
            from skimage import exposure
            do_color_correction = True
        except:
            print("Install scikit-image to perform color correction on loopback")

        for i in range(n_iter):
            if do_color_correction and i == 0:
                correction_target = cv2.cvtColor(np.asarray(init_img.copy()), cv2.COLOR_RGB2LAB)

            output_images, seed, info, stats = process_images(
                outpath=outpath,
                func_init=init,
                func_sample=sample,
                prompt=prompt,
                seed=seed,
                sampler_name=sampler_name,
                save_grid=save_grid,
                batch_size=1,
                n_iter=1,
                steps=ddim_steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                prompt_matrix=separate_prompts,
                use_gfpgan=use_gfpgan,
                use_realesrgan=use_realesrgan,  # Forcefully disable upscaling when using loopback
                realesrgan_model_name=realesrgan_model,
                normalize_prompt_weights=normalize_prompt_weights,
                save_individual_images=save_individual_images,
                init_img=init_img,
                init_mask=init_mask,
                mask_blur_strength=mask_blur_strength,
                mask_restore=mask_restore,
                denoising_strength=denoising_strength,
                noise_mode=noise_mode,
                find_noise_steps=find_noise_steps,
                resize_mode=resize_mode,
                uses_loopback=loopback,
                uses_random_seed_loopback=random_seed_loopback,
                sort_samples=group_by_prompt,
                write_info_files=write_info_files,
                jpg_sample=save_as_jpg
            )

            if initial_seed is None:
                initial_seed = seed

            input_image = init_img
            init_img = output_images[0]

            if do_color_correction and correction_target is not None:
                init_img = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
                    cv2.cvtColor(
                        np.asarray(init_img),
                        cv2.COLOR_RGB2LAB
                    ),
                    correction_target,
                    multichannel=2
                ), cv2.COLOR_LAB2RGB).astype("uint8"))
                if mask_restore is True and init_mask is not None:
                    color_mask = init_mask.filter(ImageFilter.GaussianBlur(mask_blur_strength))
                    color_mask = color_mask.convert('L')
                    source_image = input_image.convert('RGB')
                    target_image = init_img.convert('RGB')

                    init_img = Image.composite(source_image, target_image, color_mask)

            if not random_seed_loopback:
                seed = seed + 1
            else:
                seed = seed_to_int(None)

            denoising_strength = max(denoising_strength * 0.95, 0.1)
            history.append(init_img)

        output_images = history
        seed = initial_seed

    else:
        output_images, seed, info, stats = process_images(
            outpath=outpath,
            func_init=init,
            func_sample=sample,
            prompt=prompt,
            seed=seed,
            sampler_name=sampler_name,
            save_grid=save_grid,
            batch_size=batch_size,
            n_iter=n_iter,
            steps=ddim_steps,
            cfg_scale=cfg_scale,
            width=width,
            height=height,
            prompt_matrix=separate_prompts,
            use_gfpgan=use_gfpgan,
            use_realesrgan=use_realesrgan,
            realesrgan_model_name=realesrgan_model,
            normalize_prompt_weights=normalize_prompt_weights,
            save_individual_images=save_individual_images,
            init_img=init_img,
            init_mask=init_mask,
            mask_blur_strength=mask_blur_strength,
            denoising_strength=denoising_strength,
            noise_mode=noise_mode,
            find_noise_steps=find_noise_steps,
            mask_restore=mask_restore,
            resize_mode=resize_mode,
            uses_loopback=loopback,
            sort_samples=group_by_prompt,
            write_info_files=write_info_files,
            jpg_sample=save_as_jpg
        )

    del sampler
    print(output_images)
    # st.session_state["img2img"]["preview_image"] = st.image(output_images[0])
    return output_images, seed, info, stats
