import math
import re
import time
from random import randint

import torch
from PIL import Image
import requests
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
import os

from torchvision.utils import make_grid
from tqdm import tqdm, trange


from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat

from optimizedSD.optimUtils import logger
from .prompt import get_uc_and_c, split_weighted_subprompts
from .k_samplers import sampler_fn, make_inject_timing_fn
from scipy.ndimage import gaussian_filter

from .callback import SamplerCallback

from .conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from .conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from .model_wrap import CFGDenoiserWithGrad
from backend.torch_gc import torch_gc

from backend.singleton import singleton
from backend.resizeRight import resizeright, interp_methods
import k_diffusion

from ...devices import choose_torch_device

gs = singleton
def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    #if device.type == 'mps':
    #    generator = torch.Generator(device=cpu)
    #    generator.manual_seed(seed)
    #    noise = torch.randn(shape, generator=generator, device=cpu).to(device)
    #    return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=choose_torch_device())
def slerp(val, low, high):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []
    device = choose_torch_device()
    seeds = [seeds]
    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None:# and (len(seeds) > 1 and opts.enable_batch_seeds or opts.eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None
    print(type(seeds))
    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        #if sampler_noises is not None:
        #    cnt = p.sampler.number_of_needed_noises(p)
        #
        #    if opts.eta_noise_seed_delta > 0:
        #        torch.manual_seed(seed + opts.eta_noise_seed_delta)
        #
        #    for j in range(cnt):
        #        sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(device) for n in sampler_noises]

    x = torch.stack(xs).to(device)
    return x
def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def load_img(path, shape=None, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    if shape is not None:
        image = image.resize(shape, resample=Image.Resampling.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
        # Split alpha channel into a mask_image
        red, green, blue, alpha = Image.Image.split(image)
        mask_image = alpha.convert('L')
        image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.

    return image, mask_image

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    
    if isinstance(mask_input, str): # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0, invert_mask=False):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge, 
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image, 
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast
    
    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)

    if invert_mask:
        mask = ( (mask - 0.5) * -1) + 0.5
    
    mask = np.clip(mask,0,1)
    return mask

def generate(args, root, frame = 0, return_latent=False, return_sample=False, return_c=False, step_callback=None, hires=None):
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    try:
        gs.models["sd"].to("cuda")
    except:
        pass

    #HiRes_Fix
    goal_px = 512 * 512
    true_px = args.W * args.H
    scale = math.sqrt(goal_px / true_px)
    new_W = math.ceil(scale * args.W / 64) * 64
    new_H = math.ceil(scale * args.H / 64) * 64
    new_w_int = int(scale * args.W)
    new_h_int = int(scale * args.H)
    t_W = int(new_W - new_w_int) // args.f
    t_H = int(new_H - new_h_int) // args.f
    if hires:
        args.oldH = args.H
        args.oldW = args.W
        args.H = new_H
        args.W = new_W
        args.n_samples = 1
    if gs.model_version == '1.5':
        from ldm.models.diffusion.plms import PLMSSampler
        from ldm.models.diffusion.ddim import DDIMSampler
    elif gs.model_version == '2.0':
        from ldm_v2.models.diffusion.plms import PLMSSampler
        from ldm_v2.models.diffusion.ddim import DDIMSampler

    sampler = PLMSSampler(gs.models["sd"]) if args.sampler == 'plms' else DDIMSampler(gs.models["sd"])
    if gs.model_version in gs.system.gen_one_models or gs.model_resolution == 512:
        print("using old denoiser")
        #k_diffusion.external.CompVisVDenoiser = CompVisDenoiser
        model_wrap = CompVisDenoiser(gs.models["sd"])
        print(gs.model_version, gs.model_resolution)
    elif gs.model_version in gs.system.gen_two_models and gs.model_resolution == 768:
        print("using new denoiser")
        gs.denoiser = 2
        model_wrap = CompVisVDenoiser(gs.models["sd"])


    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            #gs.models["sd"].first_stage_model.to(root.device)
            #args.init_sample.float()
            init_latent = gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image, 
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to("cuda")
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(init_image))  # move to latent space
            init_latent = resizeright.resize(init_latent, scale_factors=None,
                                         out_shape=[init_latent.shape[0], init_latent.shape[1], args.H // 8, args.W // 8],
                                         interp_method=interp_methods.lanczos3, support_sz=None,
                                         antialiasing=True, by_convs=True, scale_tolerance=None,
                                         max_numerator=10, pad_mode='reflect')

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                            init_latent.shape, 
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust,
                            args.invert_mask)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to("cuda")
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
        init_mse_image, mask_image = load_img(args.init_mse_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_mse_image = init_mse_image.to("cuda")
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( args.init_mse_scale != 0 and (args.init_mse_image is None or args.init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    t_enc = int((1.0-args.strength) * args.steps)
    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    args.clamp_schedule = dict(zip(k_sigmas.tolist(), np.linspace(args.clamp_start,args.clamp_stop,args.steps+1)))
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms','ddim'] and gs.model_version == '1.5':
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)
    elif args.sampler in ['plms','ddim'] and gs.model_version == '2.0':
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='quad', verbose=False)

    if args.colormatch_scale != 0:
        assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
        colormatch_image, _ = load_img(args.colormatch_image)
        colormatch_image = colormatch_image.to('cpu')
        del(_)
    else:
        colormatch_image = None
    # Loss functions
    if args.init_mse_scale != 0:
        if args.decode_method == "linear":
            mse_loss_fn = make_mse_loss(gs.models["sd"].linear_decode(gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(init_mse_image.to("cuda")))))
        else:
            mse_loss_fn = make_mse_loss(init_mse_image)
    else:
        mse_loss_fn = None
    if args.colormatch_scale != 0:
        _,_ = get_color_palette(root, args.colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if args.decode_method == "linear":
            grad_img_shape = (int(args.W/args.f), int(args.H/args.f))
            colormatch_image = gs.models["sd"].linear_decode(gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(colormatch_image.to("cuda"))))
            colormatch_image = colormatch_image.to('cpu')
            print("Colormatch Sample Used")
        else:
            grad_img_shape = (args.W, args.H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image, 
                                                  n_colors=args.colormatch_n_colors, 
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=args.ignore_sat_weight)
    else:
        color_loss_fn = None

    if args.clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if args.aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if args.exposure_scale != 0:
        exposure_loss_fn = exposure_loss(args.exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              args.clip_scale],
        [blue_loss_fn,              args.blue_scale],
        [mean_loss_fn,              args.mean_scale],
        [exposure_loss_fn,          args.exposure_scale],
        [var_loss_fn,               args.var_scale],
        [mse_loss_fn,               args.init_mse_scale],
        [color_loss_fn,             args.colormatch_scale],
        [aesthetics_loss_fn,        args.aesthetics_scale]
    ]

    #callback = step_callback if step_callback is not None else None
    callback = SamplerCallback(args=args,
                                root=root,
                                mask=mask,
                                init_latent=init_latent,
                                sigmas=k_sigmas,
                                sampler=sampler,
                                step_callback=step_callback,
                                verbose=True).callback

    clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type, clamp_schedule=args.clamp_schedule)

    args.grad_inject_timing = int(args.grad_inject_timing) if args.grad_inject_timing != 'None' else None
    grad_inject_timing_fn = make_inject_timing_fn(args.grad_inject_timing, model_wrap, args.steps)

    cfg_model = CFGDenoiserWithGrad(model_wrap, 
                                    loss_fns_scales, 
                                    clamp_fn, 
                                    args.gradient_wrt, 
                                    args.gradient_add_to, 
                                    args.cond_uncond_sync,
                                    decode_method=args.decode_method,
                                    grad_inject_timing_fn=grad_inject_timing_fn, # option to use grad in only a few of the steps
                                    grad_consolidate_fn=None, # function to add grad to image fn(img, grad, sigma)
                                    verbose=False)

    results = []
    #gs.x = create_random_tensors([4, args.H // 8, args.W // 8], seeds=(args.seed), seed_resize_from_h=args.H, seed_resize_from_w=args.W )
    with torch.no_grad():
        with precision_scope("cuda"):
            with gs.models["sd"].ema_scope():
                for prompts in data:
                    if isinstance(prompts, tuple):

                        prompts = list(prompts)

                    if args.prompt_weighting:
                        uc, c = get_uc_and_c(prompts, gs.models["sd"], args, frame)
                    else:
                        if args.negative_prompts is not None:
                            print(f"using negative prompts: {args.negative_prompts}")
                            uc = gs.models["sd"].get_learned_conditioning(args.negative_prompts)
                        else:
                            uc = gs.models["sd"].get_learned_conditioning(batch_size * [""])
                        c = gs.models["sd"].get_learned_conditioning(prompts)


                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m", "dpmpp_sde"]:
                        #x = t_enc

                        #gs.x = create_random_tensors([4, args.H // 8, args.W // 8], seeds=(args.seed), seed_resize_from_h=args.oldH, seed_resize_from_w=args.oldW )
                        #else:
                        #    args.H = args.oldH
                        #    args.W = args.oldW
                        torch_gc()

                        samples = sampler_fn(
                            c=c, 
                            uc=uc,
                            args=args, 
                            model_wrap=cfg_model, 
                            init_latent=init_latent, 
                            t_enc=t_enc,
                            device=root.device,
                            cb=callback,
                            verbose=False)
                        if hires:
                            samples = samples[:, :, t_H//2:samples.shape[2]-t_H//2, t_W//2:samples.shape[3]-t_W//2]
                            #print(samples.shape)
                            samples = resizeright.resize(samples, scale_factors=None, out_shape=[samples.shape[0], samples.shape[1], args.oldH // 8, args.oldW // 8],
                                                            interp_method=interp_methods.lanczos3, support_sz=None,
                                                            antialiasing=True, by_convs=True, scale_tolerance=None,
                                                            max_numerator=10, pad_mode='reflect')
                            #samples = torch.nn.functional.interpolate(samples, size=(args.oldH // 8, args.oldW // 8), mode="bilinear")
                            #print(samples.shape)

                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to("cuda"))
                        else:
                            z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device="cuda")
                        if args.sampler == 'ddim':
                            samples = sampler.decode(z_enc, 
                                                     c, 
                                                     t_enc, 
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif args.sampler == 'plms': # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                            conditioning=c,
                                                            batch_size=args.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=args.ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                        else:
                            raise Exception(f"Sampler {args.sampler} not recognised.")

                    
                    if return_latent:
                        results.append(samples.clone())
                    #if hires == False:
                    if not return_latent:
                        try:
                            gs.models["sd"].first_stage_model.to(root.device)
                        except:
                            pass
                        try:
                            gs.models["sd"].cond_stage_model.to("cpu")
                        except:
                            pass
                        try:
                            gs.models["sd"].model.to("cpu")
                        except:
                            pass
                        if not hires:
                            x_samples = [
                                decode_first_stage(gs.models["sd"], samples[i:i + 1].to(root.device))[0].cpu() for i
                                in range(samples.size(0))]
                            x_samples = torch.stack(x_samples).float()
                        else:
                            x_samples = gs.models["sd"].decode_first_stage(samples)
                            #x_samples = samples[0]

                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample is not None:
                            img_original = args.init_sample
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        mask_fullres = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                                                    img_original.shape, 
                                                    args.mask_contrast_adjust, 
                                                    args.mask_brightness_adjust,
                                                    args.invert_mask)
                        mask_fullres = mask_fullres[:,:3,:,:]
                        mask_fullres = repeat(mask_fullres, '1 ... -> b ...', b=batch_size)

                        mask_fullres[mask_fullres < mask_fullres.max()] = 0
                        mask_fullres = gaussian_filter(mask_fullres, args.mask_overlay_blur)
                        mask_fullres = torch.Tensor(mask_fullres).to("cuda")

                        x_samples = img_original * mask_fullres + x_samples * ((mask_fullres * -1.0) + 1)


                    if return_sample:
                        results.append(x_samples.clone())
                    if return_latent == False:
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                        if return_c:
                            results.append(c.clone())
                        if not return_latent:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image = Image.fromarray(x_sample.astype(np.uint8))
                                results.append(image)
                                del image

    k_sigmas = None
    mask_fullres = None
    cfg_model.to('cpu')
    model_wrap.to('cpu')
    x_sample = None
    x_samples = None
    c = None
    uc = None
    model_wrap = None
    cfg_model.inner_model = None
    cfg_model = None
    del cfg_model
    del model_wrap
    del x_samples
    del c
    del uc
    del sampler

    torch_gc()
    return results
def generate_lowmem(args, root, frame = 0, return_latent=False, return_sample=False, return_c=False, step_callback=None, hires=None):
    results = generate_lm(
        prompt=args.prompt,
        ddim_steps=args.steps,
        n_iter=1,
        batch_size=1,
        Height=args.H,
        Width=args.W,
        scale=args.scale,
        ddim_eta=args.ddim_eta,
        unet_bs=1,
        device='cuda',
        seed=args.seed,
        outdir=args.outdir,
        img_format='PNG',
        turbo=False,
        full_precision=True,
        sampler='euler_a',
    )
    return results

    """
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)



    #HiRes_Fix
    goal_px = 512 * 512
    true_px = args.W * args.H
    scale = math.sqrt(goal_px / true_px)
    new_W = math.ceil(scale * args.W / 64) * 64
    new_H = math.ceil(scale * args.H / 64) * 64
    new_w_int = int(scale * args.W)
    new_h_int = int(scale * args.H)
    t_W = int(new_W - new_w_int) // args.f
    t_H = int(new_H - new_h_int) // args.f



    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            gs.models["modelFS"].to("cuda")
            init_latent = gs.models["modelFS"].get_first_stage_encoding(gs.models["modelFS"].encode_first_stage(args.init_sample))
            gs.models["modelFS"].to("cpu")
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to("cuda")
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            gs.models["modelFS"].to("cuda")
            init_latent = gs.models["modelFS"].get_first_stage_encoding(gs.models["modelFS"].encode_first_stage(init_image))  # move to latent space
            gs.models["modelFS"].to("cpu")
    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                            init_latent.shape,
                            args.mask_contrast_adjust,
                            args.mask_brightness_adjust,
                            args.invert_mask)

        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

        mask = mask.to("cuda")
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
        init_mse_image, mask_image = load_img(args.init_mse_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_mse_image = init_mse_image.to("cuda")
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( args.init_mse_scale != 0 and (args.init_mse_image is None or args.init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    t_enc = int((1.0-args.strength) * args.steps)
    # Noise schedule for the k-diffusion samplers (used for masking)
    #k_sigmas = model_wrap.get_sigmas(args.steps)
    #args.clamp_schedule = dict(zip(k_sigmas.tolist(), np.linspace(args.clamp_start,args.clamp_stop,args.steps+1)))
    #k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    #if args.sampler in ['plms','ddim']:
    #    sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, ddim_discretize='fill', verbose=False)

    #if args.colormatch_scale != 0:
    #    assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
    #    colormatch_image, _ = load_img(args.colormatch_image)
    #    colormatch_image = colormatch_image.to('cpu')
    #    del(_)
    #else:
    #    colormatch_image = None
    colormatch_image = None
    # Loss functions
    #if args.init_mse_scale != 0:
    #    if args.decode_method == "linear":
    #        mse_loss_fn = make_mse_loss(gs.models["sd"].linear_decode(gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(init_mse_image.to("cuda")))))
    #        mse_loss_fn = make_mse_loss(init_mse_image)
    #else:
    #    mse_loss_fn = None
    mse_loss_fn = None
    if args.colormatch_scale != 0:
        _,_ = get_color_palette(root, args.colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if args.decode_method == "linear":
            grad_img_shape = (int(args.W/args.f), int(args.H/args.f))
            colormatch_image = gs.models["sd"].linear_decode(gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(colormatch_image.to("cuda"))))
            colormatch_image = colormatch_image.to('cpu')
            print("Colormatch Sample Used")
        else:
            grad_img_shape = (args.W, args.H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image,
                                                  n_colors=args.colormatch_n_colors,
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=args.ignore_sat_weight)
    else:
        color_loss_fn = None

    if args.clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if args.aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if args.exposure_scale != 0:
        exposure_loss_fn = exposure_loss(args.exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              args.clip_scale],
        [blue_loss_fn,              args.blue_scale],
        [mean_loss_fn,              args.mean_scale],
        [exposure_loss_fn,          args.exposure_scale],
        [var_loss_fn,               args.var_scale],
        [mse_loss_fn,               args.init_mse_scale],
        [color_loss_fn,             args.colormatch_scale],
        [aesthetics_loss_fn,        args.aesthetics_scale]
    ]

    #callback = step_callback if step_callback is not None else None
    #callback = SamplerCallback(args=args,
    #                            root=root,
    #                            mask=mask,
    #                            init_latent=init_latent,
    #                            sigmas=k_sigmas,
    #                            sampler=sampler,
    #                            step_callback=step_callback,
    #                            verbose=True).callback

    #clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type, clamp_schedule=args.clamp_schedule)


    #grad_inject_timing_fn = make_inject_timing_fn(args.grad_inject_timing, model_wrap, args.steps)

    #cfg_model = CFGDenoiserWithGrad(model_wrap,
    #                                loss_fns_scales,
    #                                clamp_fn,
    #                                args.gradient_wrt,
    #                                args.gradient_add_to,
    #                                args.cond_uncond_sync,
    #                                decode_method=args.decode_method,
    #                                grad_inject_timing_fn=grad_inject_timing_fn, # option to use grad in only a few of the steps
    #                                grad_consolidate_fn=None, # function to add grad to image fn(img, grad, sigma)
    #                                verbose=False)
    gs.models["modelFS"].half()
    gs.models["modelCS"].half()
    gs.models["model"].half()
    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
                gs.models["modelCS"].to("cuda")
                for prompts in data:
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    if args.prompt_weighting:
                        uc, c = get_uc_and_c(prompts, gs.models["modelCS"], args, frame)
                    else:
                        if args.negative_prompts is not None:
                            print(f"using negative prompts: {args.negative_prompts}")
                            uc = gs.models["modelCS"].get_learned_conditioning(args.negative_prompts)
                        else:
                            #gs.models["modelCS"].cuda()
                            uc = gs.models["modelCS"].get_learned_conditioning(batch_size * [""])
                        c = gs.models["modelCS"].get_learned_conditioning(prompts)
                    gs.models["modelCS"].to("cpu")


                    if args.scale == 1.0:
                        uc = None
                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]:
                        x = t_enc
                        if hires:
                            args.oldH = args.H
                            args.oldW = args.W
                            args.H = new_H
                            args.W = new_W
                            x = create_random_tensors([4, new_H // 8, new_W // 8], seeds=(args.seed), seed_resize_from_h=0, seed_resize_from_w=0 )
                        #else:
                        #    args.H = args.oldH
                        #    args.W = args.oldW
                        torch_gc()
                        start_code = None
                        shape = [args.n_samples, args.C, args.H // args.f, args.W // args.f]

                        args.sampler = "euler_a"
                        samples = gs.models["model"].sample(
                            S=args.steps,
                            conditioning=c,
                            seed=args.seed,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=scale,
                            unconditional_conditioning=uc,
                            eta=args.ddim_eta,
                            x_T=start_code,
                            sampler=args.sampler,
                        )

                        if hires:
                            samples = samples[:, :, t_H//2:samples.shape[2]-t_H//2, t_W//2:samples.shape[3]-t_W//2]
                            print(samples.shape)
                            samples = resizeright.resize(samples, scale_factors=None, out_shape=[samples.shape[0], samples.shape[1], args.oldH // 8, args.oldW // 8],
                                                            interp_method=interp_methods.lanczos3, support_sz=None,
                                                            antialiasing=True, by_convs=True, scale_tolerance=None,
                                                            max_numerator=10, pad_mode='reflect')
                            #samples = torch.nn.functional.interpolate(samples, size=(args.oldH // 8, args.oldW // 8), mode="bilinear")
                            print(samples.shape)

                    else:
                        pass
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        #if init_latent is not None and args.strength > 0:
                        #    z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to("cuda"))
                        #else:
                        #    z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device="cuda")
                        #if args.sampler == 'ddim':
                        #    samples = sampler.decode(z_enc,
                        #                             c,
                        #                             t_enc,
                        #                             unconditional_guidance_scale=args.scale,
                        #                             unconditional_conditioning=uc,
                        #                             img_callback=callback)
                        #elif args.sampler == 'plms': # no "decode" function in plms, so use "sample"
                        #    shape = [args.C, args.H // args.f, args.W // args.f]
                        #   samples, _ = sampler.sample(S=args.steps,
                        #                                    conditioning=c,
                        #                                    batch_size=args.n_samples,
                        #                                    shape=shape,
                        #                                    verbose=False,
                        #                                    unconditional_guidance_scale=args.scale,
                        #                                    unconditional_conditioning=uc,
                        #                                    eta=args.ddim_eta,
                        #                                    x_T=z_enc,
                        #                                   img_callback=callback)
                        #else:
                        #    raise Exception(f"Sampler {args.sampler} not recognised.")


                    if return_latent:
                        results.append(samples.clone())
                    gs.models["modelFS"].to("cuda")
                    x_samples = gs.models["modelFS"].decode_first_stage(samples)
                    gs.models["modelFS"].to("cpu")
                    if args.use_mask and args.overlay_mask:
                        # Overlay the masked image after the image is generated
                        if args.init_sample is not None:
                            img_original = args.init_sample
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        mask_fullres = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                                    img_original.shape,
                                                    args.mask_contrast_adjust,
                                                    args.mask_brightness_adjust,
                                                    args.invert_mask)
                        mask_fullres = mask_fullres[:,:3,:,:]
                        mask_fullres = repeat(mask_fullres, '1 ... -> b ...', b=batch_size)

                        mask_fullres[mask_fullres < mask_fullres.max()] = 0
                        mask_fullres = gaussian_filter(mask_fullres, args.mask_overlay_blur)
                        mask_fullres = torch.Tensor(mask_fullres).to("cuda")

                        x_samples = img_original * mask_fullres + x_samples * ((mask_fullres * -1.0) + 1)


                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    #del args
    #del root

    del x_samples
    del c
    del uc
    #del sampler
    del image
    return results"""
def decode_first_stage(model, x):
    with autocast(choose_torch_device()):
        x = model.decode_first_stage(x)

    return x

def generate_lm(
        prompt,
        ddim_steps,
        n_iter,
        batch_size,
        Height,
        Width,
        scale,
        ddim_eta,
        unet_bs,
        device,
        seed,
        outdir,
        img_format,
        turbo,
        full_precision,
        sampler,
):
    C = 4
    f = 8
    start_code = None
    gs.models["model"].unet_bs = unet_bs
    gs.models["model"].turbo = turbo
    gs.models["model"].cdevice = device
    gs.models["modelCS"].cond_stage_model.device = device

    if seed == "":
        seed = randint(0, 1000000)
    seed = int(seed)
    seed_everything(seed)
    # Logging
    logger(locals(), "logs/txt2img_gradio_logs.csv")

    if device != "cpu" and full_precision == False:
        gs.models["model"].half()
        gs.models["modelFS"].half()
        gs.models["modelCS"].half()

    tic = time.time()
    os.makedirs(outdir, exist_ok=True)
    outpath = outdir
    sample_path = os.path.join(outpath, "_".join(re.split(":| ", prompt)))[:150]
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    # n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    assert prompt is not None
    data = [batch_size * [prompt]]

    if full_precision == False and device != "cpu":
        precision_scope = autocast
    else:
        precision_scope = nullcontext
    all_images = []
    all_samples = []
    seeds = ""
    with torch.no_grad():

        all_samples = list()
        for _ in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                with precision_scope("cuda"):
                    gs.models["modelCS"].to(device)
                    uc = None
                    if scale != 1.0:
                        uc = gs.models["modelCS"].get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)

                    subprompts, weights = split_weighted_subprompts(prompts[0])
                    if len(subprompts) > 1:
                        c = torch.zeros_like(uc)
                        totalWeight = sum(weights)
                        # normalize each "sub prompt" and add it
                        for i in range(len(subprompts)):
                            weight = weights[i]
                            # if not skip_normalize:
                            weight = weight / totalWeight
                            c = torch.add(c, gs.models["modelCS"].get_learned_conditioning(subprompts[i]), alpha=weight)
                    else:
                        c = gs.models["modelCS"].get_learned_conditioning(prompts)

                    shape = [batch_size, C, Height // f, Width // f]

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        gs.models["modelCS"].to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    samples_ddim = gs.models["model"].sample(
                        S=ddim_steps,
                        conditioning=c,
                        seed=seed,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc,
                        eta=ddim_eta,
                        x_T=start_code,
                        sampler=sampler,
                    )

                    gs.models["modelFS"].to(device)
                    print("saving images")

                    for i in range(batch_size):
                        x_samples_ddim = gs.models["modelFS"].decode_first_stage(samples_ddim[i].unsqueeze(0))
                        x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        all_samples.append(x_sample.to("cpu"))
                        x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        image.save(
                            os.path.join(sample_path, "seed_" + str(seed) + "_" + f"{base_count:05}.{img_format}")
                        )
                        all_images.append(image)
                        seeds += str(seed) + ","
                        seed += 1
                        base_count += 1

                    if device != "cpu":
                        mem = torch.cuda.memory_allocated() / 1e6
                        gs.models["modelFS"].to("cpu")
                        while torch.cuda.memory_allocated() / 1e6 >= mem:
                            time.sleep(1)

                    del samples_ddim
                    del x_sample
                    del x_samples_ddim
                    print("memory_final = ", torch.cuda.memory_allocated() / 1e6)

    toc = time.time()

    #time_taken = (toc - tic) / 60.0
    #grid = torch.cat(all_samples, 0)
    #grid = make_grid(grid, nrow=n_iter)
    #grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()

    #txt = (
    #        "Samples finished in "
    #        + str(round(time_taken, 3))
    #        + " minutes and exported to "
    #        + sample_path
    #        + "\nSeeds used = "
    #        + seeds[:-1]
    #)
    return all_images
