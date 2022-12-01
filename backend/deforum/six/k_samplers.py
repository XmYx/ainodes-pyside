from typing import Any, Callable, Optional
from k_diffusion.external import CompVisVDenoiser, CompVisDenoiser
from k_diffusion import sampling
import torch
from backend.singleton import singleton
gs = singleton

def sampler_fn(
    c: torch.Tensor,
    uc: torch.Tensor,
    args,
    model_wrap: CompVisVDenoiser,
    init_latent: Optional[torch.Tensor] = None,
    t_enc: Optional[torch.Tensor] = None,
    device=torch.device("cpu")
    if not torch.cuda.is_available()
    else torch.device("cuda"),
    cb: Callable[[Any], None] = None,
    verbose: Optional[bool] = False,
) -> torch.Tensor:
    shape = [args.C, args.H // args.f, args.W // args.f]
    #gs.karras = True
    if gs.karras == True:
        print("Using Karras Scheduler")
        sigmas = sampling.get_sigmas_karras(n=args.steps, sigma_min=0.1, sigma_max=10, device="cuda")
    else:
        sigmas: torch.Tensor = model_wrap.get_sigmas(args.steps)
    #print(f"sigmas: {sigmas}")
    sigmas = sigmas[len(sigmas) - t_enc - 1 :]
    #print(f"sigmas: {sigmas}")
    if args.use_init:
        if len(sigmas) > 0:
            x = (
                init_latent
                + torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
            )
            #x = (
            #    init_latent
            #    + gs.x * sigmas[0]
            #)
        else:
            x = init_latent
    else:
        if len(sigmas) > 0:
            x = torch.randn([args.n_samples, *shape], device=device) * sigmas[0]
            #x = gs.x * sigmas[0]
        else:
            x = torch.zeros([args.n_samples, *shape], device=device)
    sampler_args = {
        "model": model_wrap,
        "x": x,
        "sigmas": sigmas,
        "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
        "disable": False,
        "callback": cb,
    }
    min = sigmas[0].item()
    max = min
    for i in sigmas:
        if i.item() < min and i.item() != 0.0:
            min = i.item()
    if args.sampler in ["dpm_fast"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
            "n":args.steps,
            "eta": 0.0,
            "s_noise": 1.0,
        }
    elif args.sampler in ["dpm_adaptive"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigma_min": min,
            "sigma_max": max,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
            "order": 3,
            "rtol": 0.05,
            "atol": 0.0078,
            "h_init": 0.05,
            "pcoeff": 0.0,
            "icoeff": 1.0,
            "dcoeff": 0.0,
            "eta": 0.0,
            "s_noise": 1.0,
        }

    elif args.sampler in ["dpmpp_sde", "dpmpp_2s_a"]:
        sampler_args = {
            "model": model_wrap,
            "x": x,
            "sigmas": sigmas,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": args.scale},
            "disable": False,
            "callback": cb,
            "eta": 1.0,
            "s_noise": 1.0,
        }

    sampler_map = {
        "klms": sampling.sample_lms,
        "dpm2": sampling.sample_dpm_2,
        "dpm2_ancestral": sampling.sample_dpm_2_ancestral,
        "heun": sampling.sample_heun,
        "euler": sampling.sample_euler,
        "euler_ancestral": sampling.sample_euler_ancestral,
        "dpm_fast": sampling.sample_dpm_fast,
        "dpm_adaptive": sampling.sample_dpm_adaptive,
        "dpmpp_2s_a": sampling.sample_dpmpp_2s_ancestral,
        "dpmpp_2m": sampling.sample_dpmpp_2m,
        "dpmpp_sde": sampling.sample_dpmpp_sde,
    }

    samples = sampler_map[args.sampler](**sampler_args)
    return samples


def make_inject_timing_fn(inject_timing, model, steps):
    """
    inject_timing (int or list of ints or list of floats between 0.0 and 1.0): 
        int: compute every inject_timing steps
        list of floats: compute on these decimal fraction steps (eg, [0.5, 1.0] for 50 steps would be at steps 25 and 50)
        list of ints: compute on these steps
    model (CompVisDenoiser)
    steps (int): number of steps
    """
    all_sigmas = model.get_sigmas(steps)
    target_sigmas = torch.empty([0], device=all_sigmas.device)

    def timing_fn(sigma):
        is_conditioning_step = False
        if sigma in target_sigmas:
            is_conditioning_step = True
        return is_conditioning_step

    if inject_timing is None:
        timing_fn = lambda sigma: True
    elif isinstance(inject_timing,int) and inject_timing <= steps and inject_timing > 0:
        # Compute every nth step
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if (i+1) % inject_timing == 0]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,float) for t in inject_timing) and all(t>=0.0 and t<=1.0 for t in inject_timing):
        # Compute on these steps (expressed as a decimal fraction between 0.0 and 1.0)
        target_indices = [int(frac_step*steps) if frac_step < 1.0 else steps-1 for frac_step in inject_timing]
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i in target_indices]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)
    elif all(isinstance(t,int) for t in inject_timing) and all(t>0 and t<=steps for t in inject_timing):
        # Compute on these steps
        target_sigma_list = [sigma for i,sigma in enumerate(all_sigmas) if i+1 in inject_timing]
        target_sigmas = torch.Tensor(target_sigma_list).to(all_sigmas.device)

    else:
        raise Exception(f"Not a valid input: inject_timing={inject_timing}\n" +
                        f"Must be an int, list of all ints (between step 1 and {steps}), or list of all floats between 0.0 and 1.0")
    return timing_fn
def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and (len(seeds) > 1 and opts.enable_batch_seeds or opts.eta_noise_seed_delta > 0):
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

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

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            if opts.eta_noise_seed_delta > 0:
                torch.manual_seed(seed + opts.eta_noise_seed_delta)

            for j in range(cnt):
                sampler_noises[j].append(randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(gs.system.device) for n in sampler_noises]

    x = torch.stack(xs).to(gs.system.device)
    return x


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

def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if gs.system.device == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(device)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if gs.system.device == 'mps':
        generator = torch.Generator(device="cpu")
        noise = torch.randn(shape, generator=generator, device="cpu").to(gs.system.device)
        return noise

    return torch.randn(shape, device=gs.system.device)
