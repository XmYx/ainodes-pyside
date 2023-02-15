from typing import Any, Callable, Optional

import torch
from k_diffusion import sampling
from k_diffusion.external import CompVisVDenoiser
from torch import nn

from backend.singleton import singleton

gs = singleton

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale


def sampler_fn(
    c: torch.Tensor,
    uc: torch.Tensor,
    C,
    H,
    f,
    W,
    steps,
    n_samples,
    scale,
    sampler,
    model_wrap: CompVisVDenoiser,
    init_latent: Optional[torch.Tensor] = None,
    t_enc: Optional[torch.Tensor] = None,
    device=torch.device("cpu")
    if not torch.cuda.is_available()
    else torch.device("cuda"),
    cb: Callable[[Any], None] = None,
) -> torch.Tensor:
    shape = [C, H // f, W // f]
    sigmas: torch.Tensor = model_wrap.get_sigmas(steps)
    #print(f"sigmas: {sigmas}")
    sigmas = sigmas[len(sigmas) - t_enc - 1 :]
    #print(f"sigmas: {sigmas}")
    #resized = create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=H, seed_resize_from_w=W, p=None)
    if use_init:
        if len(sigmas) > 0:
            x = (
                init_latent
                + torch.randn([n_samples, *shape], device=device) * sigmas[0]
            )
        else:
            x = init_latent
    else:
        if len(sigmas) > 0:
            x = torch.randn([n_samples, *shape], device=device) * sigmas[0]
        else:
            x = torch.zeros([n_samples, *shape], device=device)
    sampler_args = {
        "model": CFGDenoiser(model_wrap),
        "x": x,
        "sigmas": sigmas,
        "extra_args": {"cond": c, "uncond": uc, "cond_scale": scale},
        "disable": False,
        "callback": cb,
    }
    sampler_map = {
        "klms": sampling.sample_lms,
        "dpm2": sampling.sample_dpm_2,
        "dpm2_ancestral": sampling.sample_dpm_2_ancestral,
        "heun": sampling.sample_heun,
        "euler": sampling.sample_euler,
        "euler_ancestral": sampling.sample_euler_ancestral,
    }

    samples = sampler_map[sampler](**sampler_args)
    return samples



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
