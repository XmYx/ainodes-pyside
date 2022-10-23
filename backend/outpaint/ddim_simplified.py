"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, \
    extract_into_tensor


class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps,
        ddim_discretize="uniform",
        ddim_eta=0.,
        verbose=True
        ):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps, verbose=verbose)
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(
            self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(
            np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(
            np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta, verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def ddim_sampling(
            self,
            cond,
            shape,
            mask=None,
            mask_encoded_strength=True,
            x0=None,
            t_start=None,
            unconditional_guidance_scale=1.,
            unconditional_conditioning=None):

        device = self.model.betas.device
        batch_size = shape[0]

        timesteps = self.ddim_timesteps

        wanted_steps = timesteps.shape[0]

        stop_mask_at_step = 0

        if mask_encoded_strength and mask is not None:
            t_start = round((1. - float(mask.min().cpu())) * wanted_steps)
            stop_mask_at_step = round((1. - float(mask.max().cpu())) * 1000)

        latent_img = None
        if t_start is not None and t_start < wanted_steps:
            latent_img = self.stochastic_encode(
                x0, torch.tensor([t_start]*batch_size).to(device))
            timesteps = timesteps[:t_start]

        if latent_img is None:
            latent_img = torch.randn(shape, device=device)

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1

            ts = torch.full((batch_size,), step, device=device,
                            dtype=torch.long)  # unique int, as a tensor
            
            # equivalent to ts = tensor([step]*batch_size).to(device)

            # print(f"step {step}")

            if mask is not None and step > stop_mask_at_step:
                assert x0 is not None
                # TODO: deterministic forward pass?

                img_orig_with_noise = self.model.q_sample(
                    x0, ts)  # [1, 4, 64, 64] => [1, 4, 64, 64]
                # equivalent to
                # img_orig_with_noise = self.stochastic_encode(x0, tensor([index]*batch_size).to(device))

                if mask_encoded_strength:
                    tmp_mask = (mask > (1 - (step / 1000))) * 1
                    latent_img = img_orig_with_noise * \
                        tmp_mask + (1. - tmp_mask) * latent_img
                else:
                    latent_img = img_orig_with_noise * \
                        mask + (1. - mask) * latent_img

            latent_img, _ = self.p_sample_ddim(
                latent_img,
                cond,
                ts,
                index=index,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=unconditional_conditioning)
        return latent_img

    @torch.no_grad()
    def p_sample_ddim(
            self,
            x,
            c,
            t,
            index,
            repeat_noise=False,
            unconditional_guidance_scale=1.,
            unconditional_conditioning=None):

        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            e_t = self.model.apply_model(x, t, c)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([t] * 2)
            c_in = torch.cat([unconditional_conditioning, c])
            e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
            e_t = e_t_uncond + unconditional_guidance_scale * \
                (e_t - e_t_uncond)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t     = torch.full((b, 1, 1, 1), alphas[index],      device=device)
        a_prev  = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index],      device=device)
        sqrt_one_minus_at = torch.full(
            (b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0

    @torch.no_grad()
    def stochastic_encode(self, x0, t, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
        sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
