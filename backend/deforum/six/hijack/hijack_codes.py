import torch
import numpy as np
from tqdm import tqdm



def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
           use_original_steps=False, img_callback=None):

    timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
    timesteps = timesteps[:t_start]

    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    print(f"Running DDIM Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
    x_dec = x_latent
    for i, step in enumerate(iterator):
        index = total_steps - i - 1

        ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
        x_dec, pred_x0 = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=unconditional_conditioning)

        if img_callback: img_callback(x_dec, pred_x0, i)

    return x_dec

def plms_sampling(self, cond, shape,
                  x_T=None, ddim_use_original_steps=False,
                  callback=None, timesteps=None, quantize_denoised=False,
                  mask=None, x0=None, img_callback=None, log_every_t=100,
                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None,):
    device = self.model.betas.device
    b = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=device)
    else:
        img = x_T

    if timesteps is None:
        timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
    elif timesteps is not None and not ddim_use_original_steps:
        subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
        timesteps = self.ddim_timesteps[:subset_end]

    intermediates = {'x_inter': [img], 'pred_x0': [img]}
    time_range = list(reversed(range(0,timesteps))) if ddim_use_original_steps else np.flip(timesteps)
    total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
    print(f"Running PLMS Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
    old_eps = []

    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((b,), step, device=device, dtype=torch.long)
        ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

        if mask is not None:
            assert x0 is not None
            img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
            img = img_orig * mask + (1. - mask) * img

        outs = self.p_sample_plms(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                  quantize_denoised=quantize_denoised, temperature=temperature,
                                  noise_dropout=noise_dropout, score_corrector=score_corrector,
                                  corrector_kwargs=corrector_kwargs,
                                  unconditional_guidance_scale=unconditional_guidance_scale,
                                  unconditional_conditioning=unconditional_conditioning,
                                  old_eps=old_eps, t_next=ts_next)
        img, pred_x0, e_t = outs
        old_eps.append(e_t)
        if len(old_eps) >= 4:
            old_eps.pop(0)
        if callback: callback(i)
        if img_callback: img_callback(img, pred_x0, i)

        if index % log_every_t == 0 or index == total_steps - 1:
            intermediates['x_inter'].append(img)
            intermediates['pred_x0'].append(pred_x0)

    return img, intermediates
