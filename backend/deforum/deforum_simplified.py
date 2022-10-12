#InvokeAI Generator import
import json

import cv2
import numpy as np
import pandas as pd
import torch, os
from PIL import Image
from einops import rearrange, repeat
from pytorch_lightning import seed_everything
from scipy.ndimage import gaussian_filter
from torch import autocast

from backend.deforum import DepthModel, sampler_fn

from backend.deforum.deforum_generator import prepare_mask, get_uc_and_c, DeformAnimKeys, sample_from_cv2, \
    sample_to_cv2, anim_frame_warp_2d, anim_frame_warp_3d, maintain_colors, add_noise, next_seed, SamplerCallback, \
    load_img
from ldm.generate import Generate
from backend import singleton
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from k_diffusion.external import CompVisDenoiser
from contextlib import contextmanager, nullcontext
import numexpr

gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

gs = singleton


gs.models = {}






class DeforumGenerator():
    prompts = [
        "a beautiful forest by Asher Brown Durand, trending on Artstation", # the first prompt I want
        "a beautiful portrait of a woman by Artgerm, trending on Artstation", # the second prompt I want
        #"this prompt I don't want it I commented it out",
        #"a nousr robot, trending on Artstation", # use "nousr robot" with the robot diffusion model (see model_checkpoint setting)
        #"touhou 1girl komeiji_koishi portrait, green hair", # waifu diffusion prompts can use danbooru tag groups (see model_checkpoint)
        #"this prompt has weights if prompt weighting enabled:2 can also do negative:-2", # (see prompt_weighting)
    ]
    animation_prompts = {
        0: "a beautiful apple, trending on Artstation",
        20: "a beautiful banana, trending on Artstation",
        30: "a beautiful coconut, trending on Artstation",
        40: "a beautiful durian, trending on Artstation",
    }
    W = 512 #@param
    H = 512 #@param
    W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
    seed = -1 #@param
    sampler = 'klms' #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = 50 #@param
    scale = 7 #@param
    ddim_eta = 0.0 #@param
    dynamic_threshold = None
    static_threshold = None
    #@markdown **Save & Display Settings**
    save_samples = True #@param {type:"boolean"}
    save_settings = True #@param {type:"boolean"}
    display_samples = True #@param {type:"boolean"}
    save_sample_per_step = False #@param {type:"boolean"}
    show_sample_per_step = False #@param {type:"boolean"}
    prompt_weighting = False #@param {type:"boolean"}
    normalize_prompt_weights = True #@param {type:"boolean"}
    log_weighted_subprompts = False #@param {type:"boolean"}

    n_batch = 1 #@param
    batch_name = "StableFun" #@param {type:"string"}
    filename_format = "{timestring}_{index}_{prompt}.png" #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = "iter" #@param ["iter","fixed","random"]
    make_grid = False #@param {type:"boolean"}
    grid_rows = 2 #@param
    outdir = "output"
    use_init = False #@param {type:"boolean"}
    strength = 0.0 #@param {type:"number"}
    strength_0_no_init = True # Set the strength to 0 automatically when no init image is used
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg" #@param {type:"string"}
    # Whiter areas of the mask are areas that change more
    use_mask = False #@param {type:"boolean"}
    use_alpha_as_mask = False # use the alpha channel of the init image as the mask
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" #@param {type:"string"}
    invert_mask = False #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = 1.0  #@param {type:"number"}
    mask_contrast_adjust = 1.0  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = True  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = 5 # {type:"number"}

    n_samples = 1 # doesnt do anything
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    #Anim Args

    animation_mode = 'None' #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = 1000 #@param {type:"number"}
    border = 'replicate' #@param ['wrap', 'replicate'] {type:'string'}
    #@markdown ####**Motion Parameters:**
    angle = "0:(0)"#@param {type:"string"}
    zoom = "0:(1.04)"#@param {type:"string"}
    translation_x = "0:(10*sin(2*3.14*t/10))"#@param {type:"string"}
    translation_y = "0:(0)"#@param {type:"string"}
    translation_z = "0:(10)"#@param {type:"string"}
    rotation_3d_x = "0:(0)"#@param {type:"string"}
    rotation_3d_y = "0:(0)"#@param {type:"string"}
    rotation_3d_z = "0:(0)"#@param {type:"string"}
    flip_2d_perspective = False #@param {type:"boolean"}
    perspective_flip_theta = "0:(0)"#@param {type:"string"}
    perspective_flip_phi = "0:(t%15)"#@param {type:"string"}
    perspective_flip_gamma = "0:(0)"#@param {type:"string"}
    perspective_flip_fv = "0:(53)"#@param {type:"string"}
    noise_schedule = "0: (0.02)"#@param {type:"string"}
    strength_schedule = "0: (0.65)"#@param {type:"string"}
    contrast_schedule = "0: (1.0)"#@param {type:"string"}
    #@markdown ####**Coherence:**
    color_coherence = 'Match Frame 0 LAB' #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = '1' #@param ['1','2','3','4','5','6','7','8'] {type:'string'}
    #@markdown ####**3D Depth Warping:**
    use_depth_warping = True #@param {type:"boolean"}
    midas_weight = 0.3#@param {type:"number"}
    near_plane = 200
    far_plane = 10000
    fov = 40#@param {type:"number"}
    padding_mode = 'border'#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = 'bicubic'#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = False #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path ='/content/video_in.mp4'#@param {type:"string"}
    extract_nth_frame = 1#@param {type:"number"}
    overwrite_extracted_frames = True #@param {type:"boolean"}
    use_mask_video = False #@param {type:"boolean"}
    video_mask_path ='/content/video_in.mp4'#@param {type:"string"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = False #@param {type:"boolean"}
    interpolate_x_frames = 4 #@param {type:"number"}

    #@markdown ####**Resume Animation:**
    resume_from_timestring = False #@param {type:"boolean"}
    resume_timestring = "20220829210106" #@param {type:"string"}
    def __init__(self, *args, **kwargs):
        self.render_animation()
        print(locals())


    def render_animation(self):
        if "sd" not in gs.models:
            gs.models["sd"] = gr.load_model()

        # animations use key framed prompts
        self.prompts = self.animation_prompts

        # expand key frame strings to values
        keys = DeformAnimKeys(self)

        # resume animation
        start_frame = 0
        if self.resume_from_timestring:
            for tmp in os.listdir(self.outdir):
                if tmp.split("_")[0] == self.resume_timestring:
                    start_frame += 1
            start_frame = start_frame - 1

        # create output folder for the batch
        os.makedirs(self.outdir, exist_ok=True)
        print(f"Saving animation frames to {self.outdir}")

        # save settings for the batch
        settings_filename = os.path.join(self.outdir, f"{self.timestring}_settings.txt")
        with open(settings_filename, "w+", encoding="utf-8") as f:
            s = {**dict(self.__dict__), **dict(self.__dict__)}
            json.dump(s, f, ensure_ascii=False, indent=4)

        # resume from timestring
        if self.resume_from_timestring:
            self.timestring = self.resume_timestring

        # expand prompts out to per-frame
        prompt_series = pd.Series([np.nan for a in range(self.max_frames)])
        for i, prompt in self.animation_prompts.items():
            prompt_series[i] = prompt
        prompt_series = prompt_series.ffill().bfill()

        # check for video inits
        using_vid_init = self.animation_mode == 'Video Input'

        # load depth model for 3D
        predict_depths = (self.animation_mode == '3D' and self.use_depth_warping) or self.save_depth_maps
        if predict_depths:
            depth_model = DepthModel('cuda')
            depth_model.load_midas('models/')
            if self.midas_weight < 1.0:
                depth_model.load_adabins()
        else:
            depth_model = None
            self.save_depth_maps = False

        # state for interpolating between diffusion steps
        turbo_steps = 1 if using_vid_init else int(self.diffusion_cadence)
        turbo_prev_image, turbo_prev_frame_idx = None, 0
        turbo_next_image, turbo_next_frame_idx = None, 0

        # resume animation
        prev_sample = None
        color_match_sample = None
        if self.resume_from_timestring:
            last_frame = start_frame-1
            if turbo_steps > 1:
                last_frame -= last_frame%turbo_steps
            path = os.path.join(self.outdir,f"{self.timestring}_{last_frame:05}.png")
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_sample = sample_from_cv2(img)
            if self.color_coherence != 'None':
                color_match_sample = img
            if turbo_steps > 1:
                turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                start_frame = last_frame+turbo_steps

        self.n_samples = 1
        frame_idx = start_frame
        while frame_idx < self.max_frames:
            print(f"Rendering animation frame {frame_idx} of {self.max_frames}")
            noise = keys.noise_schedule_series[frame_idx]
            strength = keys.strength_schedule_series[frame_idx]
            contrast = keys.contrast_schedule_series[frame_idx]
            depth = None

            # emit in-between frames
            if turbo_steps > 1:
                tween_frame_start_idx = max(0, frame_idx-turbo_steps)
                for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                    tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(frame_idx - tween_frame_start_idx)
                    print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                    advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                    advance_next = tween_frame_idx > turbo_next_frame_idx

                    if depth_model is not None:
                        assert(turbo_next_image is not None)
                        depth = depth_model.predict(turbo_next_image, self)

                    if self.animation_mode == '2D':
                        if advance_prev:
                            turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, self, self, keys, tween_frame_idx)
                        if advance_next:
                            turbo_next_image = anim_frame_warp_2d(turbo_next_image, self, self, keys, tween_frame_idx)
                    else: # '3D'
                        if advance_prev:
                            turbo_prev_image = anim_frame_warp_3d(turbo_prev_image, depth, self, keys, tween_frame_idx)
                        if advance_next:
                            turbo_next_image = anim_frame_warp_3d(turbo_next_image, depth, self, keys, tween_frame_idx)
                    turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                    if turbo_prev_image is not None and tween < 1.0:
                        img = turbo_prev_image*(1.0-tween) + turbo_next_image*tween
                    else:
                        img = turbo_next_image

                    filename = f"{self.timestring}_{tween_frame_idx:05}.png"
                    cv2.imwrite(os.path.join(self.outdir, filename), cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if self.save_depth_maps:
                        depth_model.save(os.path.join(self.outdir, f"{self.timestring}_depth_{tween_frame_idx:05}.png"), depth)
                if turbo_next_image is not None:
                    prev_sample = sample_from_cv2(turbo_next_image)

            # apply transforms to previous frame
            if prev_sample is not None:
                if self.animation_mode == '2D':
                    prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), self, self, keys, frame_idx)
                else: # '3D'
                    prev_img_cv2 = sample_to_cv2(prev_sample)
                    depth = depth_model.predict(prev_img_cv2, self) if depth_model else None
                    prev_img = anim_frame_warp_3d(prev_img_cv2, depth, self, keys, frame_idx)

                # apply color matching
                if self.color_coherence != 'None':
                    if color_match_sample is None:
                        color_match_sample = prev_img.copy()
                    else:
                        prev_img = maintain_colors(prev_img, color_match_sample, self.color_coherence)

                # apply scaling
                contrast_sample = prev_img * contrast
                # apply frame noising
                noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

                # use transformed previous frame as init for current
                self.use_init = True
                half_precision=True
                if half_precision:
                    self.init_sample = noised_sample.half().to('cuda')
                else:
                    self.init_sample = noised_sample.to('cuda')
                self.strength = max(0.0, min(1.0, strength))

            # grab prompt for current frame
            self.prompt = prompt_series[frame_idx]
            print(f"{self.prompt} {self.seed}")
            if not using_vid_init:
                print(f"Angle: {keys.angle_series[frame_idx]} Zoom: {keys.zoom_series[frame_idx]}")
                print(f"Tx: {keys.translation_x_series[frame_idx]} Ty: {keys.translation_y_series[frame_idx]} Tz: {keys.translation_z_series[frame_idx]}")
                print(f"Rx: {keys.rotation_3d_x_series[frame_idx]} Ry: {keys.rotation_3d_y_series[frame_idx]} Rz: {keys.rotation_3d_z_series[frame_idx]}")

            # grab init image for current frame
            if using_vid_init:
                init_frame = os.path.join(self.outdir, 'inputframes', f"{frame_idx+1:05}.jpg")
                print(f"Using video init frame {init_frame}")
                self.init_image = init_frame
                if self.use_mask_video:
                    mask_frame = os.path.join(self.outdir, 'maskframes', f"{frame_idx+1:05}.jpg")
                    self.mask_file = mask_frame

            # sample the diffusion model
            sample, image = self.generate(frame_idx, return_latent=False, return_sample=True)
            if not using_vid_init:
                prev_sample = sample

            if turbo_steps > 1:
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
                frame_idx += turbo_steps
            else:
                filename = f"{self.timestring}_{frame_idx:05}.png"
                image.save(os.path.join(self.outdir, filename))
                if self.save_depth_maps:
                    if depth is None:
                        depth = depth_model.predict(sample_to_cv2(sample), self)
                    depth_model.save(os.path.join(self.outdir, f"{self.timestring}_depth_{frame_idx:05}.png"), depth)
                frame_idx += 1

            #display.clear_output(wait=True)
            #display.display(image)

            self.seed = next_seed(self)



    def generate(self, frame = 0, return_latent=False, return_sample=False, return_c=False):
        seed_everything(self.seed)
        os.makedirs(self.outdir, exist_ok=True)

        sampler = PLMSSampler(gs.models["sd"]) if self.sampler == 'plms' else DDIMSampler(gs.models["sd"])
        model_wrap = CompVisDenoiser(gs.models["sd"])
        batch_size = self.n_samples
        prompt = self.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        precision_scope = autocast if self.precision == "autocast" else nullcontext

        init_latent = None
        mask_image = None
        init_image = None
        if self.init_latent is not None:
            init_latent = self.init_latent
        elif self.init_sample is not None:
            with precision_scope("cuda"):
                init_latent = gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(self.init_sample))
        elif self.use_init and self.init_image != None and self.init_image != '':
            init_image, mask_image = load_img(self.init_image,
                                              shape=(self.W, self.H),
                                              use_alpha_as_mask=self.use_alpha_as_mask)
            init_image = init_image.to('cuda')
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            with precision_scope("cuda"):
                init_latent = gs.models["sd"].get_first_stage_encoding(gs.models["sd"].encode_first_stage(init_image))  # move to latent space

        if not self.use_init and self.strength > 0 and self.strength_0_no_init:
            print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
            print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
            self.strength = 0

        # Mask functions
        if self.use_mask:
            assert self.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
            assert self.use_init, "use_mask==True: use_init is required for a mask"
            assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


            mask = prepare_mask(self.mask_file if mask_image is None else mask_image,
                                init_latent.shape,
                                self.mask_contrast_adjust,
                                self.mask_brightness_adjust)

            if (torch.all(mask == 0) or torch.all(mask == 1)) and self.use_alpha_as_mask:
                raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

            mask = mask.to('cuda')
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)
        else:
            mask = None

        assert not ( (self.use_mask and self.overlay_mask) and (self.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        t_enc = int((1.0-self.strength) * self.steps)

        # Noise schedule for the k-diffusion samplers (used for masking)
        k_sigmas = model_wrap.get_sigmas(self.steps)
        k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

        if self.sampler in ['plms','ddim']:
            sampler.make_schedule(ddim_num_steps=self.steps, ddim_eta=self.ddim_eta, ddim_discretize='fill', verbose=False)

        """callback = SamplerCallback(self,
                                   mask=mask,
                                   init_latent=init_latent,
                                   sigmas=k_sigmas,
                                   sampler=sampler,
                                   verbose=False).callback"""
        callback = None

        results = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with gs.models["sd"].ema_scope():
                    for prompts in data:
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        if self.prompt_weighting:
                            uc, c = get_uc_and_c(prompts, gs.models["sd"], self, frame)
                        else:
                            uc = gs.models["sd"].get_learned_conditioning(batch_size * [""])
                            c = gs.models["sd"].get_learned_conditioning(prompts)


                        if self.scale == 1.0:
                            uc = None
                        if self.init_c != None:
                            c = self.init_c

                        if self.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                            samples = sampler_fn(
                                self,

                                device='cuda',
                                cb=callback)
                        else:
                            # self.sampler == 'plms' or self.sampler == 'ddim':
                            if init_latent is not None and self.strength > 0:
                                z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to('cuda'))
                            else:
                                z_enc = torch.randn([self.n_samples, self.C, self.H // self.f, self.W // self.f], device='cuda')
                            if self.sampler == 'ddim':
                                samples = sampler.decode(z_enc,
                                                         c,
                                                         t_enc,
                                                         unconditional_guidance_scale=self.scale,
                                                         unconditional_conditioning=uc,
                                                         img_callback=callback)
                            elif self.sampler == 'plms': # no "decode" function in plms, so use "sample"
                                shape = [self.C, self.H // self.f, self.W // self.f]
                                samples, _ = sampler.sample(S=self.steps,
                                                            conditioning=c,
                                                            batch_size=self.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=self.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=self.ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                            else:
                                raise Exception(f"Sampler {self.sampler} not recognised.")


                        if return_latent:
                            results.append(samples.clone())

                        x_samples = gs.models["sd"].decode_first_stage(samples)

                        if self.use_mask and self.overlay_mask:
                            # Overlay the masked image after the image is generated
                            if self.init_sample is not None:
                                img_original = self.init_sample
                            elif init_image is not None:
                                img_original = init_image
                            else:
                                raise Exception("Cannot overlay the masked image without an init image to overlay")

                            mask_fullres = prepare_mask(self.mask_file if mask_image is None else mask_image,
                                                        img_original.shape,
                                                        self.mask_contrast_adjust,
                                                        self.mask_brightness_adjust)
                            mask_fullres = mask_fullres[:,:3,:,:]
                            mask_fullres = repeat(mask_fullres, '1 ... -> b ...', b=batch_size)

                            mask_fullres[mask_fullres < mask_fullres.max()] = 0
                            mask_fullres = gaussian_filter(mask_fullres, self.mask_overlay_blur)
                            mask_fullres = torch.Tensor(mask_fullres).to('cuda')

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
        return results
