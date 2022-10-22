# InvokeAI Generator import
import json
import random
import re
import subprocess
import sys
import time
from time import gmtime, strftime
import traceback
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch, os
from PIL import Image
from PySide6.QtCore import Slot, Signal
from einops import rearrange, repeat
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from scipy.ndimage import gaussian_filter
from torch import autocast

from backend.deforum import DepthModel, sampler_fn

from backend.deforum.deforum_generator import prepare_mask, get_uc_and_c, DeformAnimKeys, sample_from_cv2, \
    sample_to_cv2, anim_frame_warp_2d, anim_frame_warp_3d, maintain_colors, add_noise, next_seed, \
    load_img, get_inbetweens, parse_key_frames, check_is_number
from ldm.dream.devices import choose_torch_device

from ldm.generate import Generate
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from contextlib import contextmanager, nullcontext
import numexpr

from typing import Any, Callable, Optional
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

import gc
from k_diffusion.external import CompVisDenoiser
from k_diffusion import sampling
from torch import nn
from backend.singleton import singleton

gs = singleton
from ldm.util import instantiate_from_config


def get_uc_and_c(prompts, model, n_samples, log_weighted_subprompts, normalize_prompt_weights, frame=0):
    prompt = prompts[0]  # they are the same in a batch anyway

    # get weighted sub-prompts
    negative_subprompts, positive_subprompts = split_weighted_subprompts(
        prompt, frame, not normalize_prompt_weights
    )

    uc = get_learned_conditioning(model, negative_subprompts, "", n_samples, log_weighted_subprompts, -1)
    c = get_learned_conditioning(model, positive_subprompts, prompt, n_samples, log_weighted_subprompts, 1)

    return (uc, c)


def get_learned_conditioning(model, weighted_subprompts, text, n_samples, log_weighted_subprompts, sign=1):
    if len(weighted_subprompts) < 1:
        log_tokenization(text, model, log_weighted_subprompts, sign)
        c = model.get_learned_conditioning(n_samples * [text])
    else:
        c = None
        for subtext, subweight in weighted_subprompts:
            log_tokenization(subtext, model, log_weighted_subprompts, sign * subweight)
            if c is None:
                c = model.get_learned_conditioning(n_samples * [subtext])
                c *= subweight
            else:
                c.add_(model.get_learned_conditioning(n_samples * [subtext]), alpha=subweight)

    return c


def parse_weight(match, frame=0) -> float:
    import numexpr
    w_raw = match.group("weight")
    if w_raw == None:
        return 1
    if check_is_number(w_raw):
        return float(w_raw)
    else:
        t = frame
        if len(w_raw) < 3:
            print('the value inside `-characters cannot represent a math function')
            return 1
        return float(numexpr.evaluate(w_raw[1:-1]))


def normalize_prompt_weights(parsed_prompts):
    if len(parsed_prompts) == 0:
        return parsed_prompts
    weight_sum = sum(map(lambda x: x[1], parsed_prompts))
    if weight_sum == 0:
        print(
            "Warning: Subprompt weights add up to zero. Discarding and using even weights instead.")
        equal_weight = 1 / max(len(parsed_prompts), 1)
        return [(x[0], equal_weight) for x in parsed_prompts]
    return [(x[0], x[1] / weight_sum) for x in parsed_prompts]


def split_weighted_subprompts(text, frame=0, skip_normalize=False):
    """
    grabs all text up to the first occurrence of ':'
    uses the grabbed text as a sub-prompt, and takes the value following ':' as weight
    if ':' has no value defined, defaults to 1.0
    repeats until no text remaining
    """
    prompt_parser = re.compile("""
            (?P<prompt>         # capture group for 'prompt'
            (?:\\\:|[^:])+      # match one or more non ':' characters or escaped colons '\:'
            )                   # end 'prompt'
            (?:                 # non-capture group
            :+                  # match one or more ':' characters
            (?P<weight>((        # capture group for 'weight'
            -?\d+(?:\.\d+)?     # match positive or negative integer or decimal number
            )|(                 # or
            `[\S\s]*?`# a math function
            )))?                 # end weight capture group, make optional
            \s*                 # strip spaces after weight
            |                   # OR
            $                   # else, if no ':' then match end of line
            )                   # end non-capture group
            """, re.VERBOSE)
    negative_prompts = []
    positive_prompts = []
    for match in re.finditer(prompt_parser, text):
        w = parse_weight(match, frame)
        if w < 0:
            # negating the sign as we'll feed this to uc
            negative_prompts.append((match.group("prompt").replace("\\:", ":"), -w))
        elif w > 0:
            positive_prompts.append((match.group("prompt").replace("\\:", ":"), w))

    if skip_normalize:
        return (negative_prompts, positive_prompts)
    return (normalize_prompt_weights(negative_prompts), normalize_prompt_weights(positive_prompts))


# shows how the prompt is tokenized
# usually tokens have '</w>' to indicate end-of-word,
# but for readability it has been replaced with ' '
def log_tokenization(text, model, log=False, weight=1):
    if not log:
        return
    tokens = model.cond_stage_model.tokenizer._tokenize(text)
    tokenized = ""
    discarded = ""
    usedTokens = 0
    totalTokens = len(tokens)
    for i in range(0, totalTokens):
        token = tokens[i].replace('</w>', ' ')
        # alternate color
        s = (usedTokens % 6) + 1
        if i < model.cond_stage_model.max_length:
            tokenized = tokenized + f"\x1b[0;3{s};40m{token}"
            usedTokens += 1
        else:  # over max token length
            discarded = discarded + f"\x1b[0;3{s};40m{token}"
    print(f"\n>> Tokens ({usedTokens}), Weight ({weight:.2f}):\n{tokenized}\x1b[0m")
    if discarded != "":
        print(
            f">> Tokens Discarded ({totalTokens - usedTokens}):\n{discarded}\x1b[0m"
        )


class DeforumGenerator():

    def __init__(self):
        self.device = 'cuda'
        self.outdir = gs.system.txt2vidSingleFrame
        self.shouldStop = False


    @Slot()
    def setStop(self):
        self.shouldStop = True

    def torch_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def load_model(self):
        """Load and initialize the model from configuration variables passed at object creation time"""

        weights = 'models/sd-v1-4.ckpt'
        config = 'configs/stable-diffusion/v1-inference.yaml'
        embedding_path = None

        """Load and initialize the model from configuration variables passed at object creation time"""
        seed_everything(random.randrange(0, np.iinfo(np.uint32).max))
        try:
            config = OmegaConf.load(config)
            model = self._load_model_from_config(config, weights)
            if embedding_path is not None:
                gs.models["sd"].embedding_manager.load(
                    embedding_path, self.full_precision
                )
            # model = model.half().to(self.device)
            # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
            model.cond_stage_model.device = self.device
        except AttributeError as e:
            print(f'>> Error loading model. {str(e)}', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            raise SystemExit from e

        # self._set_sampler()

        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode

        return model

    def animkeys(self):
        self.angle_series = get_inbetweens(parse_key_frames(self.angle), self.max_frames)
        self.zoom_series = get_inbetweens(parse_key_frames(self.zoom), self.max_frames)
        self.translation_x_series = get_inbetweens(parse_key_frames(self.translation_x), self.max_frames)
        self.translation_y_series = get_inbetweens(parse_key_frames(self.translation_y), self.max_frames)
        self.translation_z_series = get_inbetweens(parse_key_frames(self.translation_z), self.max_frames)
        self.rotation_3d_x_series = get_inbetweens(parse_key_frames(self.rotation_3d_x), self.max_frames)
        self.rotation_3d_y_series = get_inbetweens(parse_key_frames(self.rotation_3d_y), self.max_frames)
        self.rotation_3d_z_series = get_inbetweens(parse_key_frames(self.rotation_3d_z), self.max_frames)
        self.perspective_flip_theta_series = get_inbetweens(parse_key_frames(self.perspective_flip_theta),
                                                            self.max_frames)
        self.perspective_flip_phi_series = get_inbetweens(parse_key_frames(self.perspective_flip_phi), self.max_frames)
        self.perspective_flip_gamma_series = get_inbetweens(parse_key_frames(self.perspective_flip_gamma),
                                                            self.max_frames)
        self.perspective_flip_fv_series = get_inbetweens(parse_key_frames(self.perspective_flip_fv), self.max_frames)
        self.noise_schedule_series = get_inbetweens(parse_key_frames(self.noise_schedule), self.max_frames)
        self.strength_schedule_series = get_inbetweens(parse_key_frames(self.strength_schedule), self.max_frames)
        self.contrast_schedule_series = get_inbetweens(parse_key_frames(self.contrast_schedule), self.max_frames)

    def deforumkeys(self, keyframes):

        gs.angle_series = get_inbetweens(parse_key_frames(keyframes.angle), keyframes.max_frames)
        gs.zoom_series = get_inbetweens(parse_key_frames(keyframes.zoom), keyframes.max_frames)
        gs.translation_x_series = get_inbetweens(parse_key_frames(keyframes.translation_x), keyframes.max_frames)
        gs.translation_y_series = get_inbetweens(parse_key_frames(keyframes.translation_y), keyframes.max_frames)
        gs.translation_z_series = get_inbetweens(parse_key_frames(keyframes.translation_z), keyframes.max_frames)
        gs.rotation_3d_x_series = get_inbetweens(parse_key_frames(keyframes.rotation_3d_x), keyframes.max_frames)
        gs.rotation_3d_y_series = get_inbetweens(parse_key_frames(keyframes.rotation_3d_y), keyframes.max_frames)
        gs.rotation_3d_z_series = get_inbetweens(parse_key_frames(keyframes.rotation_3d_z), keyframes.max_frames)
        gs.perspective_flip_theta_series = get_inbetweens(parse_key_frames(keyframes.perspective_flip_theta),
                                                          keyframes.max_frames)
        gs.perspective_flip_phi_series = get_inbetweens(parse_key_frames(keyframes.perspective_flip_phi), keyframes.max_frames)
        gs.perspective_flip_gamma_series = get_inbetweens(parse_key_frames(keyframes.perspective_flip_gamma),
                                                          keyframes.max_frames)
        gs.perspective_flip_fv_series = get_inbetweens(parse_key_frames(keyframes.perspective_flip_fv), keyframes.max_frames)
        gs.noise_schedule_series = get_inbetweens(parse_key_frames(keyframes.noise_schedule), keyframes.max_frames)
        gs.strength_schedule_series = get_inbetweens(parse_key_frames(keyframes.strength_schedule), keyframes.max_frames)
        gs.contrast_schedule_series = get_inbetweens(parse_key_frames(keyframes.contrast_schedule), keyframes.max_frames)

        # print(f"We should have a translation series of 10: {gs.translation_z_series}")

    def produce_video(self, image_path, mp4_path, max_frames, fps=12):
        print(f"{image_path} -> {mp4_path}")

        # make video
        cmd = [
            gs.system.ffmpegPath,
            '-y',
            '-vcodec', 'png',
            '-r', str(fps),
            '-start_number', str(0),
            '-i', image_path,
            '-frames:v', str(max_frames),
            '-c:v', 'libx264',
            '-vf',
            f'fps={fps}',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'veryslow',
            mp4_path
        ]
        print(cmd)
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

    def next_seed(self, seed_behavior, seed):
        if seed_behavior == 'iter':
            seed += 1
        else:
            seed = random.randint(0, 2**32 - 1)
        return seed

    def render_animation(self,
                         image_callback=None,
                         step_callback=None,
                         compviscallback=None,

                         # prompts="a beautiful forest by Asher Brown Durand, trending on Artstation",
                         animation_prompts='test',
                         H=512,
                         W=512,
                         seed=-1,  # @param
                         sampler_name='klms',
                         # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
                         steps=20,  # @param
                         scale=7,  # @param
                         ddim_eta=0.0,  # @param
                         # dynamic_threshold=None,
                         # static_threshold=None,
                         # @markdown **Save & Display Settings**
                         # save_samples=True,  # @param {type:"boolean"}
                         # save_settings=True,  # @param {type:"boolean"}
                         # display_samples=True,  # @param {type:"boolean"}
                         save_sample_per_step=False,  # @param {type:"boolean"}
                         show_sample_per_step=False,  # @param {type:"boolean"}
                         prompt_weighting=False,  # @param {type:"boolean"}
                         # normalize_prompt_weights=True,  # @param {type:"boolean"}
                         log_weighted_subprompts=False,  # @param {type:"boolean"}
                         adabins = False,
                         # n_batch=1,  # @param
                         batch_name="StableFun",  # @param {type:"string"}
                         # filename_format="{timestring}_{index}_{prompt}.png",
                         # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
                         seed_behavior="iter",  # @param ["iter","fixed","random"]
                         make_grid=False,  # @param {type:"boolean"}
                         # grid_rows=2,  # @param
                         outdir="output",
                         use_init=False,  # @param {type:"boolean"}
                         strength=0.0,  # @param {type:"number"}
                         strength_0_no_init=True,  # Set the strength to 0 automatically when no init image is used
                         init_image="",  # @param {type:"string"}
                         # Whiter areas of the mask are areas that change more
                         use_mask=False,  # @param {type:"boolean"}
                         use_alpha_as_mask=False,  # use the alpha channel of the init image as the mask

                         mask_file="",  # @param {type:"string"}
                         # invert_mask=False,  # @param {type:"boolean"}
                         # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                         mask_brightness_adjust=1.0,  # @param {type:"number"}
                         mask_contrast_adjust=1.0,  # @param {type:"number"}
                         # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
                         overlay_mask=True,  # {type:"boolean"}
                         # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
                         mask_overlay_blur=5,  # {type:"number"}

                         # n_samples=1,  # doesnt do anything
                         precision='autocast',
                         C=4,
                         f=8,

                         prompt="",
                         timestring="",
                         init_latent=None,
                         init_sample=None,
                         init_c=None,

                         # Anim Args

                         animation_mode='3D',
                         # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
                         max_frames=10,  # @Dparam {type:"number"}
                         border='replicate',  # @param ['wrap', 'replicate'] {type:'string'}
                         angle="0:(0)",  # @param {type:"string"}
                         zoom="0:(1.0)",  # @param {type:"string"}
                         translation_x="0:(10*sin(2*3.14*t/10))",  # @param {type:"string"}
                         translation_y="0:(2)",  # @param {type:"string"}
                         translation_z="0:(10)",  # @param {type:"string"}
                         rotation_3d_x="0:(0)",  # @param {type:"string"}
                         rotation_3d_y="0:(-7)",  # @param {type:"string"}
                         rotation_3d_z="0:(0)",  # @param {type:"string"}
                         flip_2d_perspective=False,  # @param {type:"boolean"}
                         perspective_flip_theta="0:(0)",  # @param {type:"string"}
                         perspective_flip_phi="0:(t%15)",  # @param {type:"string"}
                         perspective_flip_gamma="0:(0)",  # @param {type:"string"}
                         perspective_flip_fv="0:(53)",  # @param {type:"string"}
                         noise_schedule="0: (0.02)",  # @param {type:"string"}
                         strength_schedule="0: (0.65)",  # @param {type:"string"}
                         contrast_schedule="0: (1.0)",  # @param {type:"string"}
                         # @markdown ####**Coherence:**
                         color_coherence='Match Frame 0 LAB',
                         # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
                         diffusion_cadence=1,  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}
                         # @markdown ####**3D Depth Warping:**
                         use_depth_warping=True,  # @param {type:"boolean"}
                         midas_weight=0.3,  # @param {type:"number"}
                         near_plane=200,
                         far_plane=10000,
                         fov=40,  # @param {type:"number"}
                         padding_mode='border',  # @param ['border', 'reflection', 'zeros'] {type:'string'}
                         sampling_mode='bicubic',  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
                         save_depth_maps=False,  # @param {type:"boolean"}

                         # @markdown ####**Video Input:**
                         # video_init_path='/content/video_in.mp4',  # @param {type:"string"}
                         # extract_nth_frame=1,  # @param {type:"number"}
                         # overwrite_extracted_frames=True,  # @param {type:"boolean"}
                         use_mask_video=False,  # @param {type:"boolean"}
                         # video_mask_path='/content/video_in.mp4',  # @param {type:"string"}

                         # @markdown ####**Interpolation:**
                         # interpolate_key_frames=False,  # @param {type:"boolean"}
                         # interpolate_x_frames=4,  # @param {type:"number"}

                         # @markdown ####**Resume Animation:**
                         resume_from_timestring=False,  # @param {type:"boolean"}
                         resume_timestring="20220829210106",
                         # prev_sample=None,
                         clear_latent=False,
                         clear_sample=True,
                         shouldStop=False,
                         # keys={}
                         ):



        max_frames = max_frames

        near_plane = near_plane
        far_plane = far_plane
        fov = fov
        sampling_mode = sampling_mode
        padding_mode = padding_mode
        self.creation_timestamp = strftime("%d_%b_%Y_%H_%M_%S", gmtime())
        outdir = os.path.join(self.outdir, self.creation_timestamp)
        self.sampler_name = sampler_name

        print(sampler_name)


        scale = scale
        ddim_eta = ddim_eta
        # create output folder for the batch
        os.makedirs(outdir, exist_ok=True)
        print(f"Saving animation frames to {outdir}")


        if clear_latent:
            init_latent = None
            init_c = None
        if clear_sample:
            init_sample = None
            use_init = False

        # if not use_init:

        if "sd" not in gs.models:
            gs.models["sd"] = self.load_model()

        # animations use key framed prompts
        # prompts = animation_prompts

        # expand key frame strings to values
        anim_keyframes_dict = {
            'max_frames': max_frames,
            'angle': angle,
            'zoom': zoom,
            'translation_x': translation_x,
            'translation_y': translation_y,
            'translation_z': translation_z,
            'rotation_3d_x': rotation_3d_x,
            'rotation_3d_y': rotation_3d_y,
            'rotation_3d_z': rotation_3d_z,
            'perspective_flip_theta': perspective_flip_theta,
            'perspective_flip_phi': perspective_flip_phi,
            'perspective_flip_gamma': perspective_flip_gamma,
            'perspective_flip_fv': perspective_flip_fv,
            'noise_schedule': noise_schedule,
            'strength_schedule': strength_schedule,
            'contrast_schedule': contrast_schedule
        }

        anim_keyframes_dict = SimpleNamespace(**anim_keyframes_dict)
        self.deforumkeys(anim_keyframes_dict)

        # resume animation
        start_frame = 0
        if resume_from_timestring:
            for tmp in os.listdir(outdir):
                if tmp.split("_")[0] == resume_timestring:
                    start_frame += 1
            start_frame = start_frame - 1



        # save settings for the batch
        # settings_filename = os.path.join(outdir, f"{timestring}_settings.txt")
        # with open(settings_filename, "w+", encoding="utf-8") as f:
        #    s = {**dict(__dict__), **dict(__dict__)}
        #    json.dump(s, f, ensure_ascii=False, indent=4)

        # resume from timestring
        if resume_from_timestring:
            timestring = resume_timestring

        # expand prompts out to per-frame
        # prompt_series = pd.Series([np.nan for a in range(max_frames)])
        # for i, prompt in animation_prompts.items():
        #    prompt_series[i] = prompt
        prompt_series = animation_prompts

        # check for video inits
        using_vid_init = animation_mode == 'Video Input'

        # load depth model for 3D
        predict_depths = (animation_mode == '3D' and use_depth_warping) or save_depth_maps
        if predict_depths:
                depth_model = DepthModel('cuda')
                depth_model.load_midas('models/')
                if midas_weight < 1.0:
                    if adabins:
                        depth_model.load_adabins()
                    else:
                        gs.models["adabins"] = None
        else:
            depth_model = None
            save_depth_maps = False

        # state for interpolating between diffusion steps
        turbo_steps = 1 if using_vid_init else int(diffusion_cadence)
        turbo_prev_image, turbo_prev_frame_idx = None, 0
        turbo_next_image, turbo_next_frame_idx = None, 0

        # resume animation
        prev_sample = None
        color_match_sample = None
        if resume_from_timestring:
            last_frame = start_frame - 1
            if turbo_steps > 1:
                last_frame -= last_frame % turbo_steps
            path = os.path.join(outdir, f"{batch_name}_{timestring}_{last_frame:05}.png")
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            prev_sample = sample_from_cv2(img)
            if color_coherence != 'None':
                color_match_sample = img
            if turbo_steps > 1:
                turbo_next_image, turbo_next_frame_idx = sample_to_cv2(prev_sample, type=np.float32), last_frame
                turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                start_frame = last_frame + turbo_steps

        n_samples = 1
        frame_idx = start_frame
        while frame_idx < max_frames:
            if not self.shouldStop:
                print(f"Rendering animation frame {frame_idx} of {max_frames}")
                noise = gs.noise_schedule_series[frame_idx]
                strength = gs.strength_schedule_series[frame_idx]

                # print(f'keys: {keys}')
                # print(gs.contrast_schedule_series)
                # print(gs.contrast_schedule_series[frame_idx])

                contrast = gs.contrast_schedule_series[frame_idx]
                depth = None

                # emit in-between frames
                if turbo_steps > 1:
                    tween_frame_start_idx = max(0, frame_idx - turbo_steps)
                    for tween_frame_idx in range(tween_frame_start_idx, frame_idx):
                        self.torch_gc()
                        tween = float(tween_frame_idx - tween_frame_start_idx + 1) / float(
                            frame_idx - tween_frame_start_idx)
                        print(f"  creating in between frame {tween_frame_idx} tween:{tween:0.2f}")

                        advance_prev = turbo_prev_image is not None and tween_frame_idx > turbo_prev_frame_idx
                        advance_next = tween_frame_idx > turbo_next_frame_idx

                        if depth_model is not None:
                            assert (turbo_next_image is not None)
                            depth = depth_model.predict(turbo_next_image, midas_weight)

                        if animation_mode == '2D':
                            if advance_prev:
                                turbo_prev_image = anim_frame_warp_2d(turbo_prev_image, gs, tween_frame_idx, W, H,
                                                                      flip_2d_perspective, border)
                            if advance_next:
                                turbo_next_image = anim_frame_warp_2d(turbo_next_image, gs, tween_frame_idx, W, H,
                                                                      flip_2d_perspective, border)
                        else:  # '3D'
                            if advance_prev:
                                turbo_prev_image = anim_frame_warp_3d(turbo_prev_image, depth, gs, tween_frame_idx,
                                                                      near_plane, far_plane,
                                                                      fov, sampling_mode, padding_mode)
                            if advance_next:
                                turbo_next_image = anim_frame_warp_3d(turbo_next_image, depth, gs, tween_frame_idx,
                                                                      near_plane, far_plane,
                                                                      fov, sampling_mode, padding_mode)
                        turbo_prev_frame_idx = turbo_next_frame_idx = tween_frame_idx

                        if turbo_prev_image is not None and tween < 1.0:
                            img = turbo_prev_image * (1.0 - tween) + turbo_next_image * tween
                        else:
                            img = turbo_next_image

                        filename = f"{batch_name}_{timestring}_{tween_frame_idx:05}.png"
                        cv2.imwrite(os.path.join(outdir, filename),
                                    cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR))
                        if image_callback is not None:
                            # self.image_callback(Image.open(os.path.join(outdir, filename)))
                            image_callback(Image.fromarray(img.astype(np.uint8)))

                        if save_depth_maps:
                            depth_model.save(os.path.join(outdir, f"{timestring}_depth_{tween_frame_idx:05}.png"),
                                             depth)
                    if turbo_next_image is not None:
                        prev_sample = sample_from_cv2(turbo_next_image)

                # apply transforms to previous frame
                if prev_sample is not None:
                    if animation_mode == '2D':
                        prev_img = anim_frame_warp_2d(sample_to_cv2(prev_sample), gs, frame_idx, W, H,
                                                      flip_2d_perspective, border)
                    else:  # '3D'
                        prev_img_cv2 = sample_to_cv2(prev_sample)
                        depth = depth_model.predict(prev_img_cv2, midas_weight) if depth_model else None
                        prev_img = anim_frame_warp_3d(prev_img_cv2, depth, gs, frame_idx, near_plane, far_plane,
                                                      fov, sampling_mode, padding_mode)

                    # apply color matching
                    if color_coherence != 'None':
                        if color_match_sample is None:
                            color_match_sample = prev_img.copy()
                        else:
                            prev_img = maintain_colors(prev_img, color_match_sample, color_coherence)

                    # apply scaling
                    contrast_sample = prev_img * contrast
                    # apply frame noising
                    noised_sample = add_noise(sample_from_cv2(contrast_sample), noise)

                    # use transformed previous frame as init for current
                    use_init = True
                    half_precision = True
                    if half_precision:
                        init_sample = noised_sample.half().to('cuda')
                    else:
                        init_sample = noised_sample.to('cuda')
                    strength = max(0.0, min(1.0, strength))

                # grab prompt for current frame
                prompt = prompt_series[frame_idx]
                print(f"{prompt} {seed}")
                if not using_vid_init:
                    print(f"Angle: {gs.angle_series[frame_idx]} Zoom: {gs.zoom_series[frame_idx]}")
                    print(
                        f"Tx: {gs.translation_x_series[frame_idx]} Ty: {gs.translation_y_series[frame_idx]} Tz: {gs.translation_z_series[frame_idx]}")
                    print(
                        f"Rx: {gs.rotation_3d_x_series[frame_idx]} Ry: {gs.rotation_3d_y_series[frame_idx]} Rz: {gs.rotation_3d_z_series[frame_idx]}")

                # grab init image for current frame
                if using_vid_init:
                    init_frame = os.path.join(outdir, 'inputframes', f"{frame_idx + 1:05}.jpg")
                    print(f"Using video init frame {init_frame}")
                    init_image = init_frame
                    if use_mask_video:
                        mask_frame = os.path.join(outdir, 'maskframes', f"{frame_idx + 1:05}.jpg")
                        mask_file = mask_frame




                # sample the diffusion model
                sample, image = self.generate(frame=frame_idx,
                                              return_latent=False,
                                              return_sample=True,
                                              return_c=False,
                                              seed=seed,
                                              outdir=outdir,
                                              sampler_name=sampler_name,
                                              n_samples=n_samples,
                                              prompt=prompt,
                                              precision=precision,
                                              init_latent=init_latent,
                                              init_sample=init_sample,
                                              use_init=use_init,
                                              W=W,
                                              H=H,
                                              use_alpha_as_mask=use_alpha_as_mask,
                                              strength=strength,
                                              strength_0_no_init=strength_0_no_init,
                                              use_mask=use_mask,
                                              mask_file=mask_file,
                                              mask_contrast_adjust=mask_contrast_adjust,
                                              mask_brightness_adjust=mask_brightness_adjust,
                                              overlay_mask=overlay_mask,
                                              steps=steps,
                                              ddim_eta=ddim_eta,
                                              save_sample_per_step=save_sample_per_step,
                                              show_sample_per_step=show_sample_per_step,
                                              timestring=timestring,
                                              prompt_weighting=prompt_weighting,
                                              log_weighted_subprompts=log_weighted_subprompts,
                                              scale=scale,
                                              init_c=init_c,
                                              C=C,
                                              f=f,
                                              mask_overlay_blur=mask_overlay_blur,
                                              step_callback=step_callback,
                                              compviscallback=compviscallback)

                if image_callback is not None and diffusion_cadence == 0:
                    image_callback(image, seed, upscaled=False)

                if not using_vid_init:
                    prev_sample = sample

                if turbo_steps > 1:
                    turbo_prev_image, turbo_prev_frame_idx = turbo_next_image, turbo_next_frame_idx
                    turbo_next_image, turbo_next_frame_idx = sample_to_cv2(sample, type=np.float32), frame_idx
                    frame_idx += turbo_steps
                else:
                    filename = f"{batch_name}_{timestring}_{frame_idx:05}.png"
                    image.save(os.path.join(outdir, filename))
                    if save_depth_maps:
                        if depth is None:
                            depth = depth_model.predict(sample_to_cv2(sample), self)
                        depth_model.save(
                            os.path.join(outdir, f"{batch_name}_{timestring}_depth_{frame_idx:05}.png"), depth)
                    frame_idx += 1

                seed = self.next_seed(seed_behavior, seed)
                image_path = os.path.join(outdir, f"{batch_name}_{timestring}_%05d.png")
                mp4_path = os.path.join(gs.system.txt2vidOut, f"{batch_name}_{self.creation_timestamp}.mp4")
                #self.signals.deforum_image_cb.emit()

            else:
                mp4_path = os.path.join(gs.system.txt2vidOut, f"{batch_name}_{self.creation_timestamp}.mp4")
                image_path = os.path.join(outdir, f"{batch_name}_{timestring}_%05d.png")
                max_frames = frame_idx
                self.produce_video(image_path, mp4_path, max_frames)

    # max_frames = frame_idx

        if self.shouldStop == False:
            self.produce_video(image_path, mp4_path, max_frames)
        try:
            del depth_model
        except:
            pass
        try:

            del gs.models["midas_model"]
            del gs.models["adabins"]
        except:
            pass
        self.torch_gc()

    def generate(self,
                 seed,
                 outdir,
                 sampler_name,
                 n_samples,
                 prompt,
                 precision,
                 init_latent,
                 init_sample,
                 use_init,
                 W,
                 H,
                 use_alpha_as_mask,
                 strength,
                 strength_0_no_init,
                 use_mask,
                 mask_file,
                 mask_contrast_adjust,
                 mask_brightness_adjust,
                 overlay_mask,
                 steps,
                 ddim_eta,
                 save_sample_per_step,
                 show_sample_per_step,
                 timestring,
                 prompt_weighting,
                 log_weighted_subprompts,
                 scale,
                 init_c,
                 C,
                 f,
                 mask_overlay_blur,
                 step_callback,
                 compviscallback,
                 frame=0,
                 return_latent=False,
                 return_sample=False,
                 return_c=False,):
        seed_everything(seed)
        os.makedirs(outdir, exist_ok=True)

        sampler = PLMSSampler(gs.models["sd"]) if sampler_name == 'plms' else DDIMSampler(gs.models["sd"])
        model_wrap = CompVisDenoiser(gs.models["sd"])
        batch_size = n_samples
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
        precision_scope = autocast if precision == "autocast" else nullcontext

        init_latent = None
        mask_image = None
        init_image = None

        if init_latent is not None:
            print('init_latent is not None')
            init_latent = init_latent
        elif init_sample is not None:
            print('init_sample is not None')
            with precision_scope("cuda"):
                init_latent = gs.models["sd"].get_first_stage_encoding(
                    gs.models["sd"].encode_first_stage(init_sample))
        elif use_init and init_image is not None and init_image != '':
            print('use_init and init_image is not None and init_image != ""')
            init_image, mask_image = load_img(init_image,
                                              shape=(W, H),
                                              use_alpha_as_mask=use_alpha_as_mask)
            init_image = init_image.to(self.device)
            init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            with precision_scope("cuda"):
                init_latent = gs.models["sd"].get_first_stage_encoding(
                    gs.models["sd"].encode_first_stage(init_image))  # move to latent space

        if not use_init and strength > 0 and strength_0_no_init:
            print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
            print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
            strength = 0

        # Mask functions
        if use_mask:
            assert mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
            assert use_init, "use_mask==True: use_init is required for a mask"
            assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

            mask = prepare_mask(mask_file if mask_image is None else mask_image,
                                init_latent.shape,
                                mask_contrast_adjust,
                                mask_brightness_adjust)

            if (torch.all(mask == 0) or torch.all(mask == 1)) and use_alpha_as_mask:
                raise Warning(
                    "use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

            mask = mask.to(self.device)
            mask = repeat(mask, '1 ... -> b ...', b=batch_size)
        else:
            mask = None

        assert not ((use_mask and overlay_mask) and (
                init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        t_enc = int((1.0 - strength) * steps)

        # Noise schedule for the k-diffusion samplers (used for masking)
        k_sigmas = model_wrap.get_sigmas(steps)
        k_sigmas = k_sigmas[len(k_sigmas) - t_enc - 1:]

        if sampler_name in ['plms', 'ddim']:
            sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, ddim_discretize='quad',
                                  verbose=False)

        """callback = SamplerCallback(n_samples=n_samples,
                                   save_sample_per_step=save_sample_per_step,
                                   show_sample_per_step=show_sample_per_step,
                                   outdir=outdir,
                                   sampler_name=sampler_name,
                                   timestring=timestring,
                                   seed=seed,
                                   sigmas=k_sigmas,
                                   verbose=False,
                                   step_callback=step_callback,
                                   init_latent=init_latent).callback"""



        results = []
        with torch.no_grad():
            with precision_scope("cuda"):
                with gs.models["sd"].ema_scope():
                    for prompts in data:
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        if prompt_weighting:
                            uc, c = get_uc_and_c(prompts, gs.models["sd"], n_samples,
                                                           log_weighted_subprompts, normalize_prompt_weights,
                                                           frame)


                        else:
                            uc = gs.models["sd"].get_learned_conditioning(batch_size * [""])
                            c = gs.models["sd"].get_learned_conditioning(prompts)

                        if scale == 1.0:
                            uc = None
                        if init_c is not None:
                            print('init_c not none')
                            c = init_c

                        if sampler_name in ["klms", "dpm2", "dpm2_ancestral", "heun", "euler", "euler_ancestral"]:
                            samples = self.sampler_fn(init_latent,
                                                      C,
                                                      H,
                                                      W,
                                                      f,
                                                      t_enc,
                                                      model_wrap,
                                                      steps,
                                                      use_init,
                                                      n_samples,
                                                      self.device,
                                                      c,
                                                      uc,
                                                      scale,
                                                      step_callback,
                                                      sampler_name)
                        else:
                            # self.sampler == 'plms' or self.sampler == 'ddim':
                            if init_latent is not None and strength > 0:
                                z_enc = sampler.stochastic_encode(init_latent,
                                                                       torch.tensor([t_enc] * batch_size).to(
                                                                           self.device))
                            else:

                                z_enc = torch.randn([n_samples, C, H // f, W // f],
                                                         device=self.device)
                            if sampler_name == 'ddim':
                                samples = sampler.decode(z_enc,
                                                         c,
                                                         t_enc,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         img_callback=compviscallback)
                            elif sampler_name == 'plms':  # no "decode" function in plms, so use "sample"
                                shape = [C, H // f, W // f]
                                samples, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=compviscallback)
                            else:
                                raise Exception(f"Sampler {sampler} not recognised.")

                        if return_latent:
                            results.append(samples.clone())

                        x_samples = gs.models["sd"].decode_first_stage(samples)

                        if use_mask and overlay_mask:
                            # Overlay the masked image after the image is generated
                            if init_sample is not None:
                                img_original = init_sample
                            elif init_image is not None:
                                img_original = init_image
                            else:
                                raise Exception("Cannot overlay the masked image without an init image to overlay")

                            mask_fullres = prepare_mask(mask_file if mask_image is None else mask_image,
                                                        img_original.shape,
                                                        mask_contrast_adjust,
                                                        mask_brightness_adjust)
                            mask_fullres = mask_fullres[:, :3, :, :]
                            mask_fullres = repeat(mask_fullres, '1 ... -> b ...', b=batch_size)

                            mask_fullres[mask_fullres < mask_fullres.max()] = 0
                            mask_fullres = gaussian_filter(mask_fullres, mask_overlay_blur)
                            mask_fullres = torch.Tensor(mask_fullres).to(self.device)

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
                    self.torch_gc()
        return results

    def sampler_fn(self,
                   init_latent,
                   C,
                   H,
                   W,
                   f,
                   t_enc,
                   model_wrap,
                   steps,
                   use_init,
                   n_samples,
                   device,
                   c,
                   uc,
                   scale,
                   step_callback,
                   sampler) -> torch.Tensor:
        shape = [C, H // f, W // f]
        sigmas: torch.Tensor = model_wrap.get_sigmas(steps)
        sigmas = sigmas[len(sigmas) - t_enc - 1:]
        if use_init:
            print("use_init")
            if len(sigmas) > 0:
                x = (
                        init_latent
                        + torch.randn([n_samples, *shape], device=device) * sigmas[0]
                )
            else:
                print("use_init_latent")
                x = init_latent
        else:
            if len(sigmas) > 0:
                print("len(sigmas) > 0")
                x = torch.randn([n_samples, *shape], device='cuda') * sigmas[0]
            else:
                print("torch.zeros")
                x = torch.zeros([n_samples, *shape], device='cuda')

        sampler_args = {
            "model": CFGDenoiser(model_wrap),
            "x": x,
            "sigmas": sigmas,
            "extra_args": {"cond": c, "uncond": uc, "cond_scale": scale},
            "disable": False,
            "callback": step_callback,
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

    def _load_model_from_config(self, config, ckpt):
        print(f'>> Loading model from {ckpt}')

        # for usage statistics
        device_type = choose_torch_device()
        if device_type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        tic = time.time()

        # this does the work
        pl_sd = torch.load(ckpt, map_location='cpu')
        sd = pl_sd['state_dict']
        model = instantiate_from_config(config.model)
        m, u = model.load_state_dict(sd, strict=False)

        # if self.full_precision:
        #    print(
        #        '>> Using slower but more accurate full-precision math (--full_precision)'
        #    )
        # else:
        #    print(
        #        '>> Using half precision math. Call with --full_precision to use more accurate but VRAM-intensive full precision.'
        #    )
        model.half()
        model.to(self.device)
        model.eval()

        # usage statistics
        toc = time.time()
        print(
            f'>> Model loaded in', '%4.2fs' % (toc - tic)
        )
        if device_type == 'cuda':
            print(
                '>> Max VRAM used to load the model:',
                '%4.2fG' % (torch.cuda.max_memory_allocated() / 1e9),
                '\n>> Current VRAM usage:'
                '%4.2fG' % (torch.cuda.memory_allocated() / 1e9),
            )
        return model


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


class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(self, n_samples, save_sample_per_step, show_sample_per_step, outdir, sampler_name, timestring, seed,
                 mask=None, init_latent=None, sigmas=None, sampler=None,
                 step_callback=None, verbose=False):  # , *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.sampler_name = sampler_name
        # self.dynamic_threshold = dynamic_threshold
        # self.static_threshold = static_threshold
        self.mask = mask
        self.init_latent = init_latent
        self.sampler = sampler
        self.verbose = verbose
        self.step_callback = step_callback
        self.batch_size = n_samples
        self.save_sample_per_step = save_sample_per_step
        self.show_sample_per_step = show_sample_per_step
        self.paths_to_image_steps = [os.path.join(outdir, f"{timestring}_{index:02}_{seed}") for index in
                                     range(n_samples)]

        if save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0
        print('was here')
        self.noise = None
        if init_latent is not None:
            print('new noise')
            self.noise = torch.randn_like(init_latent, device=torch.device('cuda'))

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas / torch.max(sigmas))
        elif len(sigmas) == 0:
            self.mask = None  # no mask needed if no steps (usually happens because strength==1.0)

        if sampler_name in ["plms", "ddim"]:
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else:
            # Default callback function uses k-diffusion sampler variables
            print('k_callback_')
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, latents, path_name_modifier=''):

        print("view_sample_step")
        if self.save_sample_per_step or self.show_sample_per_step:
            self.step_callback(latents)
            # samples = gs.models["sd"].decode_first_stage(latents)
            if self.save_sample_per_step:
                fname = f'{path_name_modifier}_{self.step_index:05}.png'
                for i, sample in enumerate(samples):
                    sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                    sample = torch.tensor(np.array(sample))
                    grid = make_grid(sample, 4).cpu()
                    TF.to_pil_image(grid).save(os.path.join(self.paths_to_image_steps[i], fname))
            if self.show_sample_per_step:
                print(path_name_modifier)
                # self.display_images(samples)
        #return latents

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        img = (TF.to_pil_image(grid))
        return img

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1, img.ndim)))
        s = np.max(np.append(s, 1.0))
        torch.clamp_(img, -1 * s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        print('k_callback_')
        self.step_index = args_dict['i']
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict['x'], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(args_dict['x'], -1 * self.static_threshold, self.static_threshold)
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict['sigma']
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[args_dict['i']], self.mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + args_dict['x'] * torch.where(is_masked, 0, 1)
            args_dict['x'].copy_(new_img)

        self.view_sample_step(args_dict['denoised'], "x0_pred")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, i):
        print('img_callback')
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1 * self.static_threshold, self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(self.init_latent,
                                                        torch.tensor([i_inv] * self.batch_size).to(self.device),
                                                        noise=self.noise)
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[i], self.mask != 0)
            new_img = init_noise * torch.where(is_masked, 1, 0) + img * torch.where(is_masked, 0, 1)
            img.copy_(new_img)

        self.view_sample_step(img, "x")
