import gc, os, random, sys, time, traceback
from contextlib import nullcontext
from datetime import datetime

import clip
import numpy as np
import pandas as pd
import torch
from PIL import ImageFilter
from backend.devices import choose_torch_device
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn, autocast
from torchvision.utils import save_image, make_grid
from tqdm import tqdm, trange
from einops import rearrange, repeat
from PIL import Image
from backend.deforum.deforum_args import prepare_args
from backend.deforum.six.aesthetics import load_aesthetics_model
from backend.deforum.six.model_load import make_linear_decode
from backend.deforum.six.render import render_animation, render_input_video, render_image_batch, render_interpolation
from backend.ddim_outpaint import DDIMSampler
from backend.toxicode_utils import metadata, get_mask_for_latent_blending
from backend.utils import sampleToImage, encoded_to_torch_image, image_path_to_torch, \
    get_conditionings, torch_image_to_latent, get_prompts_data
from backend.ddim_simplified import DDIMSampler_simple
from backend.torch_gc import torch_gc
from ldm_deforum.modules.embedding_managerpt import EmbeddingManager
from ldm_v2.util import instantiate_from_config
from backend.hypernetworks import hypernetwork
import backend.hypernetworks.modules.sd_hijack
from backend.deforum.six.hijack import hijack_deforum
from backend.singleton import singleton
from backend.shared import model_killer
from backend.deforum.six.seamless import configure_model_padding
from backend.aesthetics.aesthetic_clip import AestheticCLIP

gs = singleton

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}


default_vae_dict = {"auto": "auto", "None": "None"}
default_vae_list = ["auto", "None"]


default_vae_values = [default_vae_dict[x] for x in default_vae_list]
vae_dict = dict(default_vae_dict)
vae_list = list(default_vae_list)
first_load = True


base_vae = None
loaded_vae_file = None
checkpoint_info = None
def load_model_from_config_lm(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

class DeforumSix:

    def __init__(self, parent):
        self.parent = parent
        self.root = None
        self.args = None
        self.anim_args = None
        # self.parent = parent
        self.device = choose_torch_device()
        self.full_precision = False
        self.prev_seamless = False
    def load_low_memory(self):
        if "model" not in gs.models:
            config = "optimizedSD/v1-inference.yaml"
            ckpt = gs.system.sdPath
            sd = load_model_from_config_lm(f"{ckpt}")
            li, lo = [], []
            for key, v_ in sd.items():
                sp = key.split(".")
                if (sp[0]) == "model":
                    if "input_blocks" in sp:
                        li.append(key)
                    elif "middle_block" in sp:
                        li.append(key)
                    elif "time_embed" in sp:
                        li.append(key)
                    else:
                        lo.append(key)
            for key in li:
                sd["model1." + key[6:]] = sd.pop(key)
            for key in lo:
                sd["model2." + key[6:]] = sd.pop(key)

            config = OmegaConf.load(f"{config}")

            gs.models["model"] = instantiate_from_config(config.modelUNet)
            _, _ = gs.models["model"].load_state_dict(sd, strict=False)
            gs.models["model"].eval()

            gs.models["modelCS"] = instantiate_from_config(config.modelCondStage)
            _, _ = gs.models["modelCS"].load_state_dict(sd, strict=False)
            gs.models["modelCS"].eval()

            gs.models["modelFS"] = instantiate_from_config(config.modelFirstStage)
            _, _ = gs.models["modelFS"].load_state_dict(sd, strict=False)
            gs.models["modelFS"].eval()
            gs.models["model"].unet_bs = 1
            gs.models["model"].turbo = False
            gs.models["model"].cdevice = "cuda"
            gs.models["modelCS"].cond_stage_model.device = "cuda"
            del li
            del lo
            del sd

    def run_post_load_model_generation_specifics(self):

        #print("Loading Hypaaaa")
        gs.model_hijack = backend.hypernetworks.modules.sd_hijack.StableDiffusionModelHijack()

        print("hijacking??")
        gs.model_hijack.hijack(gs.models["sd"])
        gs.model_hijack.embedding_db.load_textual_inversion_embeddings()

        #gs.models["sd"].cond_stage_model = backend.aesthetics.modules.PersonalizedCLIPEmbedder()

        aesthetic = AestheticCLIP()
        aesthetic.process_tokens = gs.models["sd"].cond_stage_model.process_tokens
        gs.models["sd"].cond_stage_model.process_tokens = aesthetic


    def get_autoencoder_version(self):
        return "sd-v1" #TODO this will be different for different models

    def load_model_from_config(self, config=None, ckpt=None, verbose=False):

        if ckpt is None:
            ckpt = gs.system.sdPath

        # loads config.yaml with the name of the model
        # the config yaml has to be provided with pöropper naming,
        # otherwise it is not anymore possible to do all the magic with multiple versions of the model around
        # also config.yaml needs to have one entry at root model_version
        # model_version has to be explicid like 1.4 or 1.5 or 2.0
        # it is important that you give the right version hint based on the SD model version
        # if it is some custom model based on some version of SD we need to have the SD
        # version not the version of the custom model
        #if config is None:
        config_yaml_name = os.path.splitext(ckpt)[0] + '.yaml'
        #print(config_yaml_name)
        #else:
        #    config_yaml_name = config
        #print(os.path.isfile(config_yaml_name))
        #if os.path.isfile(config_yaml_name):
        config = config_yaml_name

        if "sd" not in gs.models:
            self.prev_seamless = False
            if verbose:
                print(f"Loading model from {ckpt} with config {config}")
            config = OmegaConf.load(config)

            #print(config.model['params'])

            if 'num_heads' in config.model['params']['unet_config']['params']:
                gs.model_version = '1.5'
            elif 'num_head_channels' in config.model['params']['unet_config']['params']:
                gs.model_version = '2.0'
            if config.model['params']['conditioning_key'] == 'hybrid-adm':
                gs.model_version = '2.0'
            if 'parameterization' in config.model['params']:
                gs.model_resolution = 768
            else:
                gs.model_resolution = 512
            #if not 'model_version' in config:
            #    print('you must provide a model_version in the config yaml or we can not figure how to tread your model')
            #    return -1
            print(f'v {gs.model_version} found with resolution {gs.model_resolution}')

            #gs.model_version = config.model_version
            if verbose:
                print(gs.model_version)


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
            model.half()
            gs.models["sd"] = model
            gs.models["sd"].cond_stage_model.device = self.device
            #gs.models["sd"].embedding_manager = EmbeddingManager(gs.models["sd"].cond_stage_model)
            #embedding_path = '001glitch-core.pt'
            #if embedding_path is not None:
            #    gs.models["sd"].embedding_manager.load(
            #        embedding_path
            #    )

            for m in gs.models["sd"].modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    m._orig_padding_mode = m.padding_mode

            autoencoder_version = self.get_autoencoder_version()

            gs.models["sd"].linear_decode = make_linear_decode(autoencoder_version, self.device)
            del pl_sd
            del sd
            del m, u
            del model
            torch_gc()


            if gs.model_version == '1.5':
              self.run_post_load_model_generation_specifics()

            gs.models["sd"].eval()

            # todo make this 'cuda' a parameter
            gs.models["sd"].to(self.device)
            # todo why we do this here?
            from backend.aesthetics import modules
            if gs.diffusion.selected_vae != 'None':
                self.load_vae(gs.diffusion.selected_vae)

    def load_vae(self, vae_file=None):
        global first_load, vae_dict, vae_list, loaded_vae_file
        # save_settings = False

        if os.path.isfile(vae_file):
            assert os.path.isfile(vae_file), f"VAE file doesn't exist: {vae_file}"
            print(f"Loading VAE weights from: {vae_file}")
            vae_ckpt = torch.load(vae_file, map_location='cpu')
            vae_dict_1 = {k: v for k, v in vae_ckpt["state_dict"].items() if
                          k[0:4] != "loss" and k not in vae_ignore_keys}
            load_vae_dict(gs.models["sd"], vae_dict_1)

            # If vae used is not in dict, update it
            # It will be removed on refresh though
            #vae_opt = get_filename(vae_file)
            #if vae_opt not in vae_dict:
            #    vae_dict[vae_opt] = vae_file
            #    vae_list.append(vae_opt)
        else:
            print(f"VAE file doesn't exist: {vae_file}")

        loaded_vae_file = vae_file

        """
        # Save current VAE to VAE settings, maybe? will it work?
        if save_settings:
            if vae_file is None:
                vae_opt = "None"
            # shared.opts.sd_vae = vae_opt
        """

        first_load = False


    def load_model(self):
        """Load and initialize the model from configuration variables passed at object creation time"""
        if "inpaint" in gs.models:
            gs.models["inpaint"].to("cpu")
            del gs.models["inpaint"]
            torch_gc()
        weights = gs.system.sdPath
        config = 'configs/stable-diffusion/v1-inference-a.yaml'
        embedding_path = None

        """Load and initialize the model from configuration variables passed at object creation time"""
        seed_everything(random.randrange(0, np.iinfo(np.uint32).max))
        try:

            self._load_model_from_config(config, weights)
            if embedding_path is not None:
                gs.models["sd"].embedding_manager.load(
                    embedding_path, self.full_precision
                )
            # model = model.half().to(self.device)
            # model.to doesn't change the cond_stage_model.device used to move the tokenizer output, so set it here
            gs.models["sd"].cond_stage_model.device = self.device
        except AttributeError as e:
            print(f'>> Error loading model. {str(e)}', file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            raise SystemExit from e

        # self._set_sampler()

        for m in gs.models["sd"].modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode
        autoencoder_version = "sd-v1" #TODO this will be different for different models
        gs.models["sd"].linear_decode = make_linear_decode(autoencoder_version, self.device)
        # return model

    def load_inpaint_model(self):
        if "sd" in gs.models:
            gs.models["sd"].to('cpu')
            del gs.models["sd"]
            torch_gc()
        if "custom_model_name" in gs.models:
            del gs.models["custom_model_name"]
            torch_gc()
        """Load and initialize the model from configuration variables passed at object creation time"""
        if "inpaint" not in gs.models:
            weights = 'models/sd-v1-5-inpaint.ckpt'
            config = 'configs/stable-diffusion/inpaint.yaml'
            embedding_path = None

            config = OmegaConf.load(config)

            model = instantiate_from_config(config.model)

            model.load_state_dict(torch.load(weights)["state_dict"], strict=False)

            device = self.device
            gs.models["inpaint"] = model.half().to(device)
            del model
            return

    def run_deforum_six(self,
                        image_callback=None,
                        step_callback=None,
                        # prompts="a beautiful forest by Asher Brown Durand, trending on Artstation",
                        prompts='test',
                        keyframes='0',
                        H=512,
                        W=512,
                        seed=-1,  # @param
                        sampler='klms',
                        # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
                        steps=20,  # @param
                        scale=7,  # @param
                        ddim_eta=0.0,  # @param
                        dynamic_threshold=None,
                        static_threshold=None,
                        # @markdown **Save & Display Settings**
                        save_samples=True,  # @param {type:"boolean"}
                        save_settings=True,  # @param {type:"boolean"}
                        display_samples=False,  # @param {type:"boolean"}
                        save_sample_per_step=False,  # @param {type:"boolean"}
                        show_sample_per_step=False,  # @param {type:"boolean"}
                        prompt_weighting=False,  # @param {type:"boolean"}
                        # normalize_prompt_weights=True,  # @param {type:"boolean"}
                        log_weighted_subprompts=False,  # @param {type:"boolean"}
                        adabins=False,
                        n_batch=1,  # @param
                        batch_name="StableFun",  # @param {type:"string"}
                        filename_format="{timestring}_{index}_{prompt}.png",
                        # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
                        seed_behavior="iter",  # @param ["iter","fixed","random"]
                        make_grid=False,  # @param {type:"boolean"}
                        grid_rows=2,  # @param
                        outdir="output",
                        use_init=False,  # @param {type:"boolean"}
                        strength=0.0,  # @param {type:"number"}
                        strength_0_no_init=True,  # Set the strength to 0 automatically when no init image is used
                        init_image="",  # @param {type:"string"}
                        # Whiter areas of the mask are areas that change more
                        use_mask=False,  # @param {type:"boolean"}
                        use_alpha_as_mask=True,  # use the alpha channel of the init image as the mask
                        mask_file="",  # @param {type:"string"}
                        invert_mask=False,  # @param {type:"boolean"}
                        # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
                        mask_brightness_adjust=1.0,  # @param {type:"number"}
                        mask_contrast_adjust=1.0,  # @param {type:"number"}
                        # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
                        overlay_mask=True,  # {type:"boolean"}
                        # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
                        mask_overlay_blur=5,  # {type:"number"}

                        n_samples=1,  # doesnt do anything
                        precision='autocast',

                        # prompt="",
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
                        diffusion_cadence=1,  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

                        # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}

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
                        video_init_path='/content/video_in.mp4',  # @param {type:"string"}
                        extract_nth_frame=1,  # @param {type:"number"}
                        overwrite_extracted_frames=True,  # @param {type:"boolean"}
                        use_mask_video=False,  # @param {type:"boolean"}
                        video_mask_path='/content/video_in.mp4',  # @param {type:"string"}

                        # @markdown ####**Interpolation:**
                        interpolate_key_frames=False,  # @param {type:"boolean"}
                        interpolate_x_frames=4,  # @param {type:"number"}

                        # @markdown ####**Resume Animation:**
                        resume_from_timestring=False,  # @param {type:"boolean"}
                        resume_timestring="20220829210106",
                        # prev_sample=None,
                        clear_latent=False,
                        clear_sample=True,
                        shouldStop=False,
                        # keys={}
                        cpudepth=False,
                        device='cuda',

                        normalize_prompt_weights=False,

                        # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.

                        mean_scale=0,
                        var_scale=0,
                        exposure_scale=0,
                        exposure_target=0,
                        colormatch_scale=0,
                        colormatch_image="https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png",
                        colormatch_n_colors=0,
                        ignore_sat_weight=0,
                        clip_name='ViT-L/14',  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                        clip_scale=0,
                        aesthetics_scale=0,
                        cutn=0,
                        cut_pow=0.0,
                        init_mse_scale=0,
                        init_mse_image="https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg",
                        blue_scale=0,
                        gradient_wrt='x0_pred', # ["x", "x0_pred"]
                        gradient_add_to='both', # ["cond", "uncond", "both"]
                        decode_method='linear',# ["autoencoder","linear"]
                        grad_threshold_type='dynamic', #["dynamic", "static", "mean", "schedule"]
                        clamp_grad_threshold=0.0,
                        clamp_start=0.0,
                        clamp_stop=0.0,
                        grad_inject_timing=None, #Number? I think.. INT
                        cond_uncond_sync=False,
                        skip_video_for_run_all=True,
                        C=4,
                        f=8,
                        prompt="",
                        negative_prompts=None,
                        hires=None,
                        #use_hypernetwork=None,
                        apply_strength=0,
                        apply_circular=False,
                        lowmem=False
                        ):
        #if gs.system.xformer == True:
        #    backend.hypernetworks.modules.sd_hijack.apply_optimizations()
        gs.system.device = choose_torch_device()


        print(f'-       Deforum  0.6  Art Generator                            ')
        print(f'-            animation mode: {animation_mode}                  ')
        print(f'-            steps: {steps}                                    ')
        print(f'-            width: {W}                                        ')
        print(f'-            height: {H}                                       ')
        print(f'-            hires: {hires}                                    ')
        print(f'-                                                              ')



        #print(f'animation mode: {animation_mode}')

        if precision == 'autocast' and device != "cpu":
            precision_scope = autocast
        else:
            precision_scope = nullcontext


        hijack_deforum.deforum_hijack()

        [args, anim_args, root] = prepare_args(locals())
        root.device = self.device
        device = self.device
        for key, value in anim_args.__dict__.items():
            try:
                anim_args.__dict__[key] = self.parent.params.__dict__[key]
                print(f"settings {key} from {value} to {self.parent.params.__dict__[key]}")
            except:
                pass
        for key, value in args.__dict__.items():
            try:
                args.__dict__[key] = self.parent.params.__dict__[key]
            except:
                pass
        print(args.make_grid)
        print(self.parent.params.make_grid)
        #if args.seamless == False and self.prev_seamless == True:
        #    self.prev_seamless = False
        #    model_killer()
        if lowmem == True:
            print(f'-                 Low Memory Mode                             ')
            if "sd" in gs.models:
                del gs.models["sd"]
            if "inpaint" in gs.models:
                del gs.models["inpaint"]
            if "custom_model_name" in gs.models:
                del gs.models["custom_model_name"]
                gs.models["sd"] = None
            if 'model' not in gs.models:
                self.load_low_memory()
        else:
            if "model" in gs.models:
                del gs.models["model"]
            if "modelCS" in gs.models:
                del gs.models["modelCS"]
            if "modelFS" in gs.models:
                del gs.models["modelFS"]
            check = self.load_model_from_config(config=None, ckpt=None)
            if check == -1:
                return check



        if gs.diffusion.selected_hypernetwork != 'None':
            hypernetwork.load_hypernetwork(gs.diffusion.selected_hypernetwork)
            hypernetwork.apply_strength(apply_strength)                          #1.0, "Hypernetwork strength", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.001}),
            gs.model_hijack.apply_circular(False)
            gs.model_hijack.clear_comments()

        #W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64




        #if args.seamless == True and self.prev_seamless == False:

        #print("Running Seamless sampling...")
        seamless = args.seamless
        seamless_axes = args.axis
        configure_model_padding(gs.models["sd"], seamless, seamless_axes)
        #self.prev_seamless = True
        """
        for key, value in root.__dict__.items():
            try:
                root.__dict__[key] = self.parent.params.__dict__[key]
            except:
                pass"""

        if gs.diffusion.selected_aesthetic_embedding != 'None':
            gs.models["sd"].cond_stage_model.process_tokens.set_aesthetic_params(
                aesthetic_lr=gs.lr,
                aesthetic_weight=gs.aesthetic_weight,
                aesthetic_steps=gs.T,
                image_embs_name=gs.diffusion.selected_aesthetic_embedding,
                aesthetic_slerp=gs.slerp,
                aesthetic_imgs_text=gs.aesthetic_imgs_text,
                aesthetic_slerp_angle=gs.slerp_angle,
                aesthetic_text_negative=gs.aesthetic_text_negative)

        if hires:
            args.hiresstr = args.strength

        if not use_init:
            init_image = None
        args.strength = 0 if init_image is None else strength

        root.device = device
        root.models_path = 'models'
        root.output_path = args.outdir
        root.half_precision = True
        #Mod 2, animation prompt parsing

        if anim_args.animation_mode != 'None':
            prompt_series = pd.Series([np.nan for a in range(max_frames)])
            if keyframes == '':
                keyframes = "0"
            prom = prompts
            key = keyframes

            new_prom = list(prom.split("\n"))
            new_key = list(key.split("\n"))

            prompts = dict(zip(new_key, new_prom))

            for i, prompt in prompts.items():
                n = int(i)
                prompt_series[n] = prompt
            animation_prompts = prompt_series.ffill().bfill()
        else:
            prompts = list(prompts.split("\n"))

        animation_prompts = prompts

        # Load clip model if using clip guidance
        if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
            root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(device)
            if (args.aesthetics_scale > 0):
                root.aesthetics_model = load_aesthetics_model(args, root)

        if args.seed == -1:
            args.seed = random.randint(0, 2 ** 32 - 1)
        if not args.use_init:
            args.init_image = None
        if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
            print(f"Init images aren't supported with PLMS yet, switching to KLMS")
            args.sampler = 'klms'
        if args.sampler != 'ddim':
            args.ddim_eta = 0

        if anim_args.animation_mode == 'None':
            anim_args.max_frames = 1
        elif anim_args.animation_mode == 'Video Input':
            args.use_init = True

        # clean up unused memory
        torch_gc()

        args.clip_prompt = ['test']

        #print('anim_args.animation_mode', anim_args.animation_mode)
        #print('anim_args.translation_x', anim_args.translation_x)
        paths = []
        # dispatch to appropriate renderer
        if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
            render_animation(args, anim_args, animation_prompts, root, image_callback=image_callback, step_callback=step_callback)
        elif anim_args.animation_mode == 'Video Input':
            render_input_video(args, anim_args, animation_prompts, root, image_callback=image_callback)
        elif anim_args.animation_mode == 'Interpolation':
            render_interpolation(args, anim_args, animation_prompts, root, image_callback=image_callback)
        else:
            #print(prompts)
            paths = render_image_batch(args, prompts, root, image_callback=image_callback, step_callback=step_callback)

        #skip_video_for_run_all = True  # @param {type: 'boolean'}
        fps = 12  # @param {type:"number"}
        # @markdown **Manual Settings**
        use_manual_settings = False  # @param {type:"boolean"}
        image_path = gs.system.txt2vidSingleFrame  # @param {type:"string"}
        mp4_path = gs.system.txt2vidOut  # @param {type:"string"}
        render_steps = False  # @param {type: 'boolean'}
        path_name_modifier = "x0_pred"  # @param ["x0_pred","x"]
        file = datetime.now().strftime("%Y%m%d-%H%M%S")


        skip_video_for_run_all = True if anim_args.max_frames < 2 else False

        if skip_video_for_run_all == True:
            print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
        else:
            import os
            import subprocess
            from base64 import b64encode

            print(f"{image_path} -> {mp4_path}")

            if use_manual_settings:
                max_frames = "200"  # @param {type:"string"}
            else:
                if render_steps:  # render steps from a single image
                    fname = f"{path_name_modifier}_%05d.png"
                    all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if
                                     os.path.isdir(os.path.join(args.outdir, d))]
                    newest_dir = max(all_step_dirs, key=os.path.getmtime)
                    image_path = os.path.join(newest_dir, fname)
                    print(f"Reading images from {image_path}")
                    print(args.timestring)
                    mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                    max_frames = str(args.steps)
                else:  # render images for a video
                    image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.png")
                    mp4_path = os.path.join(mp4_path, f"{file}.mp4")
                    max_frames = str(anim_args.max_frames)

            # make video
            cmd = [
                gs.system.ffmpegPath,
                '-y',
                '-vcodec', 'png',
                '-r', str(fps),
                '-start_number', str(0),
                '-i', image_path,
                '-frames:v', max_frames,
                '-c:v', 'libx264',
                '-vf',
                f'fps={fps}',
                '-pix_fmt', 'yuv420p',
                '-crf', '17',
                '-preset', 'veryfast',
                '-pattern_type', 'sequence',
                mp4_path
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                print(stderr)
                raise RuntimeError(stderr)

            mp4 = open(mp4_path, 'rb').read()
            data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
            # display.display(display.HTML(f'<video controls loop><source src="{data_url}" type="video/mp4"></video>'))
        hijack_deforum.undo_hijack()
        del args
        del anim_args
        del root

        if lowmem == True:
            print('Low Memory Mode enabled')
            gs.models["model"].to("cpu")
            gs.models["modelCS"].to("cpu")
            gs.models["modelFS"].to("cpu")
            gs.models["sd"] = None
        else:
            #gs.models["sd"].cond_stage_model.to("cpu")
            #gs.models["sd"].to("cpu")
            gs.models["model"] = None
            gs.models["modelCS"] = None
            gs.models["modelFS"] = None
            del gs.models["model"]
            del gs.models["modelCS"]
            del gs.models["modelFS"]

        torch_gc()
        return paths # this gets images via colab api

    def render_animation_new(self):

        if 'inpaint' in gs.models:
            del gs.models['inpaint']
            torch_gc()
            #self.root.model = gs.models['sd']
        elif 'sd' not in gs.models:
            self.load_model()

        self.root.device = self.device
        self.root.models_path = 'models'
        self.root.output_path = self.args.output
        self.root.half_precision = True

        render_animation(self.args, self.anim_args, self.args.prompts, self.root)
        torch_gc()
        return


    def outpaint_txt2img(self,
                         init_image,
                         prompt="fantasy landscape",
                         seed=-1,
                         steps=10,
                         W=512,
                         H=512,
                         outdir='output/outpaint',
                         n_samples=1,
                         n_rows=1,
                         ddim_eta=0.0,
                         blend_mask=None,
                         mask_blur=10,
                         recons_blur=10,
                         strength=0.95,
                         n_iter=1,
                         scale=7,
                         skip_save=False,
                         skip_grid=True,
                         file_prefix="outpaint",
                         image_callback=None,
                         step_callback=None,
                         with_inpaint=False,
                         ):
        print("Using 1.5 InPaint model") if with_inpaint else None
        mask_img = Image.open("outpaint_mask.png")
        img = Image.open(init_image)

        # mask_img = img.split()[-1]

        width = img.size[0]
        height = img.size[1]
        for i in range(0, width):  # process all pixels
            for j in range(0, height):
                data = mask_img.getpixel((i, j))
                # print(data[3])
                # print(data)
                # data[0] = Red,  [1] = Green, [2] = Blue
                # data[0,1,2] range = 0~255
                if data[3] < 1:  # and data[1] < 1 and data[2] < 1:
                    # put black
                    mask_img.putpixel((i, j), (255, 255, 255))
                else:
                    # Put white
                    mask_img.putpixel((i, j), (0, 0, 0))
        # mask_img = mask_img.convert('L')
        if mask_blur > 0 and with_inpaint == True:
            mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))

        # mask_img = mask_img.filter(ImageFilter.GaussianBlur(mask_blur))
        # mask_img = mask_img.convert('L')
        os.makedirs('output/temp', exist_ok=True)
        mask_img.save('output/temp/mask.png')
        blend_mask = 'output/temp/mask.png'
        os.makedirs(outdir, exist_ok=True)
        outpath = outdir
        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        base_count = len(os.listdir(sample_path))
        base_name = f"{random.randint(10000000, 99999999)}_{seed}_"

        print(F"WITH INPAINT : {with_inpaint}")

        if with_inpaint == False:
            self.load_model_from_config()
            print(f"txt2img seed: {seed}   steps: {steps}  prompt: {prompt}")
            print(f"size:  {W}x{H}")

            torch_gc()

            # seeds = torch.randint(-2 ** 63, 2 ** 63 - 1, [accelerator.num_processes])
            # torch.manual_seed(seeds[accelerator.process_index].item())

            # sampler = choose_sampler(opt)

            # model_wrap = K.external.CompVisDenoiser(model)
            # sigma_min, sigma_max = model_wrap.sigmas[0].item(), model_wrap.sigmas[-1].item()

            batch_size = n_samples
            n_rows = n_rows if n_rows > 0 else batch_size

            prompts_data = get_prompts_data(prompt, n_samples)

            grid_path = ''

            image_guide = init_image
            latent_guide = None

            t_start = None
            masked_image_for_blend = None
            mask_for_reconstruction = None
            latent_mask_for_blend = None
            C = 4
            f = 8
            device = self.device
            # this explains the [1, 4, 64, 64]
            shape = (batch_size, C, H // f, W // f)

            sampler = DDIMSampler_simple()

            sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

            # if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            #    alpha = img.convert('RGBA').split()[-1]
            #    alpha.save('mask.png')

            # img = Image.open('mask.png')

            # if image_guide:
            #    image_guide = image_path_to_torch(image_guide, device)  # [1, 3, 512, 512]
            #    latent_guide = torch_image_to_latent(gs.models["sd"], image_guide, n_samples=n_samples)  # [1, 4, 64, 64]
            #   print(f'image_guide')
            #

            # if blend_mask:
            #    [mask_for_reconstruction, latent_mask_for_blend] = get_mask_for_latent_blending(
            #        device, blend_mask, blur=mask_blur)  # [512, 512]  [1, 4, 64, 64]
            #    masked_image_for_blend = (1 - mask_for_reconstruction) * image_guide[0]  # [3, 512, 512]
            #    # masked_image_for_blend = mask_for_reconstruction * image_guide[0]  # [3, 512, 512]
            #    print(f'blend mask')

            image_guide = image_path_to_torch(image_guide, device)
            latent_guide = torch_image_to_latent(gs.models["sd"], image_guide, n_samples=n_samples)
            [mask_for_reconstruction, latent_mask_for_blend] = get_mask_for_latent_blending(device, blend_mask,
                                                                                            blur=mask_blur,
                                                                                            recons_blur=recons_blur)
            masked_image_for_blend = (1 - mask_for_reconstruction) * image_guide[0]
            latent_guide = latent_guide.to("cuda")
            latent_mask_for_blend = latent_mask_for_blend.to("cuda")
            # print(type(mask_for_reconstruction))
            # print(type(latent_mask_for_blend))
            # print(type(masked_image_for_blend))

            """latent_guide, latent_mask_for_blend = load_img(init_image,
                                                           shape=(W, H),
                                                           use_alpha_as_mask=True)
    
    
            latent_guide = latent_guide.to("cuda")
            with autocast("cuda"):
                latent_guide = gs.models["sd"].get_first_stage_encoding(
                    gs.models["sd"].encode_first_stage(latent_guide))  # move to latent space
            latent_mask_for_blend = prepare_mask(latent_mask_for_blend,
                                                 latent_guide.shape,
                                                 1.0,
                                                 1.0)
    
    
            latent_mask_for_blend = latent_mask_for_blend.to("cuda")"""

            if image_guide is not None:
                assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
                t_start = int(strength * steps)

                print(f"target t_start is {t_start} steps")

            multiple_mode = (n_iter * len(prompts_data) * n_samples > 1)

            with torch.no_grad(), gs.models["sd"].ema_scope(), torch.cuda.amp.autocast():
                tic = time.time()
                all_samples = list()
                counter = 0
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(prompts_data, desc="data"):

                        seed = seed + counter
                        seed_everything(seed)

                        unconditional_conditioning, conditioning = get_conditionings(gs.models["sd"], prompts, n_samples)

                        samples = sampler.ddim_sampling(
                            conditioning,  # [1, 77, 768]
                            shape,  # (1, 4, 64, 64)
                            x0=latent_guide,  # [1, 4, 64, 64]
                            mask=latent_mask_for_blend,  # [1, 4, 64, 64]
                            # 12 (if 20 steps and strength 0.75 => 15)
                            t_start=t_start,
                            unconditional_guidance_scale=scale,
                            # [1, 77, 768]
                            unconditional_conditioning=unconditional_conditioning,
                            step_callback=step_callback,
                        )  # [1, 4, 64, 64]

                        x_samples = encoded_to_torch_image(
                            gs.models["sd"], samples)  # [1, 3, 512, 512]

                        if masked_image_for_blend is not None:
                            x_samples = mask_for_reconstruction * x_samples + masked_image_for_blend

                        all_samples.append(x_samples)

                        generated_time = time.time()

                        if (not skip_save) and (not multiple_mode):
                            for x_sample in x_samples:
                                image = sampleToImage(x_sample)
                                fpath = os.path.join(sample_path, f"{base_name}_{base_count:05}.png")
                                image.save(fpath)
                                self.temppath = fpath
                                if image_callback is not None:
                                    image_callback(image)
                                # save_image(
                                #    image,
                                #    os.path.join(sample_path, f"{base_count:05}.png"),
                                #        pnginfo=metadata(
                                #        prompt=prompts[0],  # FIXME [0]
                                #        seed=seed,
                                #        generation_time=generated_time - tic
                                #            ))

                                base_count += 1






        elif with_inpaint == True:
            torch_gc()
            if "inpaint" not in gs.models:
                self.load_inpaint_model()
            sampler = DDIMSampler(gs.models["inpaint"])
            image_guide = image_path_to_torch(init_image, self.device)
            [mask_for_reconstruction, latent_mask_for_blend] = get_mask_for_latent_blending(self.device, blend_mask,
                                                                                            blur=mask_blur,
                                                                                            recons_blur=recons_blur)
            masked_image_for_blend = (1 - mask_for_reconstruction) * image_guide[0]

            mask = mask_img
            image = Image.open(init_image)
            result = inpaint(
                # model=self.model,
                sampler=sampler,
                image=image,
                mask=mask,
                prompt=prompt,
                seed=seed,
                scale=scale,
                ddim_steps=steps,
                num_samples=1,
                h=height, w=width,
                device=self.device,
                mask_for_reconstruction=mask_for_reconstruction,
                masked_image_for_blend=masked_image_for_blend,
                callback=step_callback)
            fpath = os.path.join(sample_path, f"{base_name}_{base_count:05}.png")
            result[0].save(fpath, 'PNG')
            self.temppath = fpath
            image_callback(result[0])

        # global plms_sampler
        # global ddim_sampler
        # plms_sampler = PLMSSampler(gs.models["sd"])
        # ddim_sampler = DDIMSampler(gs.models["sd"])

        if not skip_grid:

            generated_time = time.time()
            multiple_mode = False
            if multiple_mode:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                grid = make_grid(grid, nrow=n_rows)

                # to image
                # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                # image = Image.fromarray(grid.astype(np.uint8))
            else:
                grid = all_samples[0][0]

            image = sampleToImage(grid)
            grid_path = os.path.join(outpath, f'{file_prefix}-0000.png')

            print(image)

            save_image(image,
                       grid_path,
                       pnginfo=metadata(prompt=prompts[0],  # FIXME [0]
                                        seed=seed,
                                        generation_time=generated_time - tic
                                        ))


        torch_gc()
def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, device, mask_for_reconstruction, masked_image_for_blend, num_samples=1, w=512, h=512, callback=None):
    #model = sampler.model
    gs.models["inpaint"].to(device)

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float16)

    #gs.models["inpaint"].model.to("cpu")
    gs.models["inpaint"].cond_stage_model.to(device)
    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = gs.models["inpaint"].cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in gs.models["inpaint"].concat_keys:
                cc = batch[ck].float()
                if ck != gs.models["inpaint"].masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = gs.models["inpaint"].get_first_stage_encoding(gs.models["inpaint"].encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = gs.models["inpaint"].get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            gs.models["inpaint"].cond_stage_model.to("cpu")
            #gs.models["inpaint"].model.to(device)
            shape = [gs.models["inpaint"].channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=1.0,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc_full,
                x_T=start_code,
                img_callback=callback,
            )
            x_samples = encoded_to_torch_image(
                gs.models["inpaint"], samples_cfg)  # [1, 3, 512, 512]
            all_samples = []
            if masked_image_for_blend is not None:
                x_samples = mask_for_reconstruction * x_samples + masked_image_for_blend

            all_samples.append(x_samples)

            generated_time = time.time()

            for x_sample in x_samples:
                image = sampleToImage(x_sample)
                result = [image]


                #image.save(os.path.join(sample_path, f"{base_count:05}.png"))
                #if image_callback is not None:
                #    image_callback(image)




            #result = torch.clamp((x_samples+1.0)/2.0,
            #                     min=0.0, max=1.0)

            #result = result.cpu().numpy().transpose(0,2,3,1)
            #result = result*255

    #result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    # result = [put_watermark(img for img in result]
    gs.models["inpaint"].to("cpu")
    return result


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch

def load_vae_dict(model, vae_dict_1=None):
    if vae_dict_1:
        store_base_vae(model)
        model.first_stage_model.load_state_dict(vae_dict_1)
    else:
        restore_base_vae()
    model.first_stage_model.to(choose_torch_device())

def get_base_vae(model):
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info and model:
        return base_vae
    return None


def store_base_vae(model):
    global base_vae #, checkpoint_info
    #if checkpoint_info != model.sd_checkpoint_info:
    base_vae = model.first_stage_model.state_dict().copy()
        #checkpoint_info = model.sd_checkpoint_info


def delete_base_vae():
    global base_vae, checkpoint_info
    base_vae = None
    checkpoint_info = None


def restore_base_vae(model):
    global base_vae, checkpoint_info
    if base_vae is not None and checkpoint_info == model.sd_checkpoint_info:
        load_vae_dict(model, base_vae)
    delete_base_vae()
