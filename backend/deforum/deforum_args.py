import time
from types import SimpleNamespace

def DeforumAnimArgs(attr):
    attr = SimpleNamespace(**attr)
    #@markdown ####**Animation:**
    animation_mode = attr.animation_mode #@param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
    max_frames = attr.max_frames #@param {type:"number"}
    border = attr.border #@param ['wrap', 'replicate'] {type:'string'}

    #@markdown ####**Motion Parameters:**
    angle = attr.angle#@param {type:"string"}
    zoom = attr.zoom #@param {type:"string"}
    translation_x = attr.translation_x  #@param {type:"string"}
    translation_y = attr.translation_y #@param {type:"string"}
    translation_z = attr.translation_z #@param {type:"string"}
    rotation_3d_x = attr.rotation_3d_x #@param {type:"string"}
    rotation_3d_y = attr.rotation_3d_y #@param {type:"string"}
    rotation_3d_z = attr.rotation_3d_z #@param {type:"string"}
    flip_2d_perspective = attr.flip_2d_perspective #@param {type:"boolean"}
    perspective_flip_theta = attr.perspective_flip_theta#@param {type:"string"}
    perspective_flip_phi = attr.perspective_flip_phi#@param {type:"string"}
    perspective_flip_gamma = attr.perspective_flip_gamma#@param {type:"string"}
    perspective_flip_fv = attr.perspective_flip_fv#@param {type:"string"}
    noise_schedule = attr.noise_schedule#@param {type:"string"}
    strength_schedule = attr.strength_schedule#@param {type:"string"}
    contrast_schedule = attr.contrast_schedule#@param {type:"string"}

    #@markdown ####**Coherence:**
    color_coherence = attr.color_coherence #@param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
    diffusion_cadence = attr.diffusion_cadence #@param ['1','2','3','4','5','6','7','8'] {type:'string'}

    #@markdown ####**3D Depth Warping:**
    use_depth_warping = attr.use_depth_warping #@param {type:"boolean"}
    midas_weight =attr.midas_weight#@param {type:"number"}
    near_plane = attr.near_plane
    far_plane = attr.far_plane
    fov = attr.fov#@param {type:"number"}
    padding_mode = attr.padding_mode#@param ['border', 'reflection', 'zeros'] {type:'string'}
    sampling_mode = attr.sampling_mode#@param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
    save_depth_maps = attr.save_depth_maps #@param {type:"boolean"}

    #@markdown ####**Video Input:**
    video_init_path =attr.video_init_path#@param {type:"string"}
    extract_nth_frame = attr.extract_nth_frame#@param {type:"number"}
    overwrite_extracted_frames = attr.overwrite_extracted_frames #@param {type:"boolean"}
    use_mask_video = attr.use_mask_video #@param {type:"boolean"}
    video_mask_path =attr.video_mask_path #@param {type:"string"}

    #@markdown ####**Interpolation:**
    interpolate_key_frames = attr.interpolate_key_frames #@param {type:"boolean"}
    interpolate_x_frames = attr.interpolate_x_frames #@param {type:"number"}

    #@markdown ####**Resume Animation:**
    resume_from_timestring = attr.resume_from_timestring #@param {type:"boolean"}
    resume_timestring = attr.resume_timestring #@param {type:"string"}
    del attr
    return locals()


def DeforumArgs(attr):

    attr = SimpleNamespace(**attr)

    #@markdown **Image Settings**
    W = attr.W #@param
    H = attr.H #@param
    #W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64

    #@markdown **Sampling Settings**
    seed = attr.seed #@param
    sampler = attr.sampler #@param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
    steps = attr.steps #@param
    scale = attr.scale #@param
    ddim_eta = attr.ddim_eta #@param
    dynamic_threshold = attr.dynamic_threshold
    static_threshold = attr.static_threshold

    #@markdown **Save & Display Settings**
    save_samples = attr.save_samples #@param {type:"boolean"}
    save_settings = attr.save_settings #@param {type:"boolean"}
    display_samples = attr.display_samples #@param {type:"boolean"}
    save_sample_per_step = attr.save_sample_per_step #@param {type:"boolean"}
    show_sample_per_step = attr.show_sample_per_step #@param {type:"boolean"}

    #@markdown **Prompt Settings**
    prompt_weighting = attr.prompt_weighting #@param {type:"boolean"}
    normalize_prompt_weights = attr.normalize_prompt_weights #@param {type:"boolean"}
    log_weighted_subprompts = attr.log_weighted_subprompts #@param {type:"boolean"}

    #@markdown **Batch Settings**
    n_batch = int(attr.n_batch) #@param
    batch_name = attr.batch_name #@param {type:"string"}
    filename_format = attr.filename_format #@param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
    seed_behavior = attr.seed_behavior #@param ["iter","fixed","random"]
    make_grid = attr.make_grid #@param {type:"boolean"}
    grid_rows = attr.grid_rows #@param
    outdir = attr.outdir

    #@markdown **Init Settings**
    use_init = attr.use_init #@param {type:"boolean"}
    strength = attr.strength #@param {type:"number"}
    strength_0_no_init = attr.strength_0_no_init # Set the strength to 0 automatically when no init image is used
    init_image = attr.init_image
    # Whiter areas of the mask are areas that change more
    use_mask = attr.use_mask #@param {type:"boolean"}
    use_alpha_as_mask = attr.use_alpha_as_mask # use the alpha channel of the init image as the mask{type:"boolean"}
    mask_file = attr.mask_file #@param {type:"string"}
    invert_mask = attr.invert_mask #@param {type:"boolean"}
    # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
    mask_brightness_adjust = attr.mask_brightness_adjust  #@param {type:"number"}
    mask_contrast_adjust = attr.mask_contrast_adjust  #@param {type:"number"}
    # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
    overlay_mask = attr.overlay_mask  # {type:"boolean"}
    # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
    mask_overlay_blur = attr.mask_overlay_blur # {type:"number"}

    #@markdown **Exposure/Contrast Conditional Settings**
    mean_scale = attr.mean_scale #@param {type:"number"}
    var_scale = attr.var_scale #@param {type:"number"}
    exposure_scale = attr.exposure_scale #@param {type:"number"}
    exposure_target = attr.exposure_target #@param {type:"number"}

    #@markdown **Color Match Conditional Settings**
    colormatch_scale = attr.colormatch_scale #@param {type:"number"}
    colormatch_image = attr.colormatch_image #@param {type:"string"}
    colormatch_n_colors = attr.colormatch_n_colors #@param {type:"number"}
    ignore_sat_weight = attr.ignore_sat_weight #@param {type:"number"}

    #@markdown **CLIP\Aesthetics Conditional Settings**
    clip_name = attr.clip_name #@param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
    clip_scale = attr.clip_scale #@param {type:"number"}
    aesthetics_scale = attr.aesthetics_scale #@param {type:"number"}
    cutn = attr.cutn #@param {type:"number"}
    cut_pow = attr.cut_pow #@param {type:"number"}

    #@markdown **Other Conditional Settings**
    init_mse_scale = attr.init_mse_scale #@param {type:"number"}
    init_mse_image = attr.init_mse_image #@param {type:"string"}

    blue_scale = attr.blue_scale #@param {type:"number"}

    #@markdown **Conditional Gradient Settings**
    gradient_wrt = attr.gradient_wrt #@param ["x", "x0_pred"]
    gradient_add_to = attr.gradient_add_to #@param ["cond", "uncond", "both"]
    decode_method = attr.decode_method #@param ["autoencoder","linear"]
    grad_threshold_type = attr.grad_threshold_type #@param ["dynamic", "static", "mean", "schedule"]
    clamp_grad_threshold = attr.clamp_grad_threshold #@param {type:"number"}
    clamp_start = attr.clamp_start #@param {type:"number"}
    clamp_stop = attr.clamp_stop #@param {type:"number"}
    grad_inject_timing = attr.grad_inject_timing #@param {type:"boolean"}

    #@markdown **Speed vs VRAM Settings**
    cond_uncond_sync = attr.cond_uncond_sync #@param {type:"boolean"}

    n_samples = int(attr.n_samples) # doesnt do anything
    precision = attr.precision
    C = 4
    f = 8

    prompt = attr.prompt
    timestring = attr.timestring
    init_latent = attr.init_latent
    init_sample = attr.init_sample
    init_c = attr.init_c
    outdir = attr.outdir

    negative_prompts = attr.negative_prompts
    hires = attr.hires
    lowmem = attr.lowmem
    seamless = False
    axis = {'x'}
    gradient_pass = 'Second'
    return_type = 'latent'

    del attr
    return locals()


def Root():
    return locals()

def get_model_map():
    return {
        "v1-5-pruned.ckpt": {
            'sha256': 'e1441589a6f3c5a53f5f54d0975a18a7feb7cdf0b0dee276dfc3331ae376a053',
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.ckpt',
            'requires_login': True,
        },
        "v1-5-pruned-emaonly.ckpt": {
            'sha256': 'cc6cb27103417325ff94f52b7a5d2dde45a7515b25c255d8e396c90014281516',
            'url': 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt',
            'requires_login': True,
        },
        "sd-v1-4-full-ema.ckpt": {
            'sha256': '14749efc0ae8ef0329391ad4436feb781b402f4fece4883c7ad8d10556d8a36a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-4-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-4.ckpt": {
            'sha256': 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt',
            'requires_login': True,
        },
        "sd-v1-3-full-ema.ckpt": {
            'sha256': '54632c6e8a36eecae65e36cb0595fab314e1a1545a65209f24fde221a8d4b2ca',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/blob/main/sd-v1-3-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-3.ckpt": {
            'sha256': '2cff93af4dcc07c3e03110205988ff98481e86539c51a8098d4f2236e41f7f2f',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-3-original/resolve/main/sd-v1-3.ckpt',
            'requires_login': True,
        },
        "sd-v1-2-full-ema.ckpt": {
            'sha256': 'bc5086a904d7b9d13d2a7bccf38f089824755be7261c7399d92e555e1e9ac69a',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/blob/main/sd-v1-2-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-2.ckpt": {
            'sha256': '3b87d30facd5bafca1cbed71cfb86648aad75d1c264663c0cc78c7aea8daec0d',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-2-original/resolve/main/sd-v1-2.ckpt',
            'requires_login': True,
        },
        "sd-v1-1-full-ema.ckpt": {
            'sha256': 'efdeb5dc418a025d9a8cc0a8617e106c69044bc2925abecc8a254b2910d69829',
            'url':'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1-full-ema.ckpt',
            'requires_login': True,
        },
        "sd-v1-1.ckpt": {
            'sha256': '86cd1d3ccb044d7ba8db743d717c9bac603c4043508ad2571383f954390f3cea',
            'url': 'https://huggingface.co/CompVis/stable-diffusion-v-1-1-original/resolve/main/sd-v1-1.ckpt',
            'requires_login': True,
        },
        "robo-diffusion-v1.ckpt": {
            'sha256': '244dbe0dcb55c761bde9c2ac0e9b46cc9705ebfe5f1f3a7cc46251573ea14e16',
            'url': 'https://huggingface.co/nousr/robo-diffusion/resolve/main/models/robo-diffusion-v1.ckpt',
            'requires_login': False,
        },
        "wd-v1-3-float16.ckpt": {
            'sha256': '4afab9126057859b34d13d6207d90221d0b017b7580469ea70cee37757a29edd',
            'url': 'https://huggingface.co/hakurei/waifu-diffusion-v1-3/resolve/main/wd-v1-3-float16.ckpt',
            'requires_login': False,
        },
    }

def prepare_args(attr):
    root = Root()
    args = DeforumArgs(attr)
    anim_args = DeforumAnimArgs(attr)

    root = SimpleNamespace(**root)
    args = SimpleNamespace(**args)
    anim_args = SimpleNamespace(**anim_args)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    args.strength = max(0.0, min(1.0, args.strength))

    return [args, anim_args, root]
