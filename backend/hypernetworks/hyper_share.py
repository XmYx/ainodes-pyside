import collections
import csv
import glob
import os
import sys
from collections import namedtuple

from omegaconf import OmegaConf



from backend.torch_gc import torch_gc
from backend.singleton import singleton
gs = singleton

import torch
from torch import nn
from torch.nn.functional import silu
from backend.hypernetworks.modules import sd_hijack_optimizations, sd_hijack
from backend.hypernetworks.modules import devices


#from backend.hypernetworks import hyper_jack
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import ldm.models.diffusion.ddpm
from ldm.util import instantiate_from_config

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward
util_instantiate_from_config = ldm.util.instantiate_from_config
ddpm_LatentDiffusion_init = ldm.models.diffusion.ddpm.LatentDiffusion.__init__

def undo_optimizations():

    ldm.modules.attention.CrossAttention.forward = attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward
    ldm.util.instantiate_from_config = instantiate_from_config
    ldm.models.diffusion.ddpm.LatentDiffusion.__init__ = ddpm_LatentDiffusion_init

def add_training_optimizations():
    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.attention.CrossAttention.forward = sd_hijack_optimizations.split_cross_attention_forward
    ldm.modules.diffusionmodules.model.AttnBlock.forward = sd_hijack_optimizations.cross_attention_attnblock_forward
 #   ldm.util.instantiate_from_config = hyper_jack.instantiate_from_config
 #   ldm.models.diffusion.ddpm.LatentDiffusion.__init__ = hyper_jack.LatentDiffusion_init__

def list_hypernetworks(path):
    res = {}
    for filename in glob.iglob(os.path.join(path, '**/*.pt'), recursive=True):
        name = os.path.splitext(os.path.basename(filename))[0]
        res[name] = filename
        print(filename)
    return res



data= {
    "samples_save": True,
    "samples_format": "png",
    "samples_filename_pattern": "",
    "save_images_add_number": True,
    "grid_save": True,
    "grid_format": "png",
    "grid_extended_filename": False,
    "grid_only_if_multiple": True,
    "grid_prevent_empty_spots": False,
    "n_rows": -1,
    "enable_pnginfo": True,
    "save_txt": False,
    "save_images_before_face_restoration": False,
    "jpeg_quality": 80,
    "export_for_4chan": True,
    "use_original_name_batch": False,
    "save_selected_only": True,
    "do_not_add_watermark": False,
    "outdir_samples": "",
    "outdir_txt2img_samples": "outputs/txt2img-images",
    "outdir_img2img_samples": "outputs/img2img-images",
    "outdir_extras_samples": "outputs/extras-images",
    "outdir_grids": "",
    "outdir_txt2img_grids": "outputs/txt2img-grids",
    "outdir_img2img_grids": "outputs/img2img-grids",
    "outdir_save": "log/images",
    "save_to_dirs": False,
    "grid_save_to_dirs": False,
    "use_save_to_dirs_for_ui": False,
    "directories_filename_pattern": "",
    "directories_max_prompt_words": 8,
    "ESRGAN_tile": 192,
    "ESRGAN_tile_overlap": 8,
    "realesrgan_enabled_models": ["R-ESRGAN x4+", "R-ESRGAN x4+ Anime6B"],
    "SWIN_tile": 192,
    "SWIN_tile_overlap": 8,
    "ldsr_steps": 100,
    "upscaler_for_img2img": None,
    "use_scale_latent_for_hires_fix": False,
    "face_restoration_model": None,
    "code_former_weight": 0.5,
    "face_restoration_unload": False,
    "memmon_poll_rate": 8,
    "samples_log_stdout": False,
    "multiple_tqdm": True,
    "unload_models_when_training": False,
    "dataset_filename_word_regex": "",
    "dataset_filename_join_string": " ",
    "training_image_repeats_per_epoch": 1,
    "training_write_csv_every": 500,
    "sd_model_file": "model.ckpt [a9263745]",
    "sd_checkpoint_cache": 0,
    "sd_hypernetwork": "None",
    "sd_hypernetwork_strength": 1.0,
    "inpainting_mask_weight": 1.0,
    "img2img_color_correction": False,
    "save_images_before_color_correction": False,
    "img2img_fix_steps": False,
    "enable_quantization": False,
    "enable_emphasis": True,
    "use_old_emphasis_implementation": False,
    "enable_batch_seeds": True,
    "comma_padding_backtrack": 20,
    "filter_nsfw": False,
    "CLIP_stop_at_last_layers": 1,
    "random_artist_categories": [],
    "interrogate_keep_models_in_memory": False,
    "interrogate_use_builtin_artists": True,
    "interrogate_return_ranks": False,
    "interrogate_clip_num_beams": 1,
    "interrogate_clip_min_length": 24,
    "interrogate_clip_max_length": 48,
    "interrogate_clip_dict_limit": 1500,
    "interrogate_deepbooru_score_threshold": 0.5,
    "deepbooru_sort_alpha": True,
    "deepbooru_use_spaces": False,
    "deepbooru_escape": True,
    "show_progressbar": True,
    "show_progress_every_n_steps": 0,
    "show_progress_grid": True,
    "return_grid": True,
    "do_not_show_images": False,
    "add_model_hash_to_info": True,
    "add_model_name_to_info": False,
    "disable_weights_auto_swap": False,
    "send_seed": True,
    "font": "",
    "js_modal_lightbox": True,
    "js_modal_lightbox_initially_zoomed": True,
    "show_progress_in_title": True,
    "quicksettings": "sd_model_file",
    "localization": "None",
    "hide_samplers": [],
    "eta_ddim": 0.0,
    "eta_ancestral": 1.0,
    "ddim_discretize": "uniform",
    "s_churn": 0.0,
    "s_tmin": 0.0,
    "s_noise": 1.0,
    "eta_noise_seed_delta": 0
}
model_path = 'data/models'
ckpt_dir = 'data/models'
ckpt = 'v1-5-pruned-emaonly.ckpt'
default_sd_model_file = ckpt
CheckpointInfo = namedtuple("CheckpointInfo", ['filename', 'title', 'hash', 'model_name', 'config'])
checkpoints_list = {}
config = 'configs/stable-diffusion/v1-inference-h.yaml'

def model_hash(filename):
    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'

def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    if ext_filter is None:
        ext_filter = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            if os.path.exists(place):
                for file in glob.iglob(place + '**/**', recursive=True):
                    full_path = file
                    if os.path.isdir(full_path):
                        continue
                    if len(ext_filter) != 0:
                        model_name, extension = os.path.splitext(file)
                        if extension not in ext_filter:
                            continue
                    if file not in output:
                        output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                dl = 'downloaded' #load_file_from_url(model_url, model_path, True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception:
        pass

    return output


def list_models():
    # hardcoded for now
    config = 'configs/stable-diffusion/v1-inference-h.yaml'


    checkpoints_list.clear()
    model_list = load_models(model_path=model_path, command_path=ckpt_dir, ext_filter=[".ckpt"])
    print('model_list ', model_list)
    def modeltitle(path, shorthash):
        abspath = os.path.abspath(path)

        if ckpt_dir is not None and abspath.startswith(ckpt_dir):
            name = abspath.replace(ckpt_dir, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(path)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        shortname = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]

        return f'{name} [{shorthash}]', shortname

    cmd_ckpt = os.path.join('model', ckpt)
    print('cmd_ckpt', cmd_ckpt)
    if os.path.exists(cmd_ckpt):
        h = model_hash(cmd_ckpt)
        title, short_model_name = modeltitle(cmd_ckpt, h)

        checkpoints_list[title] = CheckpointInfo(cmd_ckpt, title, h, short_model_name, config)
        data['sd_model_file'] = title
    elif cmd_ckpt is not None and cmd_ckpt != default_sd_model_file:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)
    for filename in model_list:
        h = model_hash(filename)
        title, short_model_name = modeltitle(filename, h)

        basename, _ = os.path.splitext(filename)
        config = basename + ".yaml"
        print('basename', basename)
        print('config', config)
        if not os.path.exists(config):
            print('not exist config', config)
            config = 'configs/stable-diffusion/v1-inference-h.yaml'

        checkpoints_list[title] = CheckpointInfo(filename, title, h, short_model_name, config)



def validate_train_inputs(model_name, learn_rate, batch_size, data_root, template_file, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_file, "Prompt template file is empty"
    assert os.path.isfile(template_file), "Prompt template file doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0 , "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0 , "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0 , "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"

def make_linear_decode(model_version, device='cuda:0'):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [ 0.298,  0.207,  0.208],  # L1
        [ 0.187,  0.286,  0.173],  # L2
        [-0.158,  0.189,  0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]

    if model_version[:5] == "sd-v1":
        rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
    else:
        raise Exception(f"Model name {model_version} not recognized.")

    def linear_decode(latent):
        latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
        latent_image = latent_image.permute(0, 3, 1, 2)
        return latent_image

    return linear_decode

def load_model_from_config( config, ckpt, verbose=False):

    config = OmegaConf.load(config)

    if "sd" not in gs.models:
        print(f"Loading model from {ckpt}")
        #if "sd" not in gs.models:
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
        #model.half()

        #model.embedding_manager.load(opt.embedding_path)
        gs.models["sd"] = model.half().to("cuda")

        for m in gs.models["sd"].modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m._orig_padding_mode = m.padding_mode

        autoencoder_version = "sd-v1" #TODO this will be different for different models
        gs.models["sd"].linear_decode = make_linear_decode(autoencoder_version, 'cuda')

        del pl_sd
        del sd
        del m, u
        del model
        torch_gc()
        #gs.models["sd"].eval()
        #return model

def select_checkpoint():
    list_models()


    model_checkpoint = ckpt + ' [a9263745]'
    checkpoint_info = checkpoints_list.get(model_checkpoint, None)
    print('model_checkpoint ', model_checkpoint)
    print('checkpoint_info ', checkpoint_info)


    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print(f"No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if ckpt is not None:
            print(f" - file {os.path.abspath(ckpt)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if ckpt_dir is not None:
            print(f" - directory {os.path.abspath(ckpt_dir)}", file=sys.stderr)
        print(f"Can't run without a checkpoint. Find and place a .ckpt file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info

def write_loss(log_directory, filename, step, epoch_len, values):
    if gs.training_write_csv_every == 0:
        return

    if (step + 1) % gs.training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = step // epoch_len
        epoch_step = step % epoch_len

        csv_writer.writerow({
            "step": step + 1,
            "epoch": epoch,
            "epoch_step": epoch_step + 1,
            **values,
        })

checkpoints_loaded = collections.OrderedDict()
chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}
def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def get_state_dict_from_checkpoint(pl_sd):
    if "state_dict" in pl_sd:
        pl_sd = pl_sd["state_dict"]

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

vae_ignore_keys = {"model_ema.decay", "model_ema.num_updates"}


def load_model_weights(model, checkpoint_info):
    print('checkpoint_info ', checkpoint_info)
    checkpoint_file = checkpoint_info.filename
    sd_model_hash = checkpoint_info.hash

    if checkpoint_info not in checkpoints_loaded:
        print(f"Loading weights [{sd_model_hash}] from {checkpoint_file}")

        pl_sd = torch.load(checkpoint_file, map_location='cuda')
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")

        sd = get_state_dict_from_checkpoint(pl_sd)
        del pl_sd
        model.load_state_dict(sd, strict=False)
        del sd

        opt_channelslast = False #debug
        if opt_channelslast:
            model.to(memory_format=torch.channels_last)
        no_half = False #debug
        if not no_half:
            model.half()

        no_half_vae = False # debug

        devices.dtype = torch.float32 if no_half else torch.float16
        devices.dtype_vae = torch.float32 if no_half or no_half_vae else torch.float16

        devices.dtype = torch.float16
        devices.dtype_vae = torch.float16

        vae_file = os.path.splitext(checkpoint_file)[0] + ".vae.pt"
        vae_dir = None
        if not os.path.exists(vae_file) and vae_dir is not None:
            vae_file = vae_dir
        vae_file = 'model.vae.pt'
        if os.path.exists(vae_file):
            print(f"Loading VAE weights from: {vae_file}")
            vae_ckpt = torch.load(vae_file, map_location='cpu')
            vae_dict = {k: v for k, v in vae_ckpt["state_dict"].items() if k[0:4] != "loss" and k not in vae_ignore_keys}
            model.first_stage_model.load_state_dict(vae_dict)

        model.first_stage_model.to(devices.dtype_vae)

        sd_checkpoint_cache = 0

        if sd_checkpoint_cache > 0:
            checkpoints_loaded[checkpoint_info] = model.state_dict().copy()
            while len(checkpoints_loaded) > sd_checkpoint_cache:
                checkpoints_loaded.popitem(last=False)  # LRU
    else:
        print(f"Loading weights [{sd_model_hash}] from cache")
        checkpoints_loaded.move_to_end(checkpoint_info)
        model.load_state_dict(checkpoints_loaded[checkpoint_info])

    model.sd_model_hash = sd_model_hash
    model.sd_model_file = checkpoint_file
    model.sd_checkpoint_info = checkpoint_info
