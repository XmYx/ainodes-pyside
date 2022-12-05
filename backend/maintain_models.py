import os

from backend.poor_mans_wget import wget
from backend.singleton import singleton
gs = singleton

def check_models_exist():
    model_path = {
        '1': {'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth', 'path': gs.system.realesrgan_model_file},
        '2': {'url': 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth', 'path': gs.system.realesrgan_anime_model_file},
        '3': {'url': 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', 'path': gs.system.gfpgan_model_file},
        '4': {'hint': 'https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA', 'path': gs.system.adabins_model_file},
        #'5': {'url': 'https://drive.google.com/file/d/1HMgff-FV6qw1L0ywQZJ7ECa9VPq1bIoj/view?usp=share_link', 'path': gs.system.adabins_kitty_model_file},
        '6': {'url': 'https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt', 'path': gs.system.midas_model_file},
        #'7': {'url': 'https://huggingface.co/lambdalabs/stable-diffusion-image-conditioned/resolve/main/sd-clip-vit-l14-img-embed_ema_only.ckpt', 'path': gs.system.sd_clip_model_file},
        '8': {'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_caption_capfilt_large.pth', 'path': gs.system.blip_model_file},
        '9': {'url': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_caption.pth', 'path': gs.system.blip_large_model_file},
    }

    for key in model_path.keys():
        path = model_path[key]['path']
        if 'url' in model_path[key]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.isfile(model_path[key]['path']):
                url = model_path[key]['url']
                print(f"model {path} gets downloaded from {url}")
                wget(model_path[key]['url'], model_path[key]['path'])
        if 'hint' in model_path[key]:
            if not os.path.isfile(model_path[key]['path']):
                print(f"You may have to download the model supposed to be at {model_path[key]['path']} from here: {model_path[key]['hint']}")
