

from PIL import Image
from backend.clip_interrogator import Interrogator, Config

from backend.singleton import singleton

gs = singleton


def get_prompt_guess(image_path):
    image = Image.open(image_path).convert('RGB')
    if 'clip' not in gs.models:
        gs.models['clip'] = Interrogator(Config(clip_model_name="ViT-L/14"))
    guess = gs.models['clip'].interrogate(image)
    return guess

def get_prompt_guess_img(image):
    if 'clip' not in gs.models:
        gs.models['clip'] = Interrogator(Config(clip_model_name="ViT-L/14"))
    guess = gs.models['clip'].interrogate(image)
    return guess
