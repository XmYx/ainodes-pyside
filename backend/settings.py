import json
from types import SimpleNamespace
from backend.singleton import singleton

gs = singleton


def load_settings_json():
    f = open("configs/ainodes/webui_settings.json", "r")
    settings = json.loads(f.read())

    settings = SimpleNamespace(**settings)
    gs.diffusion = SimpleNamespace(**settings.diffusion)
    gs.system = SimpleNamespace(**settings.system)
