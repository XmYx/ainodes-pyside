import json
from types import SimpleNamespace
from backend.singleton import singleton

gs = singleton


def load_settings_json():
    f = open("configs/ainodes/settings.json", "r")
    settings = json.loads(f.read())

    settings = SimpleNamespace(**settings)
    gs.diffusion = SimpleNamespace(**settings.diffusion)
    gs.system = SimpleNamespace(**settings.system)

def load_default_settings_json():
    f = open("configs/ainodes/default_settings.json", "r")
    settings = json.loads(f.read())

    settings = SimpleNamespace(**settings)
    gs.diffusion = SimpleNamespace(**settings.diffusion)
    gs.system = SimpleNamespace(**settings.system)


def save_settings_json():
    system = json.dumps(gs.system.__dict__)
    diffusion = json.dumps(gs.diffusion.__dict__)
    settings = json.dumps({
        "system": json.loads(system),
        "diffusion": json.loads(diffusion)
    })
    f = open("configs/ainodes/settings.json", "w")
    f.write(settings)
    f.close()
