import json
import os
from os import path
from types import SimpleNamespace
from backend.singleton import singleton

gs = singleton


def load_settings_json():
    f = open("configs/ainodes/default_settings.json", "r")
    settings = json.loads(f.read())
    settings = SimpleNamespace(**settings)
    gs.diffusion = SimpleNamespace(**settings.diffusion)
    gs.system = SimpleNamespace(**settings.system)

    settingsfile = 'configs/ainodes/settings.json'
    if os.path.exists(settingsfile):
        f = open("configs/ainodes/settings.json", "r")
        user_settings = json.loads(f.read())
        user_settings = SimpleNamespace(**user_settings)
        user_diffusion = SimpleNamespace(**user_settings.diffusion)
        user_system = SimpleNamespace(**user_settings.system)
        for key, value in user_diffusion.__dict__.items():
            if key in gs.diffusion.__dict__:
                gs.diffusion.__dict__[key] = value
        for key, value in user_system.__dict__.items():
            if key in gs.system.__dict__:
                gs.system.__dict__[key] = value
    save_settings_json()

def load_default_settings_json():
    load_settings_json()
    #f = open("configs/ainodes/default_settings.json", "r")
    #settings = json.loads(f.read())
    #settings = SimpleNamespace(**settings)
    #gs.diffusion = SimpleNamespace(**settings.diffusion)
    #gs.system = SimpleNamespace(**settings.system)


def save_settings_json():
    system = json.dumps(gs.system.__dict__)
    diffusion = json.dumps(gs.diffusion.__dict__)
    settings = json.dumps({
        "system": json.loads(system),
        "diffusion": json.loads(diffusion)
    })
    print(json.loads(system))
    with open("configs/ainodes/settings.json", "w") as write_file:
        json.dump({
            "system": json.loads(system),
            "diffusion": json.loads(diffusion)
        }, write_file, indent=4)




def load_windows_settings():
    filename = "configs/ainodes/windows_views.json"
    if path.exists(filename):
        try:
            f = open(filename, "r")
            gs.windows_views = json.loads(f.read())
        except:
            os.remove(filename)
            gs.windows_views = {}
    else:
        gs.windows_views = {}



def save_windows_settings():
    filename = "configs/ainodes/windows_views.json"
    with open(filename, "w") as write_file:
        json.dump(gs.windows_views, write_file, indent=4)
