import json
import os
from os import path
from types import SimpleNamespace
from backend.sqlite import db_base
from backend.sqlite import setting_db
from backend.singleton import singleton

gs = singleton

def push_settings_to_gs(user_settings):
    user_settings = SimpleNamespace(**user_settings)
    user_diffusion = SimpleNamespace(**user_settings.diffusion)
    user_system = SimpleNamespace(**user_settings.system)
    # only use those settings that are present as default settings,
    # we use this as a cleanup filter for the ever growing settings drama we had in the past
    for key, value in user_diffusion.__dict__.items():
        if key in gs.diffusion.__dict__:
            gs.diffusion.__dict__[key] = value
    for key, value in user_system.__dict__.items():
        if key in gs.system.__dict__:
            gs.system.__dict__[key] = value

def load_default():
    f = open("configs/ainodes/default_settings.json", "r")
    settings = json.loads(f.read())
    settings = SimpleNamespace(**settings)
    gs.diffusion = SimpleNamespace(**settings.diffusion)
    gs.system = SimpleNamespace(**settings.system)

def load_settings_json():
    db_base.check_settings_db_status()
    if gs.db_settings_present is True:
        settings = setting_db.get_last_settings()
    # load default anyways to have a bseline of the valid params
    load_default()
    if len(settings) < 1:
        settingsfile = 'configs/ainodes/settings.json'
        if os.path.exists(settingsfile):
            f = open("configs/ainodes/settings.json", "r")
            user_settings = json.loads(f.read())
            push_settings_to_gs(user_settings)
            setting_db.save_settings()
    else:
        user_settings = json.loads(settings[0])
        user_settings['system'] = json.loads(user_settings['system'])
        user_settings['diffusion'] = json.loads(user_settings['diffusion'])
        push_settings_to_gs(user_settings)


def save_settings_json():
    system = json.dumps(gs.system.__dict__)
    diffusion = json.dumps(gs.diffusion.__dict__)
    with open("configs/ainodes/settings.json", "w") as write_file:
        json.dump({
            "system": json.loads(system),
            "diffusion": json.loads(diffusion)
        }, write_file, indent=4)
