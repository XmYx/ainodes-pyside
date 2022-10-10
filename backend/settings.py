import json
from types import SimpleNamespace
from backend.singleton import singleton

gs = singleton


def load_settings_json():
    f = open("configs/ainodes/webui_settings.json", "r")
    gs.defaults = json.loads(f.read())

    gs.defaults = SimpleNamespace(**gs.defaults)
    gs.defaults.general = SimpleNamespace(**gs.defaults.general)
