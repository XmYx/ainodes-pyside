import json
from types import SimpleNamespace

import pytest

from backend import settings
from backend.singleton import singleton as gs


@pytest.fixture
def fresh_gs(monkeypatch):
    monkeypatch.setattr(gs, "diffusion", SimpleNamespace(), raising=False)
    monkeypatch.setattr(gs, "system", SimpleNamespace(), raising=False)
    return gs


def test_default_settings_are_valid_json_with_comfyui_keys():
    with open("configs/ainodes/default_settings.json") as f:
        data = json.load(f)
    assert {"system", "diffusion"} <= data.keys()
    assert data["system"]["use_comfyui"] is False
    assert data["system"]["comfyui_url"].startswith("http")
    assert "comfyui_checkpoint" in data["system"]


def test_load_default_populates_gs(fresh_gs):
    settings.load_default()
    assert fresh_gs.system.comfyui_url == "http://127.0.0.1:8188"
    assert hasattr(fresh_gs.diffusion, "steps")


def test_push_settings_ignores_unknown_keys(fresh_gs):
    settings.load_default()
    settings.push_settings_to_gs({"system": {"comfyui_url": "http://gpu:8188", "made_up": 1},
                                  "diffusion": {"steps": 3, "also_made_up": 2}})
    assert fresh_gs.system.comfyui_url == "http://gpu:8188"
    assert fresh_gs.diffusion.steps == 3
    assert not hasattr(fresh_gs.system, "made_up")
    assert not hasattr(fresh_gs.diffusion, "also_made_up")


def test_load_settings_json_without_db(fresh_gs, monkeypatch):
    # When no settings DB is present the loader must fall back to the defaults.
    monkeypatch.setattr(settings.db_base, "check_settings_db_status", lambda: None)
    monkeypatch.setattr(gs, "db_settings_present", False, raising=False)
    monkeypatch.setattr(settings.os.path, "exists", lambda path: False)
    settings.load_settings_json()
    assert fresh_gs.system.use_comfyui is False
