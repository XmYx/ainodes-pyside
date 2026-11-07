import io
import os
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from backend.comfyui import client as comfy
from backend.comfyui.client import (ComfyUIClient, ComfyUIError, comfy_sampler, pick_checkpoint,
                                    run_txt2img_params, split_prompts)


def png_bytes(color=(255, 0, 0), size=(8, 8)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


class FakeResponse:
    def __init__(self, payload=None, status=200, content=b"", text=""):
        self.payload, self.status_code, self.content, self.text = payload, status, content, text

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class FakeSession:
    """Serves canned ComfyUI responses keyed by path prefix."""

    def __init__(self, routes):
        self.routes = routes
        self.posts = []

    def get(self, url, timeout=None, **kwargs):
        path = url.split("8188", 1)[1]
        for prefix, response in self.routes.items():
            if path.startswith(prefix):
                return response(path) if callable(response) else response
        raise requests.ConnectionError(path)

    def post(self, url, timeout=None, json=None):
        self.posts.append((url, json))
        return self.routes.get("POST " + url.split("8188", 1)[1], FakeResponse({}))


def capture_txt2img(**kwargs):
    client = ComfyUIClient(session=FakeSession({}))
    sent = []
    client.run_workflow = lambda wf, should_stop=None: sent.append(wf) or []
    client.txt2img("a cat", checkpoint="model.safetensors", **kwargs)
    return sent[0]


def test_txt2img_graph_uses_ui_settings_for_checkpoints():
    wf = capture_txt2img(negative_prompt="blurry", width=515, height=770, steps=12, cfg=5.5, seed=7, sampler="euler_a")
    by_type = {n["class_type"]: n["inputs"] for n in wf.values()}
    assert by_type["CheckpointLoaderSimple"]["ckpt_name"] == "model.safetensors"
    assert (by_type["EmptyLatentImage"]["width"], by_type["EmptyLatentImage"]["height"]) == (512, 768)
    sampler = by_type["KSampler"]
    assert sampler["seed"] == 7 and sampler["steps"] == 12 and sampler["cfg"] == 5.5
    assert sampler["sampler_name"] == "euler_ancestral"
    texts = [n["inputs"]["text"] for n in wf.values() if n["class_type"] == "CLIPTextEncode"]
    assert texts == ["a cat", "blurry"]


@pytest.mark.parametrize("seed", [None, "", -1])
def test_txt2img_picks_random_seed_when_unset(seed):
    wf = capture_txt2img(seed=seed)
    value = next(n["inputs"]["seed"] for n in wf.values() if n["class_type"] == "KSampler")
    assert 0 <= value < 2 ** 32


def test_sampler_mapping_falls_back_to_euler():
    assert comfy_sampler("DPMPP_2M") == "dpmpp_2m"
    assert comfy_sampler("plms") == "euler"
    assert comfy_sampler("unknown") == "euler"


def test_pick_checkpoint_skips_vae_files():
    assert pick_checkpoint(["LTX_audio_vae.safetensors", "sd_xl_base_1.0.safetensors"]) == "sd_xl_base_1.0.safetensors"
    assert pick_checkpoint(["only_vae.safetensors"]) == "only_vae.safetensors"
    with pytest.raises(ComfyUIError):
        pick_checkpoint([])


def test_split_prompts_ignores_blank_lines():
    assert split_prompts("a cat\n\n  a dog  \n") == ["a cat", "a dog"]
    assert split_prompts(None) == []


def test_client_txt2img_round_trip():
    history = {"pid": {"status": {"status_str": "success", "completed": True},
                       "outputs": {"7": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]}}}}
    session = FakeSession({
        "/object_info/CheckpointLoaderSimple": FakeResponse(
            {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["x_vae.pt", "real.safetensors"]]}}}}),
        "/history/pid": FakeResponse(history),
        "/view": FakeResponse(content=png_bytes()),
        "POST /prompt": FakeResponse({"prompt_id": "pid"}),
    })
    images = ComfyUIClient(session=session).txt2img("a cat", seed=1)
    assert len(images) == 1 and images[0].size == (8, 8)
    url, body = session.posts[0]
    assert url.endswith("/prompt")
    assert body["prompt"]["1"]["inputs"]["ckpt_name"] == "real.safetensors"


def test_client_reports_rejected_and_failed_prompts():
    rejected = FakeSession({"POST /prompt": FakeResponse(status=400, text="bad node")})
    with pytest.raises(ComfyUIError, match="bad node"):
        ComfyUIClient(session=rejected).queue_prompt({})

    failed = FakeSession({"/history/pid": FakeResponse({"pid": {"status": {"status_str": "error", "messages": []}}})})
    with pytest.raises(ComfyUIError, match="failed"):
        ComfyUIClient(session=failed).wait_for("pid", poll=0)


def test_client_stop_interrupts_server():
    session = FakeSession({"/history/pid": FakeResponse({})})
    with pytest.raises(ComfyUIError, match="interrupted"):
        ComfyUIClient(session=session).wait_for("pid", poll=0, should_stop=lambda: True)
    assert session.posts[0][0].endswith("/interrupt")


def test_is_available_false_when_unreachable():
    assert ComfyUIClient(session=FakeSession({})).is_available() is False


class FakeClient:
    def __init__(self):
        self.calls = []

    def txt2img(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return [Image.new("RGB", (4, 4))]


def make_params(**overrides):
    params = dict(prompts="a cat\na dog", negative_prompts=None, W=512, H=512, steps=10, scale=7.0,
                  seed=100, sampler="euler", n_batch=1, n_samples=1, save_samples=False, outdir=None,
                  seed_behavior="iter")
    params.update(overrides)
    return SimpleNamespace(**params)


def test_run_params_generates_each_prompt_and_iterates_seed():
    client, received = FakeClient(), []
    images = run_txt2img_params(make_params(), client, image_callback=received.append)
    assert len(images) == 2 and received == images
    assert [c[0] for c in client.calls] == ["a cat", "a dog"]
    assert [c[1]["seed"] for c in client.calls] == [100, 101]
    assert client.calls[0][1]["negative_prompt"] == ""


def test_run_params_saves_samples(tmp_path):
    run_txt2img_params(make_params(prompts="one", save_samples=True, outdir=str(tmp_path / "out")), FakeClient())
    assert len(os.listdir(tmp_path / "out")) == 1


def test_run_params_honours_stop():
    client = FakeClient()
    assert run_txt2img_params(make_params(), client, should_stop=lambda: True) == []
    assert client.calls == []


@pytest.mark.comfyui
def test_live_comfyui_server():
    live = ComfyUIClient(os.environ.get("AINODES_COMFYUI_URL", comfy.DEFAULT_URL), timeout=3)
    if not live.is_available():
        pytest.skip("no ComfyUI server running")
    checkpoints = live.list_checkpoints()
    if not checkpoints:
        pytest.skip("ComfyUI has no checkpoints")
    images = live.txt2img("a red cube on a table", width=512, height=512, steps=4, seed=1)
    assert images and images[0].size == (512, 512)
