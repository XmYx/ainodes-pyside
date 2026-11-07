"""Model families (ComfyUI) and the native diffusers runner, without real weights."""
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from backend import hardware, modern_models
from backend.comfyui import families

GB = hardware.GB


# ---- ComfyUI families -------------------------------------------------------

@pytest.mark.parametrize("name, key", [
    ("krea2_turbo_fp8_scaled.safetensors", "krea2"),
    ("flux2-dev-fp8_scaled.safetensors", "flux2"),
    ("flux-2-klein-4b.safetensors", "flux2_klein"),
    ("flux-3-image-klein-distilled_int8.safetensors", "flux3"),
    ("sd_xl_base_1.0.safetensors", "sdxl"),
    ("v2-1_768-ema-pruned.safetensors", "sd2"),
    ("v1-5-pruned-emaonly.safetensors", "sd15"),
])
def test_detect_family(name, key):
    assert families.detect_family(name).key == key


AVAILABLE = {
    "checkpoints": ["sd_xl_base_1.0.safetensors"],
    "unets": ["krea2_turbo_fp8_scaled.safetensors", "flux2-dev-fp8_scaled.safetensors", "flux2_dev_bf16.safetensors"],
    "clips": ["qwen3vl_4b_fp8_scaled.safetensors", "mistral_3_small_flux2_fp8mixed.safetensors"],
    "clip_types": ["stable_diffusion", "flux2", "krea2"],
    "vaes": ["qwen_image_vae.safetensors", "ae.safetensors"],
}


def test_resolve_components_finds_parts_and_reports_missing():
    krea = families.resolve_components("krea2_turbo_fp8_scaled.safetensors", AVAILABLE)
    assert krea.ready and krea.text_encoder.startswith("qwen3vl") and krea.vae == "qwen_image_vae.safetensors"
    flux = families.resolve_components("flux2-dev-fp8_scaled.safetensors", AVAILABLE)
    assert not flux.ready and any("VAE" in m for m in flux.missing)  # ae.safetensors is the FLUX.1 VAE
    flux3 = families.resolve_components("flux-3-klein.safetensors", AVAILABLE)
    assert any("flux3" in m and "update ComfyUI" in m for m in flux3.missing)


def test_hardware_from_comfy_stats():
    hw = families.ComfyHardware.from_stats({"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync",
                                                          "type": "cuda", "vram_total": 24, "vram_free": 6}]})
    assert hw.fp8 and hw.compile
    assert not families.ComfyHardware.from_stats({"devices": [{"name": "cuda:0 NVIDIA GeForce RTX 3090"}]}).fp8


def test_fp8_weights_only_for_full_precision_files_on_fp8_gpus():
    fp8_gpu = families.ComfyHardware(fp8=True)
    assert families.weight_dtype("flux2_dev_bf16.safetensors", fp8_gpu) == "fp8_e4m3fn_fast"
    assert families.weight_dtype("flux2-dev-fp8_scaled.safetensors", fp8_gpu) == "default"
    assert families.weight_dtype("flux2_dev_bf16.safetensors", families.ComfyHardware(fp8=False)) == "default"


def nodes_by_type(graph):
    found = {}
    for node in graph.values():
        found.setdefault(node["class_type"], []).append(node["inputs"])
    return found


def test_krea2_txt2img_graph_follows_the_template():
    comps = families.resolve_components("krea2_turbo_fp8_scaled.safetensors", AVAILABLE)
    g = nodes_by_type(families.build_graph(comps, "a lake", "ignored negative", seed=5, width=1000, height=600))
    assert g["CLIPLoader"][0]["type"] == "krea2"
    assert "ConditioningZeroOut" in g  # distilled: no negative prompt
    sampler = g["KSampler"][0]
    assert (sampler["steps"], sampler["cfg"], sampler["sampler_name"], sampler["scheduler"]) == (8, 1.0, "euler", "simple")
    assert (g["EmptyLatentImage"][0]["width"], g["EmptyLatentImage"][0]["height"]) == (992, 592)


def test_flux2_graph_uses_guidance_flux2_latent_fp8_and_compile():
    comps = families.Components(families.FAMILIES["flux2"], "flux2_dev_bf16.safetensors", "mistral.safetensors",
                                "flux2-vae.safetensors")
    g = nodes_by_type(families.build_graph(comps, "x", hardware=families.ComfyHardware(fp8=True, compile=True),
                                           compile=True))
    assert g["FluxGuidance"][0]["guidance"] == 4.0
    assert "EmptyFlux2LatentImage" in g and "TorchCompileModel" in g
    assert g["UNETLoader"][0]["weight_dtype"] == "fp8_e4m3fn_fast"


def test_inpaint_graph_for_modern_family():
    comps = families.resolve_components("krea2_turbo_fp8_scaled.safetensors", AVAILABLE)
    g = nodes_by_type(families.build_graph(comps, "x", image="a.png", mask="m.png", denoise=0.45))
    assert {"DifferentialDiffusion", "InpaintModelConditioning", "LoadImageMask"} <= g.keys()
    assert g["KSampler"][0]["denoise"] == 0.45


def test_generation_models_lists_image_models_only():
    names = [m for m, _ in families.generation_models(AVAILABLE)]
    assert "sd_xl_base_1.0.safetensors" in names and "krea2_turbo_fp8_scaled.safetensors" in names


# ---- native diffusers runner -------------------------------------------------

class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: self.calls.append((name, k))


class FakePipe:
    instances = []

    def __init__(self):
        self.transformer = Recorder()
        self.text_encoder = Recorder()
        self.vae = Recorder()
        self.components = {"transformer": self.transformer, "text_encoder": self.text_encoder, "vae": self.vae}
        self.calls = []
        self.last_kwargs = None
        FakePipe.instances.append(self)

    @classmethod
    def from_pretrained(cls, repo, dtype=None):
        pipe = cls()
        pipe.calls.append(("from_pretrained", repo, dtype))
        return pipe

    def to(self, device):
        self.calls.append(("to", str(device)))
        return self

    def enable_model_cpu_offload(self, device=None):
        self.calls.append(("model_offload", str(device)))

    def enable_sequential_cpu_offload(self, device=None):
        self.calls.append(("sequential_offload", str(device)))

    def __call__(self, **kwargs):
        self.last_kwargs = kwargs
        size = (kwargs.get("width", 64), kwargs.get("height", 64))
        return SimpleNamespace(images=[Image.new("RGB", size, (0, 180, 0))])


class FakeInpaintPipe(FakePipe):
    def __init__(self, **components):
        super().__init__()
        self.components = components

    @classmethod
    def from_pipe(cls, pipe):
        raise AssertionError("from_pipe casts the shared modules to float32")


@pytest.fixture
def fake_diffusers(monkeypatch):
    FakePipe.instances = []
    fake = SimpleNamespace(Flux2KleinPipeline=FakePipe, Flux2KleinInpaintPipeline=FakeInpaintPipe, Flux2Pipeline=FakePipe)
    monkeypatch.setattr(modern_models, "_diffusers", lambda: fake)
    return fake


def make_runner(monkeypatch, strategy, fp8, compile_ok=True):
    acc = hardware.Accelerator(backend="cuda", name="test GPU", bf16=True, fp8_storage=True, compile=compile_ok)
    monkeypatch.setattr(hardware, "plan_memory", lambda *a, **k: hardware.MemoryPlan(strategy, fp8, "test"))
    runner = modern_models.ModernRunner(compile=True, accelerator=acc)
    monkeypatch.setattr(runner, "_generator", lambda seed: f"gen{seed}")
    return runner


@pytest.mark.parametrize("strategy, expected", [("gpu", "to"), ("model_offload", "model_offload"),
                                                ("sequential_offload", "sequential_offload")])
def test_runner_places_pipeline_by_plan(monkeypatch, fake_diffusers, strategy, expected):
    runner = make_runner(monkeypatch, strategy, fp8=False)
    images = runner.txt2img("flux2-klein-4b", "a cat", width=1000, height=1000, seed=3)
    pipe = FakePipe.instances[0]
    assert any(c[0] == expected for c in pipe.calls)
    assert pipe.last_kwargs["num_inference_steps"] == 4 and pipe.last_kwargs["width"] == 992
    assert pipe.last_kwargs["generator"] == "gen3" and images[0].size == (992, 992)
    assert ("compile_repeated_blocks", {"fullgraph": True}) in pipe.transformer.calls or strategy == "sequential_offload"


def test_runner_group_offload_and_fp8(monkeypatch, fake_diffusers):
    runner = make_runner(monkeypatch, "group_offload", fp8=True)
    runner.txt2img("flux2-klein-4b", "x")
    calls = [c[0] for c in FakePipe.instances[0].transformer.calls]
    assert "enable_layerwise_casting" in calls and "enable_group_offload" in calls
    assert "compile_repeated_blocks" not in calls  # compiling streamed blocks recompiles every step


def test_runner_skips_compile_where_unsupported(monkeypatch, fake_diffusers):
    runner = make_runner(monkeypatch, "gpu", fp8=False, compile_ok=False)
    runner.txt2img("flux2-klein-4b", "x")
    assert not FakePipe.instances[0].transformer.calls


def test_runner_reuses_loaded_pipeline(monkeypatch, fake_diffusers):
    runner = make_runner(monkeypatch, "gpu", fp8=False)
    runner.txt2img("flux2-klein-4b", "x")
    runner.txt2img("flux2-klein-4b", "y")
    assert len(FakePipe.instances) == 1


def test_runner_inpaint_blends_into_original(monkeypatch, fake_diffusers):
    runner = make_runner(monkeypatch, "gpu", fp8=False)
    image = Image.new("RGB", (512, 512), (200, 0, 0))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).rectangle((256, 0, 511, 511), fill=255)
    out = runner.inpaint("flux2-klein-4b", image, mask, "grass", seed=1)
    assert out.size == (512, 512)
    assert out.getpixel((500, 10)) == (0, 180, 0)
    assert out.getpixel((5, 10))[0] > 190  # the wide few-step blend band only fades out at the far edge


def test_models_without_inpaint_pipeline_say_so(monkeypatch, fake_diffusers):
    runner = make_runner(monkeypatch, "gpu", fp8=False)
    mask = Image.new("L", (64, 64), 255)
    with pytest.raises(RuntimeError, match="no inpaint pipeline"):
        runner.inpaint("flux2-dev", Image.new("RGB", (64, 64)), mask, "x")


def test_every_model_entry_is_complete():
    for spec in modern_models.MODELS.values():
        assert spec.repo.count("/") == 1 and spec.size_bytes > 0 and spec.steps > 0
