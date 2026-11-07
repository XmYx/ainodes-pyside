import numpy as np
import pytest
from PIL import Image, ImageDraw

from backend.comfyui import inpaint as ip


def half_mask(size=128):
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rectangle((size // 2, 0, size - 1, size - 1), fill=255)
    return mask


def test_canvas_mask_uses_transparency():
    rgba = Image.new("RGBA", (4, 1), (10, 20, 30, 255))
    rgba.putpixel((1, 0), (0, 0, 0, 0))
    rgba.putpixel((2, 0), (5, 5, 5, 0))
    assert list(ip.canvas_mask(rgba).getdata()) == [0, 255, 255, 0]


def test_soft_mask_stays_solid_inside_and_feathers_outward():
    # regression: a mask that was <1 inside the generated area blended the
    # canvas' empty (black) pixels back in and left a dark seam
    mask = half_mask()
    soft = np.array(ip.soften_mask(mask, grow=4, feather=6))
    assert soft[:, 64:].min() == 255
    assert 0 < soft[64, 58] < 255
    assert soft[64, 0] == 0


def test_prefill_only_touches_the_masked_area():
    image = Image.new("RGB", (128, 128), (200, 40, 40))
    ImageDraw.Draw(image).rectangle((64, 0, 127, 127), fill=(0, 0, 0))
    filled = np.array(ip.prefill(image, half_mask()))
    assert (filled[:, :64] == (200, 40, 40)).all()
    assert filled[:, 100:, 0].mean() > 150  # the black half picks up the neighbouring red


def test_prefill_without_context_uses_gray():
    full = Image.new("L", (32, 32), 255)
    assert Image.new("RGB", (32, 32)).getpixel((0, 0)) != ip.prefill(Image.new("RGB", (32, 32)), full).getpixel((0, 0))


@pytest.mark.parametrize("name, size", [("sd_xl_base_1.0.safetensors", 1024), ("v1-5-pruned.ckpt", 512),
                                        ("v2-1_768-ema.safetensors", 768)])
def test_model_size(name, size):
    assert ip.model_size(name) == size


def test_working_size_keeps_aspect_and_multiple_of_8():
    assert ip.working_size((512, 512), 1024) == (1024, 1024)
    w, h = ip.working_size((600, 300), 1024)
    assert (w, h) == (1024, 512) and w % 8 == 0 and h % 8 == 0


def test_composite_keeps_unmasked_pixels_exact():
    original = Image.new("RGB", (64, 64), (1, 2, 3))
    generated = Image.new("RGB", (128, 128), (250, 250, 250))
    soft = half_mask(64)
    out = np.array(ip.composite(original, generated, soft))
    assert (out[:, :32] == (1, 2, 3)).all() and (out[:, 32:] == (250, 250, 250)).all()


def test_inpaint_workflow_graph():
    from backend.comfyui import families
    comps = families.Components(families.detect_family("model.safetensors"), "model.safetensors")
    wf = families.build_graph(comps, "a cat", "blurry", seed=3, image="img.png", mask="mask.png", denoise=0.9)
    types = {node["class_type"] for node in wf.values()}
    assert {"DifferentialDiffusion", "InpaintModelConditioning", "LoadImageMask", "KSampler"} <= types
    sampler = wf["8"]["inputs"]
    assert sampler["model"] == ["6", 0]  # sampling goes through DifferentialDiffusion
    assert sampler["latent_image"] == ["7", 2] and sampler["denoise"] == 0.9
    assert wf["7"]["inputs"]["noise_mask"] is True


class FakeClient:
    def __init__(self):
        self.uploads, self.workflows = [], []

    def resolve_checkpoint(self, checkpoint=None):
        return checkpoint or "sd_xl_base_1.0.safetensors"

    def upload_image(self, image, name, subfolder="ainodes"):
        self.uploads.append((name, image.size))
        return f"{subfolder}/{name}"

    def graph(self, model, prompt, steps=None, cfg=None, sampler=None, **kwargs):
        from backend.comfyui import families
        comps = families.Components(families.detect_family(model), model)
        return families.build_graph(comps, prompt, steps=steps, cfg=cfg, sampler=sampler, **kwargs)

    def run_workflow(self, workflow, should_stop=None):
        self.workflows.append(workflow)
        size = next(s for n, s in self.uploads[::-1] if not n.endswith("_mask.png"))
        return [Image.new("RGB", size, (0, 200, 0))]


def test_inpaint_runs_main_and_seam_pass_at_model_resolution():
    client = FakeClient()
    image = Image.new("RGB", (512, 512), (200, 0, 0))
    out = ip.inpaint(client, image, half_mask(512), "grass", seed=7)
    assert out.size == (512, 512)
    assert [s for _, s in client.uploads] == [(1024, 1024)] * 4  # image + mask, twice
    main, seam = client.workflows
    assert main["8"]["inputs"]["denoise"] == 1.0 and seam["8"]["inputs"]["denoise"] == 0.45
    assert seam["8"]["inputs"]["seed"] == 8
    pixels = np.array(out)
    assert (pixels[:, 400:] == (0, 200, 0)).all()  # generated area fully replaced
    assert (pixels[:, :100] == (200, 0, 0)).all()  # far from the edge stays original


def test_inpaint_without_mask_is_a_no_op():
    client = FakeClient()
    image = Image.new("RGB", (64, 64), (9, 9, 9))
    assert ip.inpaint(client, image, Image.new("L", (64, 64), 0), "x") is not None
    assert client.workflows == []


def test_img2img_returns_original_size():
    client = FakeClient()
    out = ip.img2img(client, Image.new("RGB", (300, 200)), "x", denoise=0.4)
    assert out.size == (300, 200)
    assert client.workflows[0]["6"]["inputs"]["denoise"] == 0.4


def test_soften_mask_fades_only_at_inner_tile_edges():
    hard = Image.new("L", (400, 200), 0)
    ImageDraw.Draw(hard).rectangle((200, 0, 399, 199), fill=255)
    faded = ip.soften_mask(hard, 60, 40)
    assert faded.getpixel((170, 0)) == 0 and faded.getpixel((170, 100)) > 150  # kept pixels at a border stay
    assert faded.getpixel((399, 0)) == 255  # the generated area itself stays fully white
    on_canvas_edge = ip.soften_mask(hard, 60, 40, fade_edges=(True, False, True, True))
    assert on_canvas_edge.getpixel((170, 0)) > 150  # top is the canvas edge: no stale strip left
