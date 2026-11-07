"""Model families for the ComfyUI backend: SD 1.5, SDXL, Krea 2, Flux.2, Flux 3.

Each family is a recipe taken from ComfyUI's official workflow templates:
which text encoder (and CLIP type) and VAE go with the diffusion model,
and the sampler settings it was tuned for. build_graph() turns a recipe
into one ComfyUI graph for txt2img, img2img or inpaint, so the canvas,
batch outpainting and hires all work with every family.

Speed / memory: on GPUs with fp8 matmuls (RTX 40/50, H100, B200, MI300...)
full-precision diffusion models load as fp8_e4m3fn_fast, and compile=True
adds TorchCompileModel (inductor).
"""
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Family:
    key: str
    label: str
    loader: str  # "checkpoint" (one file) or "diffusion" (UNET + text encoder + VAE)
    native_size: int
    steps: int
    cfg: float
    sampler: str = "euler"
    scheduler: str = "simple"
    clip_type: str = ""
    text_encoders: tuple = ()  # filename patterns, best first
    vaes: tuple = ()
    guidance: float = 0.0  # FluxGuidance on the positive prompt
    zero_negative: bool = False  # distilled models ignore the negative prompt
    notes: str = ""


FAMILIES = {
    "sd15": Family("sd15", "Stable Diffusion 1.x", "checkpoint", 512, 25, 7.0, "dpmpp_2m", "karras"),
    "sd2": Family("sd2", "Stable Diffusion 2.x", "checkpoint", 768, 25, 7.0, "dpmpp_2m", "karras"),
    "sdxl": Family("sdxl", "SDXL", "checkpoint", 1024, 25, 6.0, "dpmpp_2m", "karras"),
    "krea2": Family("krea2", "Krea 2 Turbo", "diffusion", 1024, 8, 1.0, "euler", "simple", "krea2",
                    (r"qwen3vl_4b", r"qwen3.?vl.?4b"), (r"qwen_image_vae",), zero_negative=True),
    "flux2": Family("flux2", "FLUX.2 [dev]", "diffusion", 1024, 20, 1.0, "euler", "simple", "flux2",
                    (r"mistral_3_small_flux2", r"mistral.*flux2"), (r"flux2.?vae",), guidance=4.0,
                    zero_negative=True),
    "flux2_klein": Family("flux2_klein", "FLUX.2 [klein]", "diffusion", 1024, 4, 1.0, "euler", "simple", "flux2",
                          (r"qwen_3_8b", r"qwen_3_4b", r"qwen3_8b", r"qwen3_4b"), (r"flux2.?vae",),
                          zero_negative=True, notes="base checkpoints want ~20 steps and cfg 5"),
    "flux3": Family("flux3", "FLUX.3 [klein]", "diffusion", 1024, 4, 1.0, "euler", "simple", "flux3",
                    (r"qwen3.?vl.?4b", r"qwen_3_vl_4b"), (r"flux2.?vae",), zero_negative=True,
                    notes="needs a ComfyUI with the flux3 CLIP type (Oct 2026 or newer)"),
}

PREQUANTIZED = re.compile(r"fp8|int8|nvfp4|fp4|gguf|q[2-8]_", re.I)


def detect_family(model_name):
    name = (model_name or "").lower().replace("-", "_").replace(".", "_")
    if "krea2" in name or "krea_2" in name:
        return FAMILIES["krea2"]
    if re.search(r"flux_?3", name):
        return FAMILIES["flux3"]
    if re.search(r"flux_?2", name):
        return FAMILIES["flux2_klein"] if "klein" in name else FAMILIES["flux2"]
    if "xl" in name:
        return FAMILIES["sdxl"]
    if re.search(r"v2|768|2_1|sd2", name):
        return FAMILIES["sd2"]
    return FAMILIES["sd15"]


@dataclass
class ComfyHardware:
    """What the ComfyUI server's GPU can do, from /system_stats."""
    name: str = ""
    vram_total: int = 0
    vram_free: int = 0
    fp8: bool = False
    compile: bool = False

    @classmethod
    def from_stats(cls, stats):
        devices = (stats or {}).get("devices") or [{}]
        dev = devices[0]
        name = dev.get("name", "")
        fp8 = bool(re.search(r"RTX\s*(40|50)\d\d|RTX PRO|\bL4\b|L40|H100|H200|H20\b|B100|B200|GB200|MI3\d\d|RX\s*9\d\d\d",
                             name, re.I))
        return cls(name=name, vram_total=dev.get("vram_total", 0), vram_free=dev.get("vram_free", 0),
                   fp8=fp8, compile=dev.get("type") in ("cuda",))


@dataclass
class Components:
    family: Family
    model: str
    text_encoder: str = ""
    vae: str = ""
    missing: list = field(default_factory=list)

    @property
    def ready(self):
        return not self.missing


def _pick(patterns, available):
    for pattern in patterns:
        for name in available:
            if re.search(pattern, name, re.I):
                return name
    return ""


def resolve_components(model, available):
    """Find text encoder / VAE files for `model`. available: {"checkpoints", "unets", "clips", "vaes"} lists."""
    family = detect_family(model)
    comps = Components(family, model)
    if family.loader == "checkpoint":
        if model not in available.get("checkpoints", []):
            comps.missing.append(f"checkpoint {model}")
        return comps
    if model not in available.get("unets", []):
        comps.missing.append(f"diffusion model {model} (models/diffusion_models)")
    comps.text_encoder = _pick(family.text_encoders, available.get("clips", []))
    comps.vae = _pick(family.vaes, available.get("vaes", []))
    if not comps.text_encoder:
        comps.missing.append(f"{family.label} text encoder (models/text_encoders, e.g. {family.text_encoders[0]})")
    if not comps.vae:
        comps.missing.append(f"{family.label} VAE (models/vae, e.g. {family.vaes[0].replace('.?', '-')})")
    if family.clip_type and family.clip_type not in available.get("clip_types", [family.clip_type]):
        comps.missing.append(f"CLIP type '{family.clip_type}' (update ComfyUI)")
    return comps


def weight_dtype(model, hardware):
    """fp8 compute for full-precision files on fp8-capable GPUs; quantized files load as they are."""
    if hardware and hardware.fp8 and not PREQUANTIZED.search(model):
        return "fp8_e4m3fn_fast"
    return "default"


class _Graph:
    def __init__(self):
        self.nodes = {}

    def add(self, class_type, **inputs):
        key = str(len(self.nodes) + 1)
        self.nodes[key] = {"class_type": class_type, "inputs": inputs}
        return key


def build_graph(comps, prompt, negative_prompt="", width=1024, height=1024, seed=0, steps=None, cfg=None,
                sampler=None, scheduler=None, batch_size=1, image=None, mask=None, denoise=1.0,
                hardware=None, compile=False, filename_prefix="aiNodes"):
    """One ComfyUI graph for txt2img (no image), img2img (image) or inpaint (image + mask).

    image / mask are names of files already uploaded to ComfyUI's input folder.
    """
    fam = comps.family
    g = _Graph()
    if fam.loader == "checkpoint":
        loader = g.add("CheckpointLoaderSimple", ckpt_name=comps.model)
        model, clip, vae = [loader, 0], [loader, 1], [loader, 2]
    else:
        unet = g.add("UNETLoader", unet_name=comps.model, weight_dtype=weight_dtype(comps.model, hardware))
        clip_node = g.add("CLIPLoader", clip_name=comps.text_encoder, type=fam.clip_type, device="default")
        vae_node = g.add("VAELoader", vae_name=comps.vae)
        model, clip, vae = [unet, 0], [clip_node, 0], [vae_node, 0]
    if compile and (hardware is None or hardware.compile):
        model = [g.add("TorchCompileModel", model=model, backend="inductor"), 0]

    positive = [g.add("CLIPTextEncode", text=prompt, clip=clip), 0]
    if fam.guidance:
        positive = [g.add("FluxGuidance", conditioning=positive, guidance=fam.guidance), 0]
    if fam.zero_negative or not negative_prompt:
        negative = [g.add("ConditioningZeroOut", conditioning=positive), 0] if fam.zero_negative else \
            [g.add("CLIPTextEncode", text="", clip=clip), 0]
    else:
        negative = [g.add("CLIPTextEncode", text=negative_prompt, clip=clip), 0]

    if image and mask:
        pixels = [g.add("LoadImage", image=image), 0]
        mask_ref = [g.add("LoadImageMask", image=mask, channel="red"), 0]
        model = [g.add("DifferentialDiffusion", model=model), 0]
        cond = g.add("InpaintModelConditioning", positive=positive, negative=negative, vae=vae, pixels=pixels,
                     mask=mask_ref, noise_mask=True)
        positive, negative, latent = [cond, 0], [cond, 1], [cond, 2]
    elif image:
        pixels = [g.add("LoadImage", image=image), 0]
        latent = [g.add("VAEEncode", pixels=pixels, vae=vae), 0]
    else:
        latent_type = "EmptyFlux2LatentImage" if fam.key.startswith("flux") else "EmptyLatentImage"
        latent = [g.add(latent_type, width=int(width) // 16 * 16, height=int(height) // 16 * 16,
                        batch_size=int(batch_size)), 0]
        denoise = 1.0

    samples = g.add("KSampler", model=model, positive=positive, negative=negative, latent_image=latent,
                    seed=int(seed), steps=int(steps or fam.steps), cfg=float(fam.cfg if cfg is None else cfg),
                    sampler_name=sampler or fam.sampler, scheduler=scheduler or fam.scheduler,
                    denoise=float(denoise))
    decoded = g.add("VAEDecode", samples=[samples, 0], vae=vae)
    g.add("SaveImage", images=[decoded, 0], filename_prefix=filename_prefix)
    return g.nodes


def available_models(client):
    """Model, text encoder and VAE files a ComfyUI server offers."""
    def options(node, field_name):
        info = client._get(f"/object_info/{node}").json()[node]["input"]["required"]
        values = info[field_name][0]
        return list(values if isinstance(values, list) else info[field_name][1].get("options", []))
    return {
        "checkpoints": options("CheckpointLoaderSimple", "ckpt_name"),
        "unets": options("UNETLoader", "unet_name"),
        "clips": options("CLIPLoader", "clip_name"),
        "clip_types": options("CLIPLoader", "type"),
        "vaes": options("VAELoader", "vae_name"),
    }


def generation_models(available):
    """Models a user can pick for image generation, labelled with their family."""
    image_unets = [u for u in available.get("unets", []) if detect_family(u).key in ("krea2", "flux2", "flux2_klein", "flux3")]
    checkpoints = [c for c in available.get("checkpoints", []) if "vae" not in c.lower()]
    return [(m, detect_family(m).label) for m in checkpoints + image_unets]
