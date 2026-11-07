"""Native (in-process) support for current diffusion models through diffusers:
FLUX.2 [klein] / [dev], FLUX.3 [klein] and Krea 2.

Loading is planned from the hardware's *current* headroom (backend.hardware):
the whole pipeline on the GPU when it fits, otherwise fp8 weight storage,
component offload, block streaming or layer streaming, in that order. On
GPUs that support it the transformer's repeated blocks are compiled.

Inpaint / outpaint reuse the backend-agnostic mask handling from
backend.comfyui.inpaint (pre-fill, feathered mask, paste-back).
"""
from dataclasses import dataclass

from backend import hardware

GB = hardware.GB


@dataclass(frozen=True)
class ModernModel:
    key: str
    label: str
    repo: str
    pipeline: str  # diffusers class name
    inpaint_pipeline: str = ""
    steps: int = 4
    guidance: float = 1.0
    size_bytes: int = 0  # all components at bf16
    gated: bool = False
    modular: bool = False


MODELS = {
    "flux2-klein-4b": ModernModel("flux2-klein-4b", "FLUX.2 [klein] 4B", "black-forest-labs/FLUX.2-klein-4B",
                                  "Flux2KleinPipeline", "Flux2KleinInpaintPipeline", 4, 1.0, int(16 * GB)),
    "flux2-klein-9b": ModernModel("flux2-klein-9b", "FLUX.2 [klein] 9B", "black-forest-labs/FLUX.2-klein-9B",
                                  "Flux2KleinPipeline", "Flux2KleinInpaintPipeline", 4, 1.0, int(34 * GB), gated=True),
    "flux2-dev": ModernModel("flux2-dev", "FLUX.2 [dev]", "black-forest-labs/FLUX.2-dev", "Flux2Pipeline",
                             steps=28, guidance=4.0, size_bytes=int(112 * GB), gated=True),
    "krea2-turbo": ModernModel("krea2-turbo", "Krea 2 Turbo", "krea/Krea-2-Turbo", "Krea2Pipeline",
                               steps=8, guidance=1.0, size_bytes=int(33 * GB), gated=True),
    "flux3-klein": ModernModel("flux3-klein", "FLUX.3 [klein]", "black-forest-labs/flux-3-image-klein",
                               "Flux3DistilledModularPipeline", steps=4, guidance=1.0, size_bytes=int(20 * GB),
                               gated=True, modular=True),
}


def working_bytes(width, height):
    """Activations + VAE decode, roughly linear in pixels (measured ~1.5 GB at 1 MP)."""
    return int(1.5 * GB * max(1.0, width * height / (1024 * 1024)))


def _diffusers():
    import diffusers
    return diffusers


def _torch():
    import torch
    return torch


class ModernRunner:
    """Keeps one modern pipeline loaded and places it according to the memory plan."""

    def __init__(self, compile=False, accelerator=None):
        self.compile = compile
        self.acc = accelerator
        self.key = None
        self.pipe = None
        self.plan = None
        self._inpaint = None

    def load(self, key, width=1024, height=1024):
        if self.key == key and self.pipe is not None:
            return self.pipe
        self.unload()
        spec = MODELS[key]
        torch = _torch()
        diffusers = _diffusers()
        acc = self.acc or hardware.detect().primary
        dtype = hardware.compute_dtype(acc)
        plan = hardware.plan_memory(acc, spec.size_bytes, working_bytes(width, height))
        cls = getattr(diffusers, spec.pipeline)
        pipe = cls.from_pretrained(spec.repo, dtype=dtype)
        transformer = getattr(pipe, "transformer", None)
        if plan.fp8_storage and transformer is not None:
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=dtype)
        device = torch.device(acc.device)
        if plan.strategy == "gpu":
            pipe.to(device)
        elif plan.strategy == "model_offload":
            pipe.enable_model_cpu_offload(device=device)
        elif plan.strategy == "group_offload" and transformer is not None:
            transformer.enable_group_offload(onload_device=device, offload_type="block_level",
                                             num_blocks_per_group=1, use_stream=acc.backend in ("cuda", "rocm"))
            for name in ("text_encoder", "vae"):
                part = getattr(pipe, name, None)
                if part is not None:
                    part.to(device)
        else:
            pipe.enable_sequential_cpu_offload(device=device)
        if self.compile and acc.compile and transformer is not None and plan.strategy in ("gpu", "model_offload"):
            transformer.compile_repeated_blocks(fullgraph=True)
        self.acc, self.key, self.pipe, self.plan = acc, key, pipe, plan
        return pipe

    def unload(self):
        self.pipe = self._inpaint = None
        self.key = None
        hardware.empty_cache()

    def _generator(self, seed):
        torch = _torch()
        device = "cpu" if self.acc.backend == "mps" else self.acc.device  # MPS generators are not reproducible
        return torch.Generator(device=device).manual_seed(int(seed))

    def txt2img(self, key, prompt, width=1024, height=1024, seed=0, steps=None, guidance=None, num_images=1):
        spec = MODELS[key]
        pipe = self.load(key, width, height)
        result = pipe(prompt=prompt, width=int(width) // 16 * 16, height=int(height) // 16 * 16,
                      num_inference_steps=steps or spec.steps, guidance_scale=spec.guidance if guidance is None else guidance,
                      generator=self._generator(seed), num_images_per_prompt=num_images)
        return list(result.images)

    def inpaint(self, key, image, mask, prompt, seed=0, steps=None, guidance=None,
                fade_edges=(True, True, True, True)):
        """Regenerate the white area of `mask` in `image` (any size), blending like the ComfyUI path."""
        from PIL import Image
        from backend.comfyui import inpaint as blend
        spec = MODELS[key]
        if not spec.inpaint_pipeline:
            raise RuntimeError(f"{spec.label} has no inpaint pipeline in diffusers yet")
        image = image.convert("RGB")
        hard = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
        if not hard.getbbox():
            return image
        side = max(image.size)
        soft = blend.soften_mask(hard, side // 5, side // 8, fade_edges)  # few-step model: wide regenerated band
        filled = blend.prefill(image, hard)
        work = blend.working_size(image.size, 1024)
        pipe = self.load(key, *work)
        if self._inpaint is None:
            # not from_pipe(): it casts the shared modules to float32, doubling VRAM and undoing fp8 storage
            self._inpaint = getattr(_diffusers(), spec.inpaint_pipeline)(**pipe.components)
        result = self._inpaint(prompt=prompt, image=filled.resize(work, Image.LANCZOS),
                               mask_image=soft.resize(work, Image.BILINEAR), width=work[0], height=work[1],
                               num_inference_steps=steps or spec.steps, strength=1.0,
                               guidance_scale=spec.guidance if guidance is None else guidance,
                               generator=self._generator(seed)).images[0]
        return blend.composite(image, result, soft)


_runner = None


def runner(compile=False):
    global _runner
    if _runner is None:
        _runner = ModernRunner(compile=compile)
    _runner.compile = compile
    return _runner
