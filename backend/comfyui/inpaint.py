"""Inpaint, outpaint and img2img through ComfyUI.

Works with plain checkpoints (no inpainting model needed) using the
approach that holds up best for them:

* the area to generate is pre-filled from its surroundings (OpenCV Telea,
  as the INPAINT_MaskedFill node does) so outpainting does not start from
  empty pixels,
* DifferentialDiffusion with a feathered mask, so the denoise fades out
  toward the kept pixels instead of ending in a hard seam,
* the tile is processed at the model's native resolution and scaled back,
* only the masked area is pasted back, so kept pixels stay untouched.

Everything except the sampling runs locally, so only core ComfyUI nodes
are used.
"""
import uuid

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageFilter

from backend.comfyui.families import detect_family

def model_size(checkpoint):
    return detect_family(checkpoint).native_size


def canvas_mask(mask_image):
    """aiNodes canvas convention: transparent pixels are the ones to generate.

    Returns an 'L' mask, 255 = generate, 0 = keep.
    """
    if mask_image.mode != "RGBA":
        return mask_image.convert("L")
    alpha = np.array(mask_image.getchannel("A"))
    return Image.fromarray(np.where(alpha < 1, 255, 0).astype(np.uint8), "L")


def soften_mask(mask, grow=8, feather=12, fade_edges=(True, True, True, True)):
    """Feather the generate area outward into the kept pixels.

    The area itself stays fully white: whatever the canvas holds there
    (usually transparent / black) must never be blended back in.
    In the kept pixels the band fades out towards the image borders named in
    `fade_edges` (left, top, right, bottom): the sides of a canvas tile that
    border other content, so the pasted tile never ends in a half-blended hard
    edge. Sides on the canvas edge need no fade.
    """
    soft = mask
    if grow > 0:
        # cv2 instead of PIL's MaxFilter, which is O(size²) per pixel (~1 min for a 400 px band)
        kernel = np.ones((grow * 2 + 1, grow * 2 + 1), np.uint8)
        soft = Image.fromarray(cv2.dilate(np.array(soft.convert("L")), kernel))
    if feather > 0:
        soft = soft.filter(ImageFilter.GaussianBlur(feather))
        w, h = soft.size
        far = np.full(max(w, h), np.inf)
        left, top, right, bottom = fade_edges
        xs = np.minimum(np.arange(w) if left else far[:w], np.arange(w)[::-1] if right else far[:w])
        ys = np.minimum(np.arange(h) if top else far[:h], np.arange(h)[::-1] if bottom else far[:h])
        ramp = np.clip(np.minimum.outer(ys, xs) / float(feather), 0.0, 1.0)
        soft = Image.fromarray((np.array(soft, np.float32) * ramp).astype(np.uint8))
    return ImageChops.lighter(mask.convert("L"), soft)


def prefill(image, mask):
    """Fill the area to generate from its surroundings (Telea), then soften it."""
    rgb = np.array(image.convert("RGB"))
    hard = (np.array(mask) > 127).astype(np.uint8) * 255
    if not hard.any():
        return image.convert("RGB")
    if hard.all():
        # nothing to extend from: start from neutral gray
        return Image.new("RGB", image.size, (128, 128, 128))
    small = max(1, max(image.size) // 256)  # Telea is slow on big holes; work at reduced size
    if small > 1:
        size = (image.width // small, image.height // small)
        filled = cv2.inpaint(cv2.resize(rgb, size, interpolation=cv2.INTER_AREA),
                             cv2.resize(hard, size, interpolation=cv2.INTER_NEAREST), 5, cv2.INPAINT_TELEA)
        filled = cv2.resize(filled, image.size, interpolation=cv2.INTER_CUBIC)
    else:
        filled = cv2.inpaint(rgb, hard, 5, cv2.INPAINT_TELEA)
    filled = Image.fromarray(filled).filter(ImageFilter.GaussianBlur(max(2, max(image.size) // 64)))
    return Image.composite(filled, image.convert("RGB"), mask)


def working_size(size, target):
    """Scale (w, h) so the long side is `target`, rounded to multiples of 8."""
    w, h = size
    scale = target / max(w, h)
    return max(8, int(round(w * scale / 8)) * 8), max(8, int(round(h * scale / 8)) * 8)


def composite(original, generated, soft_mask):
    generated = generated.convert("RGB").resize(original.size, Image.LANCZOS)
    return Image.composite(generated, original.convert("RGB"), soft_mask)


def seam_band(hard, width):
    """Feathered band straddling the border of the generated area."""
    outer = hard.filter(ImageFilter.MaxFilter(width * 2 + 1))
    inner = hard.filter(ImageFilter.MinFilter(width * 2 + 1))
    return ImageChops.subtract(outer, inner).filter(ImageFilter.GaussianBlur(width / 2))


def _sample(client, image, soft, prompt, checkpoint, negative_prompt, steps, cfg, seed, sampler, denoise, should_stop):
    work = working_size(image.size, model_size(checkpoint))
    tag = uuid.uuid4().hex[:10]
    image_name = client.upload_image(image.resize(work, Image.LANCZOS), f"inpaint_{tag}.png")
    mask_name = client.upload_image(soft.resize(work, Image.BILINEAR).convert("RGB"), f"inpaint_{tag}_mask.png")
    workflow = client.graph(checkpoint, prompt, steps=steps, cfg=cfg, sampler=sampler, negative_prompt=negative_prompt,
                            seed=seed, image=image_name, mask=mask_name, denoise=denoise,
                            filename_prefix="aiNodes_inpaint")
    return client.run_workflow(workflow, should_stop=should_stop)[0]


def inpaint(client, image, mask, prompt, checkpoint=None, negative_prompt="", steps=25, cfg=6.0, seed=0,
            sampler="dpmpp_2m", denoise=1.0, grow=None, feather=None, seam_fix=True, should_stop=None,
            fade_edges=(True, True, True, True)):
    """Regenerate the white area of `mask` ('L', 255 = generate) in `image`; returns an RGB image.

    grow / feather default to a share of the tile size. With seam_fix a second,
    light pass re-renders a band across the border so the new content joins
    the kept pixels without a visible edge.
    """
    checkpoint = client.resolve_checkpoint(checkpoint)
    image = image.convert("RGB")
    hard = mask.convert("L").point(lambda v: 255 if v > 127 else 0)
    if not hard.getbbox():
        return image
    side = max(image.size)
    few_steps = detect_family(checkpoint).steps <= 10
    outpaint = sum(hard.histogram()[128:]) > 0.25 * image.width * image.height
    if few_steps and outpaint:
        # Distilled few-step models (Krea 2 Turbo, Flux klein / 3) barely move pixels the
        # differential mask only partly frees, so regenerate a wide band of the old content.
        default_grow, default_feather = side // 5, side // 8
    else:
        default_grow, default_feather = max(4, side // 25), max(6, side // 16)
    grow = default_grow if grow is None else grow
    feather = default_feather if feather is None else feather
    soft = soften_mask(hard, grow, feather, fade_edges)
    filled = prefill(image, hard)  # the blend band keeps its real pixels so the model continues them
    generated = _sample(client, filled, soft, prompt, checkpoint, negative_prompt, steps, cfg, seed, sampler,
                        denoise, should_stop)
    result = composite(image, generated, soft)
    if seam_fix and hard.getextrema()[0] == 0:
        band = seam_band(hard, max(8, side // 16))
        fixed = _sample(client, result, band, prompt, checkpoint, negative_prompt, max(10, steps * 2 // 3), cfg,
                        seed + 1, sampler, 0.45, should_stop)
        result = composite(result, fixed, band)
    return result


def img2img(client, image, prompt, checkpoint=None, negative_prompt="", steps=25, cfg=6.0, seed=0,
            sampler="dpmpp_2m", denoise=0.5, should_stop=None):
    checkpoint = client.resolve_checkpoint(checkpoint)
    image = image.convert("RGB")
    work = working_size(image.size, model_size(checkpoint))
    name = client.upload_image(image.resize(work, Image.LANCZOS), f"img2img_{uuid.uuid4().hex[:10]}.png")
    workflow = client.graph(checkpoint, prompt, steps=steps, cfg=cfg, sampler=sampler, negative_prompt=negative_prompt,
                            seed=seed, image=name, denoise=denoise, filename_prefix="aiNodes_img2img")
    generated = client.run_workflow(workflow, should_stop=should_stop)[0]
    return generated.resize(image.size, Image.LANCZOS)
