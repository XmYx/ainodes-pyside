"""Minimal ComfyUI HTTP client: run a txt2img graph on a ComfyUI server.

Only the stock API endpoints are used (/system_stats, /object_info,
/prompt, /history, /view), so any recent ComfyUI build works without
custom nodes.
"""
import io
import json
import random
import time
import urllib.parse
import uuid

import requests
from PIL import Image

DEFAULT_URL = "http://127.0.0.1:8188"

# aiNodes sampler names -> ComfyUI KSampler sampler_name
SAMPLER_MAP = {
    "euler": "euler",
    "euler_a": "euler_ancestral",
    "euler_ancestral": "euler_ancestral",
    "heun": "heun",
    "dpm2": "dpm_2",
    "dpm2_a": "dpm_2_ancestral",
    "dpm2_ancestral": "dpm_2_ancestral",
    "lms": "lms",
    "dpm_fast": "dpm_fast",
    "dpm_adaptive": "dpm_adaptive",
    "dpmpp_2s_a": "dpmpp_2s_ancestral",
    "dpmpp_2m": "dpmpp_2m",
    "dpmpp_sde": "dpmpp_sde",
    "ddim": "ddim",
    "plms": "euler",  # ComfyUI has no PLMS
}


class ComfyUIError(RuntimeError):
    pass


def comfy_sampler(name):
    return SAMPLER_MAP.get(str(name).lower(), "euler")


def pick_checkpoint(checkpoints):
    """Default checkpoint: the first one that is not a standalone VAE file."""
    if not checkpoints:
        raise ComfyUIError("ComfyUI has no checkpoints installed")
    usable = [c for c in checkpoints if "vae" not in c.lower()]
    return (usable or checkpoints)[0]


def build_txt2img_workflow(prompt, checkpoint, negative_prompt="", width=512, height=512,
                           steps=20, cfg=7.0, seed=None, sampler="euler", scheduler="normal",
                           batch_size=1, filename_prefix="aiNodes"):
    """Return a ComfyUI API-format graph for a plain checkpoint txt2img run."""
    if seed is None or seed == "" or int(seed) < 0:
        seed = random.randint(0, 2 ** 32 - 1)
    # ComfyUI latents must be multiples of 8
    width, height = int(width) // 8 * 8, int(height) // 8 * 8
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt or "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": int(batch_size)}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": int(seed), "steps": int(steps),
                         "cfg": float(cfg), "sampler_name": comfy_sampler(sampler),
                         "scheduler": scheduler, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix}},
    }


class ComfyUIClient:
    def __init__(self, base_url=DEFAULT_URL, timeout=10, session=None):
        self.base_url = (base_url or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self.client_id = uuid.uuid4().hex

    def _get(self, path, **kwargs):
        response = self.session.get(f"{self.base_url}{path}", timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def is_available(self):
        try:
            self._get("/system_stats")
            return True
        except requests.RequestException:
            return False

    def system_stats(self):
        return self._get("/system_stats").json()

    def list_checkpoints(self):
        info = self._get("/object_info/CheckpointLoaderSimple").json()
        return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])

    def queue_prompt(self, workflow):
        response = self.session.post(f"{self.base_url}/prompt", timeout=self.timeout,
                                     json={"prompt": workflow, "client_id": self.client_id})
        if response.status_code != 200:
            raise ComfyUIError(f"ComfyUI rejected the workflow: {response.text[:500]}")
        return response.json()["prompt_id"]

    def wait_for(self, prompt_id, poll=0.5, max_wait=600, should_stop=None):
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            if should_stop and should_stop():
                self.interrupt()
                raise ComfyUIError("interrupted")
            history = self._get(f"/history/{prompt_id}").json()
            entry = history.get(prompt_id)
            if entry:
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyUIError(f"ComfyUI execution failed: {json.dumps(status.get('messages'))[:500]}")
                if status.get("completed", True):
                    return entry
            time.sleep(poll)
        raise ComfyUIError(f"timed out after {max_wait}s waiting for {prompt_id}")

    def interrupt(self):
        try:
            self.session.post(f"{self.base_url}/interrupt", timeout=self.timeout)
        except requests.RequestException:
            pass

    def fetch_images(self, history_entry):
        images = []
        for output in history_entry.get("outputs", {}).values():
            for item in output.get("images", []):
                query = urllib.parse.urlencode({"filename": item["filename"],
                                                "subfolder": item.get("subfolder", ""),
                                                "type": item.get("type", "output")})
                data = self._get(f"/view?{query}").content
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
        return images

    def txt2img(self, prompt, checkpoint=None, should_stop=None, **kwargs):
        if not checkpoint:
            checkpoint = pick_checkpoint(self.list_checkpoints())
        workflow = build_txt2img_workflow(prompt, checkpoint, **kwargs)
        entry = self.wait_for(self.queue_prompt(workflow), should_stop=should_stop)
        return self.fetch_images(entry)


def split_prompts(text):
    """aiNodes prompt boxes hold one prompt per line; blank lines are ignored."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def run_txt2img_params(params, client, image_callback=None, should_stop=None, checkpoint=None):
    """Generate every prompt in an aiNodes params namespace through ComfyUI.

    Images are saved to params.outdir (when save_samples is on) and passed
    to image_callback one by one, like the local deforum path does.
    """
    import os

    images = []
    seed = params.seed if params.seed not in ("", None) else None
    for prompt in split_prompts(params.prompts):
        for _ in range(max(1, int(getattr(params, "n_batch", 1) or 1))):
            if should_stop and should_stop():
                return images
            result = client.txt2img(prompt, checkpoint=checkpoint, should_stop=should_stop,
                                    negative_prompt=params.negative_prompts or "",
                                    width=params.W, height=params.H, steps=params.steps,
                                    cfg=params.scale, seed=seed, sampler=params.sampler,
                                    batch_size=getattr(params, "n_samples", 1) or 1)
            for image in result:
                if getattr(params, "save_samples", False) and getattr(params, "outdir", None):
                    os.makedirs(params.outdir, exist_ok=True)
                    image.save(os.path.join(params.outdir, f"comfyui_{uuid.uuid4().hex[:10]}.png"))
                if image_callback:
                    image_callback(image)
                images.append(image)
            if seed is not None and getattr(params, "seed_behavior", "") == "iter":
                seed = int(seed) + 1
    return images
