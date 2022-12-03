import itertools
import os
from pathlib import Path
import html
import gc
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from tqdm.auto import tqdm, trange
from backend.singleton import singleton
gs = singleton

def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


def generate_imgs_embd(name="doom", folder="learn", batch_size=5):
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-large-patch14",
    ).to("cuda")
    processor = CLIPProcessor.from_pretrained(model.name_or_path)

    with torch.no_grad():
        embs = []
        for paths in tqdm(iter_to_batched(get_all_images_in_folder(folder), batch_size),
                          desc=f"Generating embeddings for {name}"):
            print(paths)
            inputs = processor(images=[Image.open(path) for path in paths], return_tensors="pt").to("cuda")
            outputs = model.get_image_features(**inputs).cpu()
            embs.append(torch.clone(outputs))
            inputs.to("cpu")
            del inputs, outputs

        embs = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

        # The generated embedding will be located here
        path = f"models/aesthetic_gradients_dir/{name}.pt"
        torch.save(embs, path)

        model.cpu()
        del processor
        del embs
        gc.collect()
        torch.cuda.empty_cache()
        res = f"Done generating embedding for {name} saved to {path}"
        return res


def get_all_images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if
            os.path.isfile(os.path.join(folder, f)) and check_is_valid_image_file(f)]


def check_is_valid_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', ".gif", ".tiff", ".webp"))


def batched(dataset, total, n=1):
    for ndx in range(0, total, n):
        yield [dataset.__getitem__(i) for i in range(ndx, min(ndx + n, total))]


def iter_to_batched(iterable, n=1):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk
