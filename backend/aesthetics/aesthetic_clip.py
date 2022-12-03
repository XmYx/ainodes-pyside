import copy
import itertools
import os
from pathlib import Path
import html
import gc
from collections import OrderedDict


import torch
from PIL import Image
from torch import optim

from backend.devices import choose_torch_device
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from tqdm.auto import tqdm, trange
from backend.singleton import singleton
from backend.torch_gc import torch_gc

gs = singleton


#aesthetic_embeddings_dir = os.path.join(scripts.basedir(), "aesthetic_embeddings")
#os.makedirs(aesthetic_embeddings_dir, exist_ok=True)

aesthetic_embeddings = {}
aesthetic_embeddings_dir = ''

def update_aesthetic_embeddings():
    global aesthetic_embeddings
    aesthetic_embeddings = {f.replace(".pt", ""): os.path.join(aesthetic_embeddings_dir, f) for f in os.listdir(aesthetic_embeddings_dir) if f.endswith(".pt")}
    aesthetic_embeddings = OrderedDict(**{"None": None}, **aesthetic_embeddings)


#update_aesthetic_embeddings()


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



aesthetic_clip_model = None


def aesthetic_clip():
    global aesthetic_clip_model

    if aesthetic_clip_model is None or aesthetic_clip_model.name_or_path != gs.models["sd"].cond_stage_model.wrapped.transformer.name_or_path:
        aesthetic_clip_model = CLIPModel.from_pretrained(gs.models["sd"].cond_stage_model.wrapped.transformer.name_or_path)
        aesthetic_clip_model.cpu()

    return aesthetic_clip_model


def generate_imgs_embd(name, folder, batch_size):
    device = choose_torch_device()
    model = aesthetic_clip().to(device)
    processor = CLIPProcessor.from_pretrained(model.name_or_path)

    with torch.no_grad():
        embs = []
        for paths in tqdm(iter_to_batched(get_all_images_in_folder(folder), batch_size),
                          desc=f"Generating embeddings for {name}"):
            #if shared.state.interrupted:
            #    break
            inputs = processor(images=[Image.open(path) for path in paths], return_tensors="pt").to(device)
            outputs = model.get_image_features(**inputs).cpu()
            embs.append(torch.clone(outputs))
            inputs.to("cpu")
            del inputs, outputs

        embs = torch.cat(embs, dim=0).mean(dim=0, keepdim=True)

        # The generated embedding will be located here
        path = str(Path(aesthetic_embeddings_dir) / f"{name}.pt")
        torch.save(embs, path)

        model.cpu()
        del processor
        del embs
        gc.collect()
        torch.cuda.empty_cache()
        res = f"""
Done generating embedding for {name}!
Aesthetic embedding saved to {html.escape(path)}
"""
        update_aesthetic_embeddings()
        return res


def slerp(low, high, val):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res


class AestheticCLIP:
    def __init__(self):
        self.skip = False
        self.aesthetic_steps = 0
        self.aesthetic_weight = 0
        self.aesthetic_lr = 0
        self.slerp = False
        self.aesthetic_text_negative = ""
        self.aesthetic_slerp_angle = 0
        self.aesthetic_imgs_text = ""

        self.image_embs_name = None
        self.image_embs = None
        #self.load_image_embs(None)
        self.process_tokens = None
        self.device = choose_torch_device()

    def set_aesthetic_params(self, p={}, aesthetic_lr=0, aesthetic_weight=0, aesthetic_steps=0, image_embs_name=None,
                             aesthetic_slerp=True, aesthetic_imgs_text="",
                             aesthetic_slerp_angle=0.15,
                             aesthetic_text_negative=False):

        self.aesthetic_imgs_text = aesthetic_imgs_text
        self.aesthetic_slerp_angle = aesthetic_slerp_angle
        self.aesthetic_text_negative = aesthetic_text_negative
        self.slerp = aesthetic_slerp
        self.aesthetic_lr = aesthetic_lr
        self.aesthetic_weight = aesthetic_weight
        self.aesthetic_steps = aesthetic_steps
        self.image_embs_name = None
        self.load_image_embs(image_embs_name)


        #if self.image_embs_name is not None:
        #    p.extra_generation_params.update({
        #        "Aesthetic LR": aesthetic_lr,
        #        "Aesthetic weight": aesthetic_weight,
        #        "Aesthetic steps": aesthetic_steps,
        #        "Aesthetic embedding": self.image_embs_name,
        #        "Aesthetic slerp": aesthetic_slerp,
        #        "Aesthetic text": aesthetic_imgs_text,
        #        "Aesthetic text negative": aesthetic_text_negative,
        #        "Aesthetic slerp angle": aesthetic_slerp_angle,
        #    })

    def set_skip(self, skip):
        self.skip = skip

    def load_image_embs(self, image_embs_name):
        self.device = choose_torch_device()
        if image_embs_name is None or len(image_embs_name) == 0 or image_embs_name == "None":
            image_embs_name = None
            self.image_embs_name = None
        #print(gs.aesthetic_embedding_path)
        #print(os.path.join(gs.system.aesthetic_gradients, 'None'))
        if gs.diffusion.selected_aesthetic_embedding is not 'None':
            self.image_embs_name = image_embs_name
            self.image_embs = torch.load(os.path.join(gs.system.aesthetic_gradients, gs.diffusion.selected_aesthetic_embedding), map_location=self.device)
            self.image_embs /= self.image_embs.norm(dim=-1, keepdim=True)
            self.image_embs.requires_grad_(False)

    def __call__(self, remade_batch_tokens, multipliers):
        print(gs.diffusion.selected_aesthetic_embedding)
        if gs.diffusion.selected_aesthetic_embedding != 'None':
            self.load_image_embs(gs.diffusion.selected_aesthetic_embedding)
        z = self.process_tokens(remade_batch_tokens, multipliers)
        self.aesthetic_steps = gs.T
        self.aesthetic_lr = gs.lr
        self.image_embs_name = gs.aesthetic_embedding_path

        self.device = choose_torch_device()
        gs.CLIP_stop_at_last_layers = 1
        #print(self.aesthetic_embedding_path)

        if not self.skip and self.aesthetic_steps != 0 and self.aesthetic_lr != 0 and self.aesthetic_weight != 0 and gs.aesthetic_embedding_path != os.path.join(gs.system.aesthetic_gradients, 'None'):
            #gs.models["sd"].cond_stage_model.tokenizer = gs.models["sd"].cond_stage_model.tokenizer
            use_old_emphasis_implementation = None
            if not use_old_emphasis_implementation:
                remade_batch_tokens = [[gs.models["sd"].cond_stage_model.tokenizer.bos_token_id] + x[:75] + [gs.models["sd"].cond_stage_model.tokenizer.eos_token_id] for x in remade_batch_tokens]

            tokens = torch.asarray(remade_batch_tokens).to(self.device)

            model = copy.deepcopy(aesthetic_clip()).to(self.device)
            model.requires_grad_(True)
            if self.aesthetic_imgs_text is not "" and len(self.aesthetic_imgs_text) > 0:
                text_embs_2 = model.get_text_features(
                    **gs.models["sd"].cond_stage_model.tokenizer([self.aesthetic_imgs_text], padding=True, return_tensors="pt").to(self.device))
                if self.aesthetic_text_negative:
                    text_embs_2 = self.image_embs - text_embs_2
                    text_embs_2 /= text_embs_2.norm(dim=-1, keepdim=True)
                img_embs = slerp(self.image_embs, text_embs_2, self.aesthetic_slerp_angle)
            else:
                img_embs = self.image_embs

            with torch.enable_grad():

                # We optimize the model to maximize the similarity
                optimizer = optim.Adam(
                    model.text_model.parameters(), lr=self.aesthetic_lr
                )
                gs.models["sd"].to("cpu")
                for _ in trange(self.aesthetic_steps, desc="Aesthetic optimization"):

                    text_embs = model.get_text_features(input_ids=tokens)
                    text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
                    sim = text_embs @ img_embs.T
                    loss = -sim
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
                    #print(loss)

                zn = model.text_model(input_ids=tokens, output_hidden_states=-gs.CLIP_stop_at_last_layers)
                if gs.CLIP_stop_at_last_layers > 1:
                    zn = zn.hidden_states[-gs.CLIP_stop_at_last_layers]
                    zn = model.text_model.final_layer_norm(zn)
                else:
                    zn = zn.last_hidden_state
                model.cpu()

                del model
                del optimizer
                del loss
                del sim
                del text_embs
                del img_embs

                torch_gc()
            zn = torch.concat([zn[77 * i:77 * (i + 1)] for i in range(max(z.shape[1] // 77, 1))], 1)
            if self.slerp:
                z = slerp(z, zn, self.aesthetic_weight)
            else:
                z = z * (1 - self.aesthetic_weight) + zn * self.aesthetic_weight
        gs.models["sd"].to("cuda")
        return z
