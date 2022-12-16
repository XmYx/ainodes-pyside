import torch
from PIL import Image

from lavis.processors import load_processor

from lavis.models import BlipBase, load_model

def load_model_cache(name, model_type, is_eval, device):
    return load_model(name, model_type, is_eval, device)

def run():
    raw_img = Image.open("G:/sd/xmyx/ainodes-pyside/data/input/images/00000-0-1.png").convert("RGB")

    # setup device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # we associate a model with its preprocessors to make it easier for inference.
    #model, vis_processors, _ = load_model_and_preprocess(
    #    name="blip_caption", model_type="large_coco", is_eval=True, device=device
    #)
    # uncomment to use base model
    #model, vis_processors, _ = load_model_and_preprocess(
    #    name="blip_caption", model_type="base_coco", is_eval=True, device=device
    # )
    #vis_processors.keys()

    vis_processor = load_processor("blip_image_eval").build(image_size=384)

    model_type= "BLIP_large"

    if model_type.startswith("BLIP"):
        blip_type = model_type.split("_")[1].lower()
        model = load_model_cache(
            "blip_caption",
            model_type=f"{blip_type}_coco",
            is_eval=True,
            device=device,
        )

    use_beam = False
    img = vis_processor(raw_img).unsqueeze(0).to(device)
    captions = generate_caption(
        model=model, image=img, use_nucleus_sampling=not use_beam
    )






    image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

    print(model.generate({"image": image}))



def generate_caption(
        model, image, use_nucleus_sampling=False, num_beams=3, max_length=40, min_length=5
):
    samples = {"image": image}

    captions = []
    if use_nucleus_sampling:
        for _ in range(5):
            caption = model.generate(
                samples,
                use_nucleus_sampling=True,
                max_length=max_length,
                min_length=min_length,
                top_p=0.9,
            )
            captions.append(caption[0])
    else:
        caption = model.generate(
            samples,
            use_nucleus_sampling=False,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
        )
        captions.append(caption[0])

    return captions
