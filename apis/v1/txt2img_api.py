import os
import sys
import random
from typing import Union

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from backend.singleton import singleton
from backend.deforum.deforum_adapter import DeforumSix
from apis.v1.http.response import respond_500

gs = singleton

router = APIRouter()

def_six_render = DeforumSix()

class Txt2Img(BaseModel):
    prompt: str
    W: int
    H: int
    scale: float
    seed: Union[int, str, None] = random.randint(0, 2 ** 32 - 1)
    iterations: int
    batch_size: int
    steps: int
    sampler: str
    sampling_mode: str
    separate_prompts: bool
    normalize_prompt_weights: bool
    save_individual_images: bool
    save_grid: bool
    group_by_prompt: bool
    save_as_jpg: bool
    use_gfpgan: bool
    use_realesrgan: bool
    realesrgan_model: str
    realesrgan_model_name: str
    fp: Union[int, None] = None
    variant_amount: int
    variant_seed: Union[int, None] = None
    write_info_files: bool
    animation_mode: Union[str, None] = 'None'
    keyframes: Union[str, None] = ''
    max_frames: Union[int, None] = 1
    prompts: Union[str, None] = ''
    border: Union[str, None] = None
    angle: Union[str, None] = None
    zoom: Union[str, None] = None
    translation_x: Union[str, None] = None
    translation_y: Union[str, None] = None
    translation_z: Union[str, None] = None
    rotation_3d_x: Union[str, None] = None
    rotation_3d_y: Union[str, None] = None
    rotation_3d_z: Union[str, None] = None
    noise_schedule: Union[str, None] = None
    flip_2d_perspective: Union[bool, None] = None
    perspective_flip_theta: Union[str, None] = None
    perspective_flip_phi: Union[str, None] = None
    perspective_flip_gamma: Union[str, None] = None
    perspective_flip_fv: Union[str, None] = None
    strength_schedule: Union[str, None] = None
    contrast_schedule: Union[str, None] = None
    color_coherence: Union[str, None] = None
    diffusion_cadence: Union[str, None] = None
    use_depth_warping: Union[bool, None] = None
    midas_weight: Union[float, None] = None
    near_plane: Union[int, None] = None
    far_plane: Union[int, None] = None
    fov: Union[int, None] = None
    padding_mode: Union[str, None] = None
    save_depth_maps: Union[bool, None] = None
    video_init_path: Union[str, None] = None
    extract_nth_frame: Union[int, None] = None
    interpolate_key_frames: Union[bool, None] = None
    interpolate_x_frames: Union[int, None] = None
    resume_from_timestring: Union[bool, None] = None
    resume_timestring: Union[str, None] = None
    outdir: Union[str, None]
    n_samples: Union[int, None] = None
    ddim_eta: Union[float, None] = None
    save_samples: Union[bool, None] = None
    save_settings: Union[bool, None] = None
    display_samples: Union[bool, None] = None
    n_batch: Union[int, None] = None
    batch_name: Union[str, None] = None
    filename_format: Union[str, None] = None
    seed_behavior: Union[str, None] = None
    make_grid: Union[bool, None] = None
    grid_rows: Union[int, None] = None
    use_init: Union[bool, None] = None
    strength: Union[int, None] = None
    strength_0_no_init: Union[bool, None] = None
    init_image: Union[str, None] = None
    use_mask: Union[bool, None] = None
    use_alpha_as_mask: Union[bool, None] = None
    mask_file: Union[str, None] = None
    invert_mask: Union[bool, None] = None
    mask_brightness_adjust: Union[int, None] = None
    mask_contrast_adjust: Union[int, None] = None
    generation_mode: Union[str, None] = None
    batch_size: Union[int, None] = None


#    mask_file: Union[int, None]


def txt2img_json(t2i_json):


    if not t2i_json.seed :
        t2i_json.seed = random.randint(0, 2 ** 32 - 1)


    if t2i_json.seed == '':
        t2i_json.seed = random.randint(0, 2 ** 32 - 1)

    def_six_render.run_deforum_six()  # here all the args, or find an entry point where you just push the incoming json
                                      # you may use a callback here which will send back the image to here
                                      # so that you can send it back to the remote ui

    return {
         'image':  image #i dont know yet how the image gets here....
    }


@router.post('/api/v1/txttoimg/run', status_code=201)
async def post(t2i_json: Txt2Img):
    try:
        gs.current_images = []
        gs.rendering = True
        t2i_json.prompts = t2i_json.prompt
        return txt2img_json(t2i_json)    # this will return the image in a json object to the UI

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        message = 'txt2img error' + str(e)
        respond_500(message)


@router.get('/api/v1/txttoimg/get_results')
async def get():
    try:
        res = {
            'rendering': gs.rendering,
            'current_images': gs.current_images[::-1]
        }
        return res
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        message = 'txt2vid Error: ' + str(e)
        respond_500(message)
