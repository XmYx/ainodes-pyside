import os
import sys
import random
from typing import Union
from PIL import Image
from fastapi import APIRouter, BackgroundTasks
from fastapi.openapi.models import Response
from pydantic import BaseModel
from fastapi.responses import FileResponse
from zipfile import ZipFile
from backend.singleton import singleton
import backend.settings as settings

gs = singleton
settings.load_settings_json()

from backend.deforum.deforum_adapter import DeforumSix
from apis.v1.http.response import respond_500

router = APIRouter()

def_six_render = DeforumSix()


class Txt2Img(BaseModel):
    prompt: str
    W: int = 512
    H: int = 512
    scale: float = 7.5
    seed: Union[int, str, None] = random.randint(0, 2 ** 32 - 1)
    iterations: int = 1
    batch_size: int = 1
    steps: int = 25
    sampler: str = 'euler'
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
    prompts: str
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
    n_batch: Union[int, None] = None
    hires: bool = False
    cond_uncond_sync: bool = True


class image_response(Response):
    media_type = "image/png"


def txt2img_json(t2i_json):
    if not t2i_json.seed:
        t2i_json.seed = random.randint(0, 2 ** 32 - 1)

    if t2i_json.seed == '':
        t2i_json.seed = random.randint(0, 2 ** 32 - 1)
    params = t2i_json.__dict__
    print(t2i_json)
    paths = def_six_render.run_deforum_six(W=int(t2i_json.W),
                                           H=int(t2i_json.H),
                                           seed=int(t2i_json.seed),
                                           sampler=str(t2i_json.sampler),
                                           steps=int(t2i_json.steps),
                                           scale=float(t2i_json.scale),
                                           ddim_eta=float(t2i_json.ddim_eta),
                                           save_settings=bool(t2i_json.save_settings),
                                           save_samples=bool(t2i_json.save_samples),
                                           animation_mode='None',
                                           n_batch=int(t2i_json.n_batch),
                                           seed_behavior=t2i_json.seed_behavior,
                                           # make_grid=t2i_json.makegrid,
                                           # grid_rows=t2i_json.grid_rows,
                                           # use_init=t2i_json.use_init,
                                           # init_image=t2i_json.init_image,
                                           # strength=float(t2i_json.strength),
                                           # strength_0_no_init=t2i_json.strength_0_no_init,
                                           # device=t2i_json.device,
                                           # animation_mode=t2i_json.animation_mode,
                                           prompts=t2i_json.prompt,
                                           # max_frames=t2i_json.max_frames,
                                           outdir=t2i_json.outdir,
                                           n_samples=t2i_json.n_samples,
                                           # mean_scale=t2i_json.mean_scale,
                                           # var_scale=t2i_json.var_scale,
                                           # exposure_scale=t2i_json.exposure_scale,
                                           # exposure_target=t2i_json.exposure_target,
                                           # colormatch_scale=float(t2i_json.colormatch_scale),
                                           # colormatch_image=t2i_json.colormatch_image,
                                           # colormatch_n_colors=t2i_json.colormatch_n_colors,
                                           # ignore_sat_weight=t2i_json.ignore_sat_weight,
                                           # clip_name=t2i_json.clip_name,
                                           # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32
                                           # clip_scale=t2i_json.clip_scale,
                                           # aesthetics_scale=t2i_json.aesthetics_scale,
                                           # cutn=t2i_json.cutn,
                                           # cut_pow=t2i_json.cut_pow,
                                           # init_mse_scale=t2i_json.init_mse_scale,
                                           # init_mse_image=t2i_json.init_mse_image,
                                           # blue_scale=t2i_json.blue_scale,
                                           # gradient_wrt=t2i_json.gradient_wrt,  # ["x", "x0_pred"]
                                           # gradient_add_to=t2i_json.gradient_add_to,  # ["cond", "uncond", "both"]
                                           # decode_method=t2i_json.decode_method,  # ["autoencoder","linear"]
                                           # grad_threshold_type=t2i_json.grad_threshold_type,
                                           # ["dynamic", "static", "mean", "schedule"]
                                           # clamp_grad_threshold=t2i_json.clamp_grad_threshold,
                                           # clamp_start=t2i_json.clamp_start,
                                           # clamp_stop=t2i_json.clamp_stop,
                                           # grad_inject_timing=1,
                                           # if self.parent.unicontrol.w.grad_inject_timing.text() == '' else self.parent.unicontrol.w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                           cond_uncond_sync=t2i_json.cond_uncond_sync,
                                           # step_callback=None,
                                           # image_callback=None,
                                           # negative_prompts=t2i_json.negative_prompts if t2i_json.negative_prompts != False else None,
                                           hires=t2i_json.hires,
                                           # prompt_weighting=t2i_json.prompt_weighting,
                                           # normalize_prompt_weights=t2i_json.normalize_prompt_weights,
                                           # lowmem=False,
                                           )  # here all the args, or find an entry point where you just push the incoming json
    # you may use a callback here which will send back the image to here
    # so that you can send it back to the remote ui
    zipObj = ZipFile('return.zip', 'w')
    for i in paths:
        zipObj.write(i)
    zipObj.close()

    # image = paths[0]
    return 'return.zip'
    # return {
    #     'image':  image #i dont know yet how the image gets here....
    # }


@router.post('/api/v1/txttoimg/run',
             # Set what the media type will be in the autogenerated OpenAPI specification.
             # fastapi.tiangolo.com/advanced/additional-responses/#additional-media-types-for-the-main-response
             responses={
                 200: {
                     "content": {"image/png": {}}
                 }
             },

             # Prevent FastAPI from adding "application/json" as an additional
             # response media type in the autogenerated OpenAPI specification.
             # https://github.com/tiangolo/fastapi/issues/3258
             response_class=Response

             )
async def post(t2i_json: Txt2Img, background_tasks: BackgroundTasks):
    try:
        gs.current_images = []
        gs.rendering = True
        t2i_json.prompts = t2i_json.prompt
        filename = txt2img_json(t2i_json)
        return FileResponse(filename)  # this will return the image in a json object to the UI

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
