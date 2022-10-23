import shutil

from fastapi import APIRouter, File, UploadFile, Form
import os
import random
from PIL.PngImagePlugin import PngInfo
from fastapi.responses import FileResponse
from backend.outpaint.outpaint import * # outpaint_txt2img
from typing import Union
from backend.singleton import singleton

gs = singleton
router = APIRouter()


tmp_directory = "./outputs/outpaint/"
global_prefix = 'outpaint'
output_directory = "./outputs/outpain_out"

current_id = 1

os.makedirs(tmp_directory, exist_ok=True)
os.makedirs(global_prefix, exist_ok=True)

def get_default_dict():
    return {
        "C" : 4,
        "f" : 8,
        "dyn" : None,
        "from_file": None,
        "n_rows" : 2,
        "plms" : False,
        "ddim_eta" : 0.0,
        "n_iter" : 1,
        "outdir" : output_directory,
        "skip_grid" : False,
        "skip_save" : True, #FIXME
        "fixed_code": False,
        "save_intermediate_every": 1000
    }

def save_upload_file(upload_file: UploadFile, destination) -> None:
    try:
        with open(destination,"wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()

@router.post('/api/v1/canvas/run', status_code=201)
# @api.param('body', 'The JSON Data',  consumes="multipart/form-data")
async def canvas_run(

        prompt: str = Form(),
        width: int = Form(512),
        height: int = Form(512),
        guidance: float = Form(7),
        steps: int = Form(50),
        samples: int = Form(1),
        imageGuide: Union[UploadFile, bytes]  = File(default=None),
        maskForBlend: Union[UploadFile, bytes]  = File(default=None),
        blend_mask: str = Form(None),
        strength: int = Form(None),
        seed: str = Form(None),
        maskBlur: int = Form()
):

    try:
        my_data = get_default_dict()

        #my_data['strength'] = 0.5  # why it is missing?

        if seed != b'undefined':
            seed = random.randrange(4200000000)

        my_data['file_prefix'] = f"{global_prefix}_{current_id}"

        my_data['prompt'] = prompt
        my_data['W'] = width
        my_data['H'] = height
        my_data['scale'] = guidance
        my_data['seed'] = seed
        my_data['steps'] = steps
        my_data['n_samples'] = samples
        my_data['blend_mask'] = None
        #my_data['return_changes_only'] = request.form.get('returnChangesOnly', default=False, type=bool)

        if imageGuide != b'undefined' and imageGuide.filename != '':
            path = os.path.join(tmp_directory, f"{global_prefix}_{current_id}.png")
            save_upload_file(imageGuide, path)
            my_data['image_guide'] = path

            if maskForBlend  != b'undefined' and maskForBlend.filename != '':
                mask_path = os.path.join(
                    tmp_directory, f"{global_prefix}_{current_id}_mask.png")
                save_upload_file(maskForBlend, mask_path)
                my_data['blend_mask'] = mask_path
                my_data['mask_blur'] = maskBlur
            else:
                my_data['strength'] = strength

        else:
            my_data['image_guide'] = False

        mdata = PngInfo()

        image = outpaint_txt2img(my_data)
        rnd = int(random.randrange(10000000000))
        filename = f"www/web/static/img/" + str(rnd) + "temp.png"

        image[0].save(filename, pnginfo=mdata)

        return FileResponse(filename)


    except Exception as e:
        print(e, flush=True)
        return {'success': False}, 500
