import gc
import sys
import traceback

import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

from backend.modelloader import load_upscaler

from PySide6.QtCore import QObject, QFile, Signal
from backend.singleton import singleton
gs = singleton


class Callbacks(QObject):
    upscale_counter = Signal(int)

class Upscale:

    def __init__(self, progress_callback=None):
        self.signals = Callbacks()


    def run_gfpgan(self, image, strength, seed):
        print(f'>> GFPGAN - Restoring Faces for image seed:{seed}')

        try:
            image = image.convert('RGB')

            cropped_faces, restored_faces, restored_img = gs.models["GFPGAN"].enhance(
                np.array(image, dtype=np.uint8),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            res = Image.fromarray(restored_img)

            if strength < 1.0:
                # Resize the image to the new image if the sizes have changed
                if restored_img.size != image.size:
                    image = image.resize(res.size)
                res = Image.blend(image, res, strength)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print('error gfpgan ', e)

        return res

    def real_esrgan_upscale(self, image, strength, upsampler_scale):
        print(
            f'>> Real-ESRGAN Upscaling scale:{upsampler_scale}x'
        )

        output, img_mode = gs.models["RealESRGAN"].enhance(
            np.array(image, dtype=np.uint8),
            outscale=upsampler_scale,
            alpha_upsampler='realesrgan',
        )

        res = Image.fromarray(output)

        if strength < 1.0:
            # Resize the image to the new image if the sizes have changed
            if output.size != image.size:
                image = image.resize(res.size)
            res = Image.blend(image, res, strength)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return res

    def torch_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def upscale_and_reconstruct(self,
                                image_list,
                                upscale       = False,
                                upscale_scale = 0 ,
                                upscale_strength = 0,
                                use_gfpgan    = False,
                                strength      = 0.0,
                                gfpgan_seed = 1,
                                image_callback = None):
        try:
            if upscale:
                from backend.gfpgan_tools import real_esrgan_upscale
            if strength > 0:
                from backend.gfpgan_tools import run_gfpgan
        except (ModuleNotFoundError, ImportError):
            print(traceback.format_exc(), file=sys.stderr)
            print('>> You may need to install the ESRGAN and/or GFPGAN modules')
            return

        file_count = 0;
        for path in image_list:

            image = Image.open(path)
            gfpgan_seed = seed_everything(gfpgan_seed)

            try:
                if use_gfpgan and strength > 0:
                    image = self.run_gfpgan(
                        image, strength, gfpgan_seed
                    )
                if upscale:
                    if upscale_strength == 0:
                        upscale_strength = 0.75
                    image = self.real_esrgan_upscale(
                        image,
                        upscale_strength,
                        int(upscale_scale)
                    )

            except Exception as e:
                print(
                    f'>> Error running RealESRGAN or GFPGAN. Your image was not upscaled.\n{e}'
                )

            outpath = path + '.enhanced.png'
            image = image.convert("RGBA")
            image.save(outpath)

            self.torch_gc()

            if image_callback is not None:
                image_callback(image, upscale_seed, upscaled=True)
            file_count += 1
            self.signals.upscale_counter.emit(file_count)

    def run_upscale(self,
                    filelist=[],
                    use_esrgan=False,
                    model_name='RealESRGAN_x4plus',
                    use_gfpgan=False,
                    esr_scale=1,
                    esr_strength=0,
                    gfp_strength=0,
                    progress_callback=None):


        load_upscaler(use_gfpgan, use_esrgan, model_name)
        if len(filelist) > 0:
            if use_esrgan or use_esrgan:
                print('upscaling')

                self.upscale_and_reconstruct(filelist,
                                             upscale          = use_esrgan,
                                             upscale_scale    = esr_scale,
                                             upscale_strength = esr_strength/100,
                                             use_gfpgan       = use_gfpgan,
                                             strength         = gfp_strength/100,
                                             image_callback   = None)
