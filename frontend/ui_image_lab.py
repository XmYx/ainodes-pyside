import gc
import os
import sys
import traceback

import cv2
import numpy as np
import torch
from PySide6 import QtUiTools, QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, QFile, Signal

from backend.modelloader import load_upscaler
from backend.singleton import singleton
from PIL import Image

gs = singleton


class ImageLab_ui(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/img_lab.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class DropListView(QtWidgets.QListWidget):

    fileDropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(DropListView, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.fileDropped.emit(links)
        else:
            event.ignore()

class Callbacks(QObject):
    upscale_start = Signal()
    upscale_stop = Signal()
    upscale_counter = Signal(int)


class ImageLab():  # for signaling, could be a QWidget  too

    def __init__(self):
        super().__init__()
        self.signals = Callbacks()
        self.imageLab = ImageLab_ui()
        self.dropWidget = DropListView()
        self.dropWidget.setAccessibleName('fileList')
        self.dropWidget.fileDropped.connect(self.pictureDropped)
        self.imageLab.w.dropZone.addWidget(self.dropWidget)
        self.imageLab.w.startUpscale.clicked.connect(self.run_upscale)

    def show(self):
        self.imageLab.w.show()

    def pictureDropped(self, l):
        self.fileList = []
        self.dropWidget.clear()
        for path in l:
            if os.path.isdir(path):
                result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']
                for file in result:
                    self.fileList.append(file)
                    self.dropWidget.insertItem(0, file)
            elif os.path.isfile(path):
                self.dropWidget.insertItem(0, path)
                self.fileList.append(path)
        self.imageLab.w.filesCount.display(str(len(self.fileList)))

    def run_gfpgan(self, image, strength, seed, upsampler_scale=4):
        print(f'>> GFPGAN - Restoring Faces for image seed:{seed}')

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

        return res

    def real_esrgan_upscale(self, image, strength, upsampler_scale, seed):
        print(
            f'>> Real-ESRGAN Upscaling seed:{seed} : scale:{upsampler_scale}x'
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
                                upscale_strength= 0,
                                use_gfpgan    = False,
                                strength      = 0.0,
                                image_callback = None):
        try:
            if upscale:
                from ldm.gfpgan.gfpgan_tools import real_esrgan_upscale
            if strength > 0:
                from ldm.gfpgan.gfpgan_tools import run_gfpgan
        except (ModuleNotFoundError, ImportError):
            print(traceback.format_exc(), file=sys.stderr)
            print('>> You may need to install the ESRGAN and/or GFPGAN modules')
            return

        for path in image_list:

            #image = cv2.imread(path)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.open(path)
            seed = 1
            try:
                if upscale:
                    if upscale_strength == 0:
                        upscale_strength = 0.75
                    image = self.real_esrgan_upscale(
                        image,
                        upscale_strength,
                        int(upscale_scale),
                        seed,
                    )
                if use_gfpgan and strength > 0:
                    image = self.run_gfpgan(
                        image, strength, seed, 1
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
                image_callback(image, seed, upscaled=True)

    def run_upscale(self):
        model_name=''
        if self.imageLab.w.ESRGAN.isChecked():
            if self.imageLab.w.RealESRGAN.isChecked():
                model_name = 'RealESRGAN_x4plus'
            if self.imageLab.w.RealESRGANAnime.isChecked():
                model_name = 'RealESRGAN_x4plus_anime_6B'

            load_upscaler(self.imageLab.w.GFPGAN.isChecked(), self.imageLab.w.ESRGAN.isChecked(), model_name)
        if len(self.fileList) > 0:
            if self.imageLab.w.ESRGAN.isChecked() or self.imageLab.w.GFPGAN.isChecked():
                print('upscaling')

                self.upscale_and_reconstruct(self.fileList,
                                             upscale          = self.imageLab.w.ESRGAN.isChecked(),
                                             upscale_scale    = self.imageLab.w.esrScale.value(),
                                             upscale_strength = self.imageLab.w.esrStrength.value()/100,
                                             use_gfpgan       = self.imageLab.w.GFPGAN.isChecked(),
                                             strength         = self.imageLab.w.gfpStrength.value()/100,
                                             image_callback   = None)

        """
                if use_RealESRGAN:
            if "RealESRGAN" in st.session_state and st.session_state["RealESRGAN"].model.name == RealESRGAN_model:
                print("RealESRGAN already loaded")
            else:
                # Load RealESRGAN
                try:
                    # We first remove the variable in case it has something there,
                    # some errors can load the model incorrectly and leave things in memory.
                    del st.session_state["RealESRGAN"]
                except KeyError:
                    pass

                if os.path.exists(st.session_state['defaults'].general.RealESRGAN_dir):
                    # st.session_state is used for keeping the models in memory across multiple pages or runs.
                    st.session_state["RealESRGAN"] = load_RealESRGAN(RealESRGAN_model)
                    print("Loaded RealESRGAN with model " + st.session_state["RealESRGAN"].model.name)

                use_RealESRGAN and st.session_state["RealESRGAN"] is not None and not use_GFPGAN:
                st.session_state["progress_bar_text"].text(
                    "Running RealESRGAN on image %d of %d..." % (i + 1, len(x_samples_ddim)))
                # skip_save = True # #287 >_>
                torch_gc()

                if st.session_state["RealESRGAN"].model.name != realesrgan_model_name:
                    # try_loading_RealESRGAN(realesrgan_model_name)
                    load_models(use_GFPGAN=use_GFPGAN, use_RealESRGAN=use_RealESRGAN,
                                RealESRGAN_model=realesrgan_model_name)

                output, img_mode = st.session_state["RealESRGAN"].enhance(x_sample[:, :, ::-1])
                esrgan_filename = original_filename + '-esrgan4x'
                esrgan_sample = output[:, :, ::-1]
                esrgan_image = Image.fromarray(esrgan_sample)
        """
