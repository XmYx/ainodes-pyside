"""This is an example plugin file that you can use to build your own plugins for aiNodes.

Welcome to aiNodes. Please refer to PySide 6.4 documentation for UI functions.

Please also note the following features at your disposal by default:
DeforumSix
Hypernetworks
Singleton

at plugin loading time, the plugins initme function will be called automatically to make sure that
all defaults are set correctly, and that your new UI element is loaded, with its signals and slots connected.

Your plugin's parent is the MainWindow, and by default, it has a canvas loaded. You can access all of its functions,
such as addrect_atpos, and image_preview_func (make sure to set self.parent.image before doing so).

It is good to know, that if you are doing heavy lifting, you have to use its own QThreadPool, otherwise your gui freezes
while processing. To do so, just use the worker from backend.worker

        worker = Worker(self.parent.deforum_ui.run_deforum_six_txt2img)
        self.parent.threadpool.start(worker)

It is also worth mentioning, that ui should only be modified from the main thread, therefore when displaying an image,
set self.parent.image, then call self.parent.image_preview_signal, which will emit a signal to call
the image_preview_func from the main thread.
"""

import os
import zipfile
import time
import random
import json
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PySide6 import QtCore, QtNetwork
from PySide6.QtCore import QObject, Signal, QJsonDocument, Slot, QFile, QIODevice
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QMainWindow, QLineEdit, QFrame, QWidget, QHBoxLayout

import frontend.ui_deforum
from backend.singleton import singleton
import torchvision.transforms as T
from torchvision.utils import make_grid
from einops import rearrange
from fonts.ttf import Roboto
from backend.worker import Worker
from frontend.session_params import translate_sampler
from frontend import ui_model_chooser
gs = singleton

class aiNodesPlugin():
    def __init__(self, parent):
        self.parent = parent

    def initme(self):
        print("Using API")
        self.use_api_processing()

    def use_api_processing(self):
        #self.parent.ui_deforum = None
        try:
            self.parent.unicontrol.w.dream.clicked.disconnect()
        except:
            pass
        frontend.ui_deforum.Deforum_UI = DeforumAPI
        self.parent.ui_deforum = DeforumAPI(self.parent)
        self.parent.unicontrol.w.dream.clicked.connect(self.parent.ui_deforum.run_deforum_six_txt2img)
        self.widget = QWidget()
        self.parent.urledit = QLineEdit()
        self.layout = QHBoxLayout(self.widget)
        self.layout.addWidget(self.parent.urledit)
        ui_model_chooser.ModelChooser_UI.set_model = self.parent.ui_deforum.set_model
        try:
            self.parent.path_setup.w.activateModel.disconnect()
        except:
            pass
        self.parent.path_setup.w.activateModel.clicked.connect(self.parent.ui_deforum.set_model)
        #self.parent.path_setup.w.reloadModelList.clicked.connect(self.load_folder_content)

        self.widget.show()
        #self.parent.ui_deforum = Deforum_UI(self.parent)



class Callbacks(QObject):
    txt2img_step = Signal()
    reenable_runbutton = Signal()
    txt2img_image_cb = Signal()
    deforum_step = Signal()
    deforum_image_cb = Signal()
    compviscallback = Signal()
    add_image_to_thumbnail_signal = Signal(str)
    setStatusBar = Signal(str)
    vid2vid_one_percent = Signal(int)
    prepare_hires_batch = Signal(str)

class DeforumAPI(QObject):
    def __init__(self, parent):
        # super(QObject, self).__init__()
        self.renderedFrames = None
        self.currentFrames = None
        self.onePercent = None
        self.updateRate = None
        self.update = None
        self.progress = None
        self.deforum = None
        self.parent = parent
        # self.deforum = DeforumGenerator()
        self.signals = Callbacks()
        # self.deforum_six = DeforumSix()


    def run(self):
        params = self.parent.sessionparams.update_params()
        print(f"updated kutya to: {params}")
        self.deforum_six.run_deforum_six(W=int(params['W']),
                                         H=int(params['H']),
                                         seed=int(params['seed']) if params['seed'] != '' else seed,
                                         sampler=str(params['sampler']),
                                         steps=int(params['steps']),
                                         scale=float(params['scale']),
                                         ddim_eta=float(params['ddim_eta']),
                                         save_settings=bool(params['save_settings']),
                                         save_samples=bool(params['save_samples']),
                                         show_sample_per_step=bool(params['show_sample_per_step']),
                                         n_batch=int(params['n_batch']),
                                         seed_behavior=params['seed_behavior'],
                                         make_grid=params['makegrid'],
                                         grid_rows=params['grid_rows'],
                                         use_init=params['use_init'],
                                         init_image=params['init_image'],
                                         strength=float(params['strength']),
                                         strength_0_no_init=params['strength_0_no_init'],
                                         device=params['device'],
                                         animation_mode=params['animation_mode'],
                                         prompts=params['prompts'],
                                         max_frames=params['max_frames'],
                                         outdir=params['outdir'],
                                         n_samples=params['n_samples'],
                                         mean_scale=params['mean_scale'],
                                         var_scale=params['var_scale'],
                                         exposure_scale=params['exposure_scale'],
                                         exposure_target=params['exposure_target'],
                                         colormatch_scale=float(params['colormatch_scale']),
                                         colormatch_image=params['colormatch_image'],
                                         colormatch_n_colors=params['colormatch_n_colors'],
                                         ignore_sat_weight=params['ignore_sat_weight'],
                                         clip_name=params['clip_name'],
                                         # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                                         clip_scale=params['clip_scale'],
                                         aesthetics_scale=params['aesthetics_scale'],
                                         cutn=params['cutn'],
                                         cut_pow=params['cut_pow'],
                                         init_mse_scale=params['init_mse_scale'],
                                         init_mse_image=params['init_mse_image'],
                                         blue_scale=params['blue_scale'],
                                         gradient_wrt=params['gradient_wrt'],  # ["x", "x0_pred"]
                                         gradient_add_to=params['gradient_add_to'],
                                         # ["cond", "uncond", "both"]
                                         decode_method=params['decode_method'],  # ["autoencoder","linear"]
                                         grad_threshold_type=params['grad_threshold_type'],
                                         # ["dynamic", "static", "mean", "schedule"]
                                         clamp_grad_threshold=params['clamp_grad_threshold'],
                                         clamp_start=params['clamp_start'],
                                         clamp_stop=params['clamp_stop'],
                                         grad_inject_timing=1,
                                         # if self.parent.unicontrol.w.grad_inject_timing.text() == '' else self.parent.unicontrol.w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                         cond_uncond_sync=params['cond_uncond_sync'],
                                         step_callback=self.parent.tensor_preview_signal if self.parent.unicontrol.w.show_sample_per_step.isChecked() else None,
                                         image_callback=self.parent.image_preview_signal,
                                         negative_prompts=params['negative_prompts'] if params[
                                                                                            'negative_prompts'] != False else None,
                                         hires=params['hires'],
                                         prompt_weighting=params['prompt_weighting'],
                                         normalize_prompt_weights=params['normalize_prompt_weights'],
                                         lowmem=params['lowmem'],
                                         )

    def run_deforum_six_txt2img(self, progress_callback=None, plotting=True):

        params = self.parent.sessionparams.update_params()
        print(f"updated tyutya to: {params}")
        if "inpaint" in gs.models:
            del gs.models["inpaint"]
        #if params.with_inpaint == True:
        #    self.parent.params.advanced = True
        #else:
        #    self.parent.params.advanced = False
        seed = random.randint(0, 2 ** 32 - 1)
        # print('strength ui', float(params['strength']))

        plotting = self.parent.unicontrol.w.plotting.isChecked()
        print('plotting', plotting)
        # plotting = None
        if plotting:

            attrib2 = self.parent.unicontrol.w.plotX.currentText()
            attrib1 = self.parent.unicontrol.w.plotY.currentText()

            ploty_list_string = self.parent.unicontrol.w.plotXLine.text()
            plotx_list_string = self.parent.unicontrol.w.plotYLine.text()
            plotY = plotx_list_string.split(', ')
            plotX = ploty_list_string.split(', ')
            self.onePercent = 100 / (
                        len(plotX) * len(plotY) * params.n_batch * params.n_samples * params.steps)
            # print(self.onePercent)

        else:
            plotX = [1]
            plotY = [1]
            self.onePercent = 100 / (params.n_batch * params.n_samples * params.steps)
        # print(plotY, plotX)
        all_images = []
        # print(f"Grid Dimensions: {len(plotX)}, {len(plotY)}")
        # print(self.onePercent)
        # print(params)
        self.parent.w = params.W
        for i in plotY:
            for j in plotX:
                if plotting:
                    params.__dict__[attrib1] = i
                    params.__dict__[attrib2] = j
                    if attrib1 == 'T': gs.T = int(i)
                    if attrib1 == 'lr': gs.lr = float(i)
                    if attrib2 == 'T': gs.T = int(j)
                    if attrib2 == 'lr': gs.lr = float(j)
                print("PARAMS BELOW")
                params = params.__dict__
                self.url = QtCore.QUrl(f"{self.parent.urledit.text()}/api/v1/txttoimg/run")
                # self.url = QtCore.QUrl("https://www.google.com/")
                #params = {}
                print(params['prompts'])
                #params['prompts'] = "corgi"
                params['prompt'] = params['prompts']

                #params['prompts'] = list(params['prompts'])
                params['makegrid'] = False
                params['iterations'] = 1
                params['separate_prompts'] = False
                params['save_individual_images'] = True
                params['save_grid'] = False
                params['group_by_prompt'] = False
                params['save_as_jpg'] = False
                params['use_gfpgan'] = False
                params['use_realesrgan'] = False
                params['realesrgan_model'] = ""
                params['realesrgan_model_name'] = ""
                params['variant_amount'] = 0
                params['write_info_files'] = False
                params['karras'] = self.parent.unicontrol.w.karras.isChecked()
                params['sampler'] = translate_sampler(params['sampler'])
                print(params['sampler'])

                self.manager = QtNetwork.QNetworkAccessManager()
                self.manager.finished.connect(self.handleResponse)
                self.request = QtNetwork.QNetworkRequest()
                self.request.setUrl(self.url)
                self.request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.ContentTypeHeader,
                                       "application/json")
                obj = QJsonDocument(params)
                self.data = QtCore.QByteArray(obj.toJson())
                self.manager.post(self.request, self.data)

                # self.manager.get(self.request)
                # self.response.finished.connect(self.handleResponse)

                """self.sendurl = QtCore.QUrl("http://www.google.com")
                self.rdata = params
                self.rdata = json.dumps(self.rdata)

                self.request = QtNetwork.QNetworkRequest()
                self.manager = QtNetwork.QNetworkAccessManager()
                self.request.setUrl(self.sendurl)
                self.request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.ContentTypeHeader, 'application/json')
                self.data = bytes(self.rdata, 'UTF-8')
                #self.data = QtCore.QByteArray(self.rdata)

                self.buffer = QtCore.QBuffer()

                #self.buffer.open(QtCore.QBuffer.ReadWrite)

                #self.buffer.writeData(self.data, len(self.data))
                self.buffer.seek(0)

                self.patchbytes = bytes('PATCH', 'UTF-8')
                self.patchverb = QtCore.QByteArray(self.patchbytes)
                self.response = QtCore.QByteArray()
                self.response = self.manager.sendCustomRequest(self.request, self.patchverb, self.buffer)

                self.response = self.response.readAll().data().decode('utf-8')
                self.response = str(self.response)
                print(self.response)"""
                """self.deforum_six.run_deforum_six(W=int(params['W']),
                                                 H=int(params['H']),
                                                 seed=int(params['seed']) if params['seed'] != '' else seed,
                                                 sampler=str(params['sampler']),
                                                 steps=int(params['steps']),
                                                 scale=float(params['scale']),
                                                 ddim_eta=float(params['ddim_eta']),
                                                 save_settings=bool(params['save_settings']),
                                                 save_samples=bool(params['save_samples']),
                                                 show_sample_per_step=bool(params['show_sample_per_step']),
                                                 n_batch=int(params['n_batch']),
                                                 seed_behavior=params['seed_behavior'],
                                                 make_grid=params['makegrid'],
                                                 grid_rows=params['grid_rows'],
                                                 use_init=params['use_init'],
                                                 init_image=params['init_image'],
                                                 strength=float(params['strength']),
                                                 strength_0_no_init=params['strength_0_no_init'],
                                                 device=params['device'],
                                                 animation_mode=params['animation_mode'],
                                                 prompts=params['prompts'],
                                                 max_frames=params['max_frames'],
                                                 outdir=params['outdir'],
                                                 n_samples=params['n_samples'],
                                                 mean_scale=params['mean_scale'],
                                                 var_scale=params['var_scale'],
                                                 exposure_scale=params['exposure_scale'],
                                                 exposure_target=params['exposure_target'],
                                                 colormatch_scale=float(params['colormatch_scale']),
                                                 colormatch_image=params['colormatch_image'],
                                                 colormatch_n_colors=params['colormatch_n_colors'],
                                                 ignore_sat_weight=params['ignore_sat_weight'],
                                                 clip_name=params['clip_name'],
                                                 # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
                                                 clip_scale=params['clip_scale'],
                                                 aesthetics_scale=params['aesthetics_scale'],
                                                 cutn=params['cutn'],
                                                 cut_pow=params['cut_pow'],
                                                 init_mse_scale=params['init_mse_scale'],
                                                 init_mse_image=params['init_mse_image'],
                                                 blue_scale=params['blue_scale'],
                                                 gradient_wrt=params['gradient_wrt'],  # ["x", "x0_pred"]
                                                 gradient_add_to=params['gradient_add_to'],  # ["cond", "uncond", "both"]
                                                 decode_method=params['decode_method'],  # ["autoencoder","linear"]
                                                 grad_threshold_type=params['grad_threshold_type'],
                                                 # ["dynamic", "static", "mean", "schedule"]
                                                 clamp_grad_threshold=params['clamp_grad_threshold'],
                                                 clamp_start=params['clamp_start'],
                                                 clamp_stop=params['clamp_stop'],
                                                 grad_inject_timing=1,
                                                 # if self.parent.unicontrol.w.grad_inject_timing.text() == '' else self.parent.unicontrol.w.grad_inject_timing.text(), #it is a float an int or a list of floats
                                                 cond_uncond_sync=params['cond_uncond_sync'],
                                                 step_callback=self.parent.tensor_preview_signal if self.parent.unicontrol.w.show_sample_per_step.isChecked() else None,
                                                 image_callback=self.parent.image_preview_signal,
                                                 negative_prompts=params['negative_prompts'] if params['negative_prompts'] != False else None,
                                                 hires=params['hires'],
                                                 prompt_weighting=params['prompt_weighting'],
                                                 normalize_prompt_weights=params['normalize_prompt_weights'],
                                                 lowmem=params['lowmem'],
                                                 )"""
                if plotting:
                    all_images.append(T.functional.pil_to_tensor(self.parent.image))
        if plotting:
            ver_texts = []
            hor_texts = []
            for i in plotY:
                ver_texts.append([GridAnnotation(f"{attrib1}: {i}")])
            for j in plotX:
                hor_texts.append([GridAnnotation(f"{attrib2}: {j}")])
            print(hor_texts)
            grid = make_grid(all_images, nrow=len(plotX))
            grid = rearrange(grid, 'c h w -> h w c').cpu().numpy()
            filename = f"{time.strftime('%Y%m%d%H%M%S')}_{attrib1}_{attrib2}_grid_{params['seed']}.png"
            grid_image = Image.fromarray(grid.astype(np.uint8))

            grid_image = draw_grid_annotations(grid_image, grid_image.size[0], grid_image.size[1], hor_texts,
                                               ver_texts, params['W'],
                                               params['H'], params)
            self.parent.image = grid_image
            self.parent.image_preview_signal(grid_image)
            grid_image.save(os.path.join(params['outdir'], filename))
        # self.signals.reenable_runbutton.emit()
        self.deforum_six = None
        return

    @Slot()
    def handleResponse(self, response):
        bytes_string = response.readAll()
        print(type(bytes_string))
        file = QFile("response.zip")
        file.open(QIODevice.WriteOnly)
        file.write(bytes_string)
        file.close()
        outdir = os.path.join(gs.system.outdir, f'response_{time.strftime("%Y%m%d%H%M%S")}')
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile('response.zip', 'r') as zip_ref:
            zip_ref.extractall(outdir)
        for root, dirs, files in os.walk(outdir):
            for filename in files:
                filename = os.path.join(root, filename)
                if os.path.isfile(filename):
                    image = Image.open(filename)
                    self.parent.image_preview_signal(image)


        #img = QImage()
        #img.loadFromData(bytes_string)
        #pixmap = QPixmap.fromImage(img)
        #pixmap.save("test.png")
        #image = Image.open("test.png")
        #self.parent.image_preview_signal(image)
        del response
        return
    def run_deforum_outpaint(self, params=None, progress_callback=None):
        # self.deforum = DeforumGenerator()
        # self.deforum.signals = Callbacks()

        self.deforum_six = DeforumSix()
        self.progress = 0.0
        self.parent.update = 0
        self.onePercent = 100 / self.parent.unicontrol.w.steps_slider.value()
        # self.updateRate = self.parent.sizer_count.w.previewSlider.value()
        self.updateRate = 1
        self.parent.currentFrames = []
        self.parent.renderedFrames = 0
        self.parent.sample_number = 1
        if self.parent.unicontrol.w.n_samples.value() == 1:
            makegrid = False
        else:
            makegrid = self.parent.animKeys.w.makeGrid.isChecked()
        # sampler_name = translate_sampler(self.parent.sampler.w.sampler.currentText())
        sampler_name = "ddim"
        init_image = "outpaint.png"
        gs.T = self.parent.unicontrol.w.gradient_steps.value()
        gs.lr = self.parent.unicontrol.w.gradient_scale.value() / 1000000000
        gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients,
                                                   self.parent.unicontrol.w.aesthetic_embedding.currentText())
        if params == None:
            params = self.parent.params

        if params is not None:
            # print(params)
            steps = int(params['steps'])
            H = int(params['H'])
            W = int(params['W'])
            seed = int(params['seed']) if params['seed'] != "" else random.randint(0, 44444444)
            prompt = str(params['prompts'])
            strength = float(params['strength'])
            mask_blur = float(params['mask_blur'])
            reconstruction_blur = float(params['reconstruction_blur'])
            scale = float(params['scale'])
            ddim_eta = float(params['ddim_eta'])
            with_inpaint = bool(params['use_inpaint'])

        self.parent.params['advanced'] = True

        self.deforum_six.outpaint_txt2img(init_image=init_image,
                                          steps=steps,
                                          H=H,
                                          W=W,
                                          seed=seed,
                                          prompt=prompt,
                                          strength=strength,
                                          mask_blur=mask_blur,
                                          recons_blur=reconstruction_blur,
                                          scale=scale,
                                          ddim_eta=ddim_eta,
                                          image_callback=self.parent.image_preview_signal,
                                          step_callback=self.parent.tensor_preview_signal,
                                          with_inpaint=with_inpaint)

        # self.run_txt2img_lm(init_img=init_image, init_mask='outpaint_mask.png')

        self.signals.reenable_runbutton.emit()

    def deforum_outpaint_thread(self):

        self.parent.params = self.parent.sessionparams.update_params()
        self.choice = "Outpaint"
        worker = Worker(self.run_deforum_outpaint)
        self.parent.threadpool.start(worker)
    def set_model(self):
        self.url = QtCore.QUrl(f"{self.parent.urledit.text()}/api/v1/txttoimg/change_model")
        print(os.path.join(gs.system.customModels, self.parent.path_setup.w.modelList.currentText()))
        params = {
            "ckpt": str(self.parent.path_setup.w.modelList.currentText())
        }
        self.manager = QtNetwork.QNetworkAccessManager()
        #self.manager.finished.connect(self.handleResponse)
        self.request = QtNetwork.QNetworkRequest()
        self.request.setUrl(self.url)
        self.request.setHeader(QtNetwork.QNetworkRequest.KnownHeaders.ContentTypeHeader,
                               "application/json")

        obj = QJsonDocument(params)
        self.data = QtCore.QByteArray(obj.toJson())
        self.manager.post(self.request, self.data)


class GridAnnotation:
    def __init__(self, text='', is_active=True):
        self.text = text
        self.is_active = is_active
        self.size = None

def draw_grid_annotations(im, width, height, hor_texts, ver_texts, W, H, params):
    def wrap(drawing, text, font, line_length):
        lines = ['']
        for word in text.split():
            line = f'{lines[-1]} {word}'.strip()
            if drawing.textlength(line, font=font) <= line_length:
                lines[-1] = line
            else:
                lines.append(word)
        return lines

    def draw_texts(drawing, draw_x, draw_y, lines):
        for i, line in enumerate(lines):
            drawing.multiline_text((draw_x, draw_y + line.size[1] / 2), line.text, font=fnt,
                                   fill=color_active if line.is_active else color_inactive, anchor="mm",
                                   align="center")

            if not line.is_active:
                drawing.line((
                             draw_x - line.size[0] // 2, draw_y + line.size[1] // 2, draw_x + line.size[0] // 2,
                             draw_y + line.size[1] // 2), fill=color_inactive, width=4)

            draw_y += line.size[1] + line_spacing

    fontsize = (W + H) // 100
    line_spacing = fontsize // 2

    try:
        fnt = ImageFont.truetype(Roboto, fontsize)
    except Exception:
        fnt = ImageFont.truetype(Roboto, fontsize)

    color_active = (0, 0, 0)
    color_inactive = (153, 153, 153)

    pad_left = 0 if sum([sum([len(line.text) for line in lines]) for lines in ver_texts]) == 0 else W // 4

    cols = im.width // W
    rows = im.height // H

    print(f"DEBUG: {cols}, {rows}, of which at least one should be more then 1...")

    assert cols == len(hor_texts), f'bad number of horizontal texts: {len(hor_texts)}; must be {cols}'
    assert rows == len(ver_texts), f'bad number of vertical texts: {len(ver_texts)}; must be {rows}'

    calc_img = Image.new("RGB", (1, 1), "white")
    calc_d = ImageDraw.Draw(calc_img)

    for texts, allowed_width in zip(hor_texts + ver_texts, [W] * len(hor_texts) + [pad_left] * len(ver_texts)):
        items = [] + texts
        texts.clear()

        for line in items:
            wrapped = wrap(calc_d, line.text, fnt, allowed_width)
            texts += [GridAnnotation(x, line.is_active) for x in wrapped]

        for line in texts:
            bbox = calc_d.multiline_textbbox((0, 0), line.text, font=fnt)
            line.size = (bbox[2] - bbox[0], bbox[3] - bbox[1])

    hor_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing for lines in
                        hor_texts]
    ver_text_heights = [sum([line.size[1] + line_spacing for line in lines]) - line_spacing * len(lines) for
                        lines in
                        ver_texts]

    pad_top = max(hor_text_heights) + line_spacing * 2

    result = Image.new("RGB", (im.width + pad_left, im.height + pad_top), "white")
    result.paste(im, (pad_left, pad_top))

    d = ImageDraw.Draw(result)
    # p_pad = len(params["prompts"][0]) * 1.75
    # d.multiline_text(((pad_left / 2) + p_pad, pad_top / 2), params["prompts"][0], font=fnt, fill=color_active if line.is_active else color_inactive, anchor="mm", align="left")

    for col in range(cols):
        x = pad_left + W * col + W / 2
        y = pad_top / 2 - hor_text_heights[col] / 2

        draw_texts(d, x, y, hor_texts[col])

    for row in range(rows):
        x = pad_left / 2
        y = pad_top + H * row + H / 2 - ver_text_heights[row] / 2

        draw_texts(d, x, y, ver_texts[row])

    return result


