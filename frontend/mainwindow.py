import pandas as pd
from PySide6.QtGui import QBitmap

import backend.settings as settings
from backend.singleton import singleton

import importlib
# from memory_profiler import profile

settings.load_settings_json()

gs = singleton
import requests
import urllib
import json
from types import SimpleNamespace
import random
import time, sys, gc
import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtWidgets import QProgressBar
from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QListWidgetItem
from einops import rearrange

import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertTokenizerFast

import backend.settings as settings

from backend.deforum.deforum_simplified import DeforumGenerator
from backend.ui_func import getLatestGeneratedImagesFromPath
from backend.worker import Worker
from backend.prompt_ai.prompt_gen import generate_prompt
from frontend.ui_classes import *
from frontend.nodeeditor import *
from frontend.ui_image_lab import ImageLab
from frontend.ui_outpaint import OutpaintUI
from frontend.ui_timeline import Timeline, KeyFrame
from frontend import paintwindow_func
from frontend.ui_camera_controller import Window
from ldm.generate import Generate

from PySide6.QtCore import *

from datetime import datetime
from uuid import uuid4


class Callbacks(QObject):
    txt2img_step = Signal()
    reenable_runbutton = Signal()
    txt2img_image_cb = Signal()
    deforum_step = Signal()
    deforum_image_cb = Signal()
    compviscallback = Signal()
    add_image_to_thumbnail_signal = Signal(str)


class GenerateWindow(QObject):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/main/main_window.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()

    def __init__(self, *args, **kwargs):
        super(GenerateWindow, self).__init__(*args, **kwargs)
        self.deforum = None
        self.previewpos = "outerdock"
        self.path_setup = None
        self.ipixmap = None
        self.painter = None
        self.data2 = None
        self.data = None
        self.image = None
        self.threadpool = None
        self.timeline = None
        self.animSliders = None
        self.animKeys = None
        self.animKeyEditor = None
        self.vpainter = None
        self.newPixmap = None
        self.livePainter = None
        self.tpixmap = None
        self.value = None
        self.selection = None
        self.choice = None
        self.renderedFrames = None
        self.currentFrames = None
        self.updateRate = None
        self.onePercent = None
        self.steps = None
        self.update = None
        self.progress = None
        self.ftimer = QTimer(self)
        self.signals = Callbacks()
        self.image_lab = ImageLab()

        # self.kf = Keyframes()

        settings.load_settings_json()
        self.videoPreview = False
        self.image_path = ""
        self.gr = Generate(
            weights=gs.system.sdPath,
            config=gs.system.sdInference, )
        gs.album = {}
        gs.models = {}
        gs.result = ""
        gs.album = getLatestGeneratedImagesFromPath()
        self.now = 0

        self.home()

        self.signals.reenable_runbutton.connect(self.reenableRunButton)
        self.signals.txt2img_image_cb.connect(self.imageCallback_func)
        self.signals.deforum_step.connect(self.deforumstepCallback_func)
        self.signals.deforum_image_cb.connect(self.add_image_to_thumbnail)
        self.signals.compviscallback.connect(self.deforumTest)
        self.signals.add_image_to_thumbnail_signal.connect(self.add_image_to_thumbnail)

        # self.w.thumbnails.thumbs.installEventFilter(self)
        self.w.statusBar().showMessage('Ready')
        self.w.progressBar = QProgressBar()

        self.w.statusBar().addPermanentWidget(self.w.progressBar)

        # This is simply to show the bar
        self.w.progressBar.setGeometry(30, 40, 200, 25)
        self.w.progressBar.setValue(0)

        # self.nodeWindow = NodeWindow()
        self.load_history()

        self.w.actionAnim.triggered.connect(self.show_anim)
        self.w.actionPreview.triggered.connect(self.show_preview)
        self.w.actionPrompt.triggered.connect(self.show_prompt)
        self.w.actionSampler.triggered.connect(self.show_sampler)
        self.w.actionSliders.triggered.connect(self.show_sizer_count)
        self.w.actionThumbnails.triggered.connect(self.show_thumbnails)
        self.w.actionSave_System_Settings.triggered.connect(self.save_system_settings)
        self.w.actionSave_Diffusion_Settings.triggered.connect(self.save_diffusion_settings)
        self.w.actionLoad_Default_Settings.triggered.connect(self.load_default_diffusion_settings)
        self.w.actionRestart.triggered.connect(self.restart)
        self.w.actionImageLab.triggered.connect(self.show_image_lab)

        self.w.actionOutpaint.triggered.connect(self.show_paint)

        self.animKeyEditor.w.comboBox.currentTextChanged.connect(self.showTypeKeyframes)

    # INIT UI, AND PRELOAD FUNCTIONS
    def home(self):
        self.w.thumbnails = Thumbnails()
        self.threadpool = QThreadPool()
        self.w.preview = Preview()
        self.w.sizer_count = SizerCount()
        self.w.sampler = Sampler()
        self.w.anim = Anim()
        self.w.prompt = Prompt()
        self.w.dynaview = Dynaview()
        self.timeline = Timeline()
        self.animSliders = AnimSliders()
        self.animKeys = AnimKeys()
        self.animKeyEditor = AnimKeyEditor()
        self.path_setup = PathSetup()
        self.nodeWindow = NodeWindow()
        self.prompt_fetcher = FetchPrompts()
        self.dynaimage = Dynaimage()
        self.camera = Window()
        self.outpaint = OutpaintUI()
        self.compass = Compass()

        # self.pw = paintwindow_func.MainWindow()
        # self.outpaint.show()
        # self.camera.show()
        # self.w.setCentralWidget(self.nodeWindow)
        self.w.setCentralWidget(self.outpaint)
        # self.pw.show()
        # self.outpaint.update()
        # self.nodeWindow.addDockWidget(Qt.RightDockWidgetArea, self.dynaimage.w.dockWidget)
        # self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.outpaint)

        widget = QWidget.createWindowContainer(self.camera)
        widget.setMaximumSize(200, 200)
        widget.setMinimumSize(100, 100)

        widget.mouseMoveEvent

        self.compass.w.camlayout.addWidget(widget)

        # self.w.setCentralWidget(self.nodeWindow)
        self.nodeWindow.addDockWidget(Qt.RightDockWidgetArea, self.dynaimage.w.dockWidget)

        self.timeline.timeline.keyFramesUpdated.connect(self.updateKeyFramesFromTemp)
        self.timeline.timeline.selectedValueType = self.animKeyEditor.w.comboBox.currentText()  # self.nodes = NodeEditorWindow()
        # self.nodes.nodeeditor.addNodes()
        self.timeline.timeline.update()

        self.dynaimage.w.prevFrame.clicked.connect(self.prevFrame)
        self.dynaimage.w.nextFrame.clicked.connect(self.nextFrame)
        self.dynaimage.w.stopButton.clicked.connect(self.stop_timer)
        self.dynaimage.w.playButton.clicked.connect(self.start_timer)
        self.animKeyEditor.w.keyButton.clicked.connect(self.addCurrentFrame)

        self.animSliders.w.frames.valueChanged.connect(self.update_timeline)

        self.w.thumbnails.thumbs.itemClicked.connect(self.viewImageClicked)
        self.w.thumbnails.thumbs.itemDoubleClicked.connect(self.tileImageClicked)

        self.w.sizer_count.w.heightNumber.display(str(self.w.sizer_count.w.heightSlider.value()))
        self.w.sizer_count.w.widthNumber.display(str(self.w.sizer_count.w.widthSlider.value()))
        self.w.sizer_count.w.samplesNumber.display(str(self.w.sizer_count.w.samplesSlider.value()))
        self.w.sizer_count.w.batchSizeNumber.display(str(self.w.sizer_count.w.batchSizeSlider.value()))
        self.w.sampler.w.stepsNumber.display(str(self.w.sampler.w.steps.value()))
        self.w.sampler.w.scaleNumber.display(str(self.w.sampler.w.scale.value() / 100))

        self.animSliders.w.framesNumber.display(str(self.animSliders.w.frames.value()))
        self.animSliders.w.ddim_etaNumber.display(str(self.animSliders.w.ddim_eta.value()))
        self.animSliders.w.strengthNumber.display(str(self.animSliders.w.strength.value()))
        self.animSliders.w.mask_contrastNumber.display(str(self.animSliders.w.mask_contrast.value()))
        self.animSliders.w.mask_brightnessNumber.display(str(self.animSliders.w.mask_brightness.value()))
        self.animSliders.w.mask_blurNumber.display(str(self.animSliders.w.mask_blur.value()))
        self.animSliders.w.fovNumber.display(str(self.animSliders.w.fov.value()))
        self.animSliders.w.midas_weightNumber.display(str(self.animSliders.w.midas_weight.value()))
        self.animSliders.w.near_planeNumber.display(str(self.animSliders.w.near_plane.value()))
        self.animSliders.w.far_planeNumber.display(str(self.animSliders.w.far_plane.value()))

        self.w.dynaview.w.setMinimumSize(QtCore.QSize(512, 256))

        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.compass.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.animKeys.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sampler.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.w.sizer_count.w.dockWidget)

        self.w.tabifyDockWidget(self.animKeys.w.dockWidget, self.w.sampler.w.dockWidget)
        self.w.tabifyDockWidget(self.w.sampler.w.dockWidget, self.w.sizer_count.w.dockWidget)

        # self.w.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.camera)

        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.prompt_fetcher.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.w.prompt.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline)

        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.thumbnails)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.animSliders.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.w.dynaview.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.path_setup.w.dockWidget)
        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dynaimage.w.dockWidget)

        self.dynaimage.w.setMinimumSize(QtCore.QSize(400, 256))

        self.w.tabifyDockWidget(self.path_setup.w.dockWidget, self.w.thumbnails)
        self.w.tabifyDockWidget(self.w.thumbnails, self.w.dynaview.w.dockWidget)
        self.w.tabifyDockWidget(self.w.dynaview.w.dockWidget, self.animSliders.w.dockWidget)
        self.w.tabifyDockWidget(self.animSliders.w.dockWidget, self.dynaimage.w.dockWidget)

        self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.animKeyEditor.w.dockWidget)

        self.w.tabifyDockWidget(self.timeline, self.w.prompt.w.dockWidget)

        self.path_setup.w.dockWidget.setWindowTitle('Path Setup')
        self.animKeys.w.dockWidget.setWindowTitle('Anim Keys')
        self.w.thumbnails.setWindowTitle('Thumbnails')
        self.w.sampler.w.dockWidget.setWindowTitle('Sampler')
        self.w.sizer_count.w.dockWidget.setWindowTitle('Image Setup')
        self.animSliders.w.dockWidget.setWindowTitle('Anim Setup')
        self.timeline.setWindowTitle('Timeline')
        self.w.prompt.w.dockWidget.setWindowTitle('Prompt')
        self.w.dynaview.w.dockWidget.setWindowTitle('Tensor Preview')
        self.dynaimage.w.dockWidget.setWindowTitle('Image Preview')
        self.w.preview.w.setWindowTitle('Canvas')
        self.prompt_fetcher.w.setWindowTitle('Prompt Fetcher')
        self.outpaint.setWindowTitle('Outpaint')
        self.compass.w.dockWidget.setWindowTitle('Compass')

        self.vpainter = {}
        self.w.preview.w.scene = QGraphicsScene()
        self.w.preview.w.graphicsView.setScene(self.w.preview.w.scene)
        self.w.preview.canvas = QPixmap(512, 512)
        self.w.preview.canvas.fill(Qt.white)
        self.w.preview.w.scene.addPixmap(self.w.preview.canvas)
        self.w.thumbnails.thumbsZoom.valueChanged.connect(self.updateThumbsZoom)
        self.w.thumbnails.refresh.clicked.connect(self.load_history)
        self.w.imageItem = QGraphicsPixmapItem()
        self.newPixmap = {}
        self.tpixmap = {}
        self.updateRate = self.w.sampler.w.steps.value()  # todo whats that? todo is plans we might share there for us and contributors to see what to implement next
        self.livePainter = QPainter()
        self.vpainter["iins"] = QPainter()
        self.tpixmap = QPixmap(512, 512)
        self.prompt_fetcher.w.getPrompts.clicked.connect(self.get_prompts)
        self.prompt_fetcher.w.aiPrompt.clicked.connect(self.ai_prompt)
        self.prompt_fetcher.w.usePrompt.clicked.connect(self.use_prompt)
        self.prompt_fetcher.w.dreamPrompt.clicked.connect(self.dream_prompt)

        self.load_settings()
        self.w.actiontest_save_output.triggered.connect(self.test_save_outpaint)


    def load_upscalers(self):
        gfpgan = False
        try:
            from realesrgan import RealESRGANer

            gfpgan = True
        except ModuleNotFoundError:
            pass

        if gfpgan:
            print('Loading models from RealESRGAN and facexlib')
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from facexlib.utils.face_restoration_helper import FaceRestoreHelper

                RealESRGANer(
                    scale=2,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth',
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=2,
                    ),
                )

                RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    model=RRDBNet(
                        num_in_ch=3,
                        num_out_ch=3,
                        num_feat=64,
                        num_block=23,
                        num_grow_ch=32,
                        scale=4,
                    ),
                )

                FaceRestoreHelper(1, det_model='retinaface_resnet50')
                print('...success')
            except Exception:
                import traceback

                print('Error loading GFPGAN:')
                print(traceback.format_exc())

    def prepare_loading(self):
        transformers.logging.set_verbosity_error()

        # this will preload the Bert tokenizer fles
        print('preloading bert tokenizer...')

        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        print('...success')

        # this will download requirements for Kornia
        print('preloading Kornia requirements (ignore the deprecation warnings)...')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DeprecationWarning)
        print('...success')

        version = 'openai/clip-vit-large-patch14'

        print('preloading CLIP model (Ignore the deprecation warnings)...')
        sys.stdout.flush()
        # self.load_upscalers()
        tokenizer = CLIPTokenizer.from_pretrained(version)
        transformer = CLIPTextModel.from_pretrained(version)
        print('\n\n...success')

        # In the event that the user has installed GFPGAN and also elected to use
        # RealESRGAN, this will attempt to download the model needed by RealESRGANer

    def restart(self):
        os.execl(sys.executable, sys.executable, *sys.argv)

    def del_widgets(self):
        self.w.thumbnails.destroy()

    def show_image_lab(self):
        self.image_lab.show()

    def ai_prompt(self):
        out_text = ''
        prompts_txt = self.prompt_fetcher.w.input.toPlainText()
        prompts_array = prompts_txt.split('\n')
        for prompt in prompts_array:
            out_text += generate_prompt(prompt)
        self.prompt_fetcher.w.output.setPlainText(out_text)


    def use_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        self.w.prompt.w.textEdit.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))

    def dream_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        self.w.prompt.w.textEdit.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))
        self.taskSwitcher()

    def get_prompts(self):
        out_text = ''
        prompts_txt = self.prompt_fetcher.w.input.toPlainText()
        prompts_array = prompts_txt.split('\n')
        for prompt in prompts_array:
            prompt = urllib.parse.quote_plus(prompt)
            response = requests.get("https://lexica.art/api/v1/search?q=" + prompt)
            res = response.text
            res = json.loads(res)
            if 'images' in res:
                for image in res['images']:
                    out_text = out_text + str(image['prompt']) + '\n\n'  # this is here for it to be better to read
        self.prompt_fetcher.w.output.setPlainText(out_text)

    # show
    def show_anim(self):
        self.w.anim.w.show()

    def show_preview(self):
        if self.previewpos == "outerdock":
            self.preview_as_central()
            self.previewpos = "center"
        else:
            self.show_paint()
            self.dynaimage = Dynaimage()

            self.w.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.dynaimage.w.dockWidget)
            self.previewpos = "outerdock"

    def show_prompt(self):
        self.w.prompt.w.show()

    def show_sampler(self):
        self.w.sampler.w.show()

    def show_sizer_count(self):
        self.w.sizer_count.w.show()

    def show_thumbnails(self):
        self.w.thumbnails.show()

    def show_path_settings(self):
        self.w.path_setup.show()

    def show_nodes(self):
        try:
            self.outpaint.destroy()
            del self.outpaint
        except:
            pass
        self.nodeWindow = NodeWindow()
        self.w.setCentralWidget(self.nodeWindow)

    def preview_as_central(self):
        self.dynaimage.w.destroy()
        self.w.setCentralWidget(self.dynaimage.w.dockWidget)

    def show_paint(self):
        try:
            self.nodeWindow.destroy()
            del self.nodeWindow
        except:
            pass
        self.outpaint = OutpaintUI()
        self.w.setCentralWidget(self.outpaint)

    def update_outpaint_parameters(self):
        W = self.w.sizer_count.w.widthSlider.value()
        H = self.w.sizer_count.w.heightSlider.value()
        W, H = map(lambda x: x - x % 64, (W, H))



        self.outpaint.canvas.w = W
        self.outpaint.canvas.h = H

    def torch_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def test_save_outpaint(self):
        """painter = QPainter()
        bitmap = QBitmap(512, 512)
        bitmap.clear()  # Starts with random data visible.


        pixmap = self.outpaint.canvas.pixmap.copy()
        painter.begin(bitmap)
        rect = QRect(256, 0, 256, 512)
        painter.setPen(QPen(Qt.color1))
        painter.setBrush(QBrush(Qt.color1))
        painter.drawPolygon(rect)


        pixmap.setMask(bitmap)

        painter.end()"""
        self.outpaint.canvas.pixmap = self.outpaint.canvas.pixmap.copy(QRect(64, 32, 512, 512))
        # Set our created mask on the image.

        # Calculate the bounding rect and return a copy of that region.
        # self.outpaint.canvas.pixmap = pixmap.copy(rect.boundingRect())
        self.outpaint.canvas.setPixmap(self.outpaint.canvas.pixmap)
        self.outpaint.canvas.update()

    def translate_sampler(self, sampler):
        if sampler == "k_lms":
            sampler = "klms"
        elif sampler == "k_dpm_2":
            sampler = "dpm2"
        elif sampler == "k_dpm_2_a":
            sampler = "dpm2_ancestral"
        elif sampler == "k_heun":
            sampler = "heun"
        elif sampler == "k_euler":
            sampler = "euler"
        elif sampler == "k_euler_a":
            sampler = "euler_ancestral"

        return sampler

    # deforum
    def run_deforum(self, progress_callback=None):

        self.currentFrames = []
        self.renderedFrames = 0
        self.now = 0
        self.progress = 0.0
        self.update = 0
        self.updateRate = self.w.sizer_count.w.previewSlider.value()
        max_frames = self.animSliders.w.frames.value()
        self.onePercent = 100 / (1 * self.w.sampler.w.steps.value() * max_frames * max_frames)

        flip_2d_perspective = False

        keyframes = self.w.prompt.w.keyFrames.toPlainText()
        prompts = self.w.prompt.w.textEdit.toPlainText()
        prompt_series = pd.Series([np.nan for a in range(max_frames)])
        if keyframes == '':
            keyframes = "0"
        prom = prompts
        key = keyframes

        new_prom = list(prom.split("\n"))
        new_key = list(key.split("\n"))

        prompts = dict(zip(new_key, new_prom))

        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt
        prompt_series = prompt_series.ffill().bfill()

        # print(prompt_series)
        show_sample_per_step = True

        W = self.w.sizer_count.w.widthSlider.value()
        H = self.w.sizer_count.w.heightSlider.value()
        W, H = map(lambda x: x - x % 64, (W, H))
        self.w.sizer_count.w.widthSlider.setValue(W)
        self.w.sizer_count.w.heightSlider.setValue(H)

        self.torch_gc()

        self.deforum.render_animation(
            image_callback=self.imageCallback_signal,
            step_callback=self.deforumstepCallback_signal if self.w.sampler.w.tensorPreview.isChecked() else None,
            animation_prompts=prompt_series,
            H=H,
            W=W,
            seed=random.randint(0, 2**32 - 1) if self.w.sampler.w.seed.text() == '' else int(self.w.sampler.w.seed.text()),
            sampler_name=self.translate_sampler(self.w.sampler.w.sampler.currentText()),
            steps=self.w.sampler.w.steps.value(),
            scale=self.w.sampler.w.scale.value() / 100,
            ddim_eta=self.animSliders.w.ddim_eta.value() / 1000,
            dynamic_threshold=None,
            static_threshold=None,
            save_samples=self.animKeys.w.saveSamples.isChecked(),
            save_settings=self.animKeys.w.saveSettings.isChecked(),
            display_samples=self.animKeys.w.displaySamples.isChecked(),
            save_sample_per_step=self.animKeys.w.saveStepSample.isChecked(),
            show_sample_per_step=self.animKeys.w.showStepSample.isChecked(),
            prompt_weighting=self.animKeys.w.promptWeighting.isChecked(),
            log_weighted_subprompts=self.animKeys.w.logPromptWeight.isChecked(),
            adabins=self.animSliders.w.adabins.isChecked(),
            batch_name="StableFun",
            seed_behavior=self.w.sampler.w.seedBehavior.currentText(),
            make_grid=self.animKeys.w.makeGrid.isChecked(),
            use_init=self.animKeys.w.useInit.isChecked(),
            strength=self.animSliders.w.strength.value() / 1000,
            strength_0_no_init=self.animKeys.w.strength0.isChecked(),
            init_image="",
            use_mask=self.animKeys.w.useMask.isChecked(),
            use_alpha_as_mask=self.animKeys.w.useAlphaMask.isChecked(),
            mask_file="",  # @param {type:"string"}
            mask_contrast_adjust=self.animSliders.w.mask_contrast.value() / 1000,
            mask_brightness_adjust=self.animSliders.w.mask_brightness.value() / 1000,
            overlay_mask=self.animKeys.w.overlayMask.isChecked(),
            #mask_blur = self.animSliders.w.mask_blur.value() / 1000,
            mask_overlay_blur=5,
            precision='autocast',
            timestring="",
            init_latent=None,
            init_sample=None,
            init_c=None,
            animation_mode='3D',
            max_frames = self.animSliders.w.frames.value(),
            border=self.animKeys.w.border.currentText(),  # @param ['wrap', 'replicate'] {type:'string'}
            angle=self.animKeys.w.angle.toPlainText(),
            zoom=self.animKeys.w.zoom.toPlainText(),
            translation_x=self.animKeys.w.trans_x.toPlainText(),
            translation_y=self.animKeys.w.trans_y.toPlainText(),
            translation_z=self.animKeys.w.trans_z.toPlainText(),
            rotation_3d_x=self.animKeys.w.rot_x.toPlainText(),
            rotation_3d_y=self.animKeys.w.rot_y.toPlainText(),
            rotation_3d_z=self.animKeys.w.rot_z.toPlainText(),
            flip_2d_perspective=flip_2d_perspective,
            perspective_flip_theta=self.animKeys.w.persp_theta.toPlainText(),
            perspective_flip_phi=self.animKeys.w.persp_phi.toPlainText(),
            perspective_flip_gamma=self.animKeys.w.persp_gamma.toPlainText(),
            perspective_flip_fv=self.animKeys.w.persp_fv.toPlainText(),
            noise_schedule=self.animKeys.w.noise_sched.toPlainText(),
            strength_schedule=self.animKeys.w.strength_sched.toPlainText(),
            contrast_schedule=self.animKeys.w.contrast_sched.toPlainText(),
            diffusion_cadence=self.animKeys.w.cadenceSlider.value(),
            color_coherence=self.animKeys.w.colorCoherence.currentText(),  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB'] {type:'string'}
            use_depth_warping=self.animKeys.w.useDepthWarp.isChecked(),  # @param {type:"boolean"}
            midas_weight=self.animSliders.w.midas_weight.value() / 1000,
            near_plane=self.animSliders.w.near_plane.value(),
            far_plane=self.animSliders.w.far_plane.value(),
            fov=self.animSliders.w.fov.value(),
            padding_mode=self.w.sampler.w.paddingMode.currentText(),  # @param ['border', 'reflection', 'zeros'] {type:'string'}
            sampling_mode=self.w.sampler.w.sampleMode.currentText(),  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
            save_depth_maps=self.animKeys.w.saveDepthMask.isChecked(),  # @param {type:"boolean"}
            use_mask_video=self.animKeys.w.useMaskVideo.isChecked(),  # @param {type:"boolean"}
            resume_from_timestring=self.animKeys.w.resumeTimestring.isChecked(),  # @param {type:"boolean"}
            resume_timestring=self.animKeys.w.timestring.text(),
            clear_latent=self.animSliders.w.clearLatent.isChecked(),
            clear_sample=self.animSliders.w.clearSample.isChecked(),
            shouldStop=False,
            cpudepth=self.animSliders.w.cpudepth_checkBox.isChecked()
            )

        self.torch_gc()
        self.stop_painters()

        self.signals.reenable_runbutton.emit()

    def deforumTest(self, *args, **kwargs):

        print(type(self.data1))
        print(self.data1)
        # saved_args = locals()
        # print(callback.x)
        # print("saved_args is", saved_args)

    def compviscallbackSignal(self, data, *args, **kwargs):
        self.data1 = data
        self.signals.compviscallback.emit()

    def deforumstepCallback_signal(self, data, data2=None):
        self.data = data
        if data2 is not None:
            self.data2 = data2
        else:
            self.data2 = None
        self.signals.deforum_step.emit()

    def deforum_thread(self):
        self.deforum = DeforumGenerator()
        self.deforum.signals = Callbacks()
        self.w.prompt.w.stopButton.clicked.connect(self.deforum.setStop)

        self.w.prompt.w.runButton.setEnabled(False)
        self.prompt_fetcher.w.dreamPrompt.setEnabled(False)
        QTimer.singleShot(100, lambda: self.pass_object())  # todo why we need that timer here doing nothing?

        worker = Worker(self.run_deforum)
        # Execute
        self.threadpool.start(worker)

    def run_deforum_txt2img(self, progress_callback=None):
        self.deforum = DeforumGenerator()
        self.deforum.signals = Callbacks()
        #self.w.prompt.w.runButton.setEnabled(False)
        #self.prompt_fetcher.w.dreamPrompt.setEnabled(False)
        #self.torch_gc()
        self.progress = 0.0
        self.update = 0
        self.onePercent = 100 / (1 * self.w.sampler.w.steps.value())
        self.updateRate = self.w.sizer_count.w.previewSlider.value()
        self.currentFrames = []
        self.renderedFrames = 0

        if self.w.sizer_count.w.samplesSlider.value() == 1:
            makegrid = False
        else:
            makegrid = self.animKeys.w.makeGrid.isChecked()
        sampler_name = self.translate_sampler(self.w.sampler.w.sampler.currentText())
        #init_image = self.outpaint.canvas.outpaintsource if
        self.deforum.sampler_name = sampler_name
        self.deforum.run_txt2img(strength=0,#strength=self.animSliders.w.strength.value() / 1000,
                                 seed=random.randint(0, 2**32 - 1) if self.w.sampler.w.seed.text() == '' else int(self.w.sampler.w.seed.text()),
                                 use_init=self.animSliders.w.useInit.isChecked(),
                                 init_image=None,
                                 sampler_name=sampler_name,
                                 ddim_eta=self.animSliders.w.ddim_eta.value() / 1000,
                                 animation_mode='None',
                                 prompts=self.w.prompt.w.textEdit.toPlainText(),
                                 max_frames=1,
                                 outdir=self.path_setup.w.txt2imgOut.text(),
                                 save_settings=False,
                                 save_samples=True,
                                 n_batch=self.w.sizer_count.w.batchSizeSlider.value(),
                                 makegrid=makegrid,
                                 grid_rows=2,
                                 filename_format="{timestring}_{index}_{prompt}.png",
                                 seed_behavior=self.w.sampler.w.seedBehavior.currentText(),
                                 steps=self.w.sampler.w.steps.value(),
                                 H=self.w.sizer_count.w.heightSlider.value(),
                                 W=self.w.sizer_count.w.widthSlider.value(),
                                 n_samples=self.w.sizer_count.w.samplesSlider.value(),  # batchsize
                                 scale=self.w.sampler.w.scale.value() / 100,
                                 step_callback=self.deforumstepCallback_signal if self.w.sampler.w.tensorPreview.isChecked() else None,
                                 image_callback=self.imageCallback_signal)

        #self.torch_gc()
        #self.stop_painters()

        self.signals.reenable_runbutton.emit()

    def run_deforum_outpaint(self, progress_callback=None):
        self.deforum = DeforumGenerator()
        self.deforum.signals = Callbacks()
        #self.w.prompt.w.runButton.setEnabled(False)
        #self.prompt_fetcher.w.dreamPrompt.setEnabled(False)
        #self.torch_gc()
        self.progress = 0.0
        self.update = 0
        self.onePercent = 100 / (1 * self.w.sampler.w.steps.value())
        self.updateRate = self.w.sizer_count.w.previewSlider.value()
        self.currentFrames = []
        self.renderedFrames = 0
        self.deforum.sample_number = 1
        if self.w.sizer_count.w.samplesSlider.value() == 1:
            makegrid = False
        else:
            makegrid = self.animKeys.w.makeGrid.isChecked()
        sampler_name = self.translate_sampler(self.w.sampler.w.sampler.currentText())

        #self.outpaint.canvas.outpaintsource = 'test0.jpg'
        init_image = self.outpaint.canvas.outpaintsource
        init_image = 'test0.png'
        self.deforum.sampler_name = sampler_name
        self.deforum.outpaint_txt2img(init_image=init_image,
                                      mask_blur=self.animSliders.w.mask_blur.value() / 10,
                                      scale=self.w.sampler.w.scale.value() / 100,
                                      ddim_eta=self.animSliders.w.ddim_eta.value() / 1000,
                                      image_callback=self.imageCallback_signal)

        #self.torch_gc()
        #self.stop_painters()

        self.signals.reenable_runbutton.emit()


    def deforum_txt2img_thread(self):
        # for debug
        #self.run_deforum_txt2img()
        # Pass the function to execute
        worker = Worker(self.run_deforum_txt2img)
        # Execute
        self.threadpool.start(worker)

    def deforum_outpaint_thread(self):
        # for debug
        #self.run_deforum_txt2img()
        # Pass the function to execute
        worker = Worker(self.run_deforum_outpaint)
        # Execute
        self.threadpool.start(worker)

    # slots
    @Slot()
    def deforumstepCallback_func(self):
        self.updateRate = self.w.sizer_count.w.previewSlider.value()
        self.progress = self.progress + self.onePercent
        self.w.progressBar.setValue(self.progress)
        if self.choice == "Text to Video" or self.choice == "Text to Image":
            if self.deforum.sampler_name == "ddim" or self.deforum.sampler_name == "plms":
                self.liveUpdate(self.data)
            else:
                self.liveUpdate(self.data['denoised'], self.data['i'])
        elif self.choice == "Text to Image LM":
            self.liveUpdate(self.data)

    @Slot(str)
    def add_image_to_thumbnail(self, path):
        #self.w.statusBar().showMessage("Ready...")

        #path = self.image_path
        self.w.thumbnails.thumbs.addItem(
            QListWidgetItem(QIcon(path), str(path)))

    @Slot()
    def imageCallback_func(self, image=None, seed=None, upscaled=False, use_prefix=None, first_seed=None, advance=True):
        self.painter = QPainter()
        self.ipixmap = QPixmap(self.image.im.size[0], self.image.im.size[1])
        self.painter.begin(self.ipixmap)
        if self.videoPreview == True and self.renderedFrames > 0:
            qimage = ImageQt(self.currentFrames[self.now])
            self.painter.drawImage(QRect(0, 0, self.image.im.size[0], self.image.im.size[1]), qimage)
            if advance == True:
                self.now += 1
            if self.now > (self.renderedFrames - 1):
                self.now = 0
            self.timeline.timeline.pointerTimePos = self.now

        elif self.renderedFrames > 0 and self.videoPreview == False:
            image = self.image.convert("RGBA")
            qimage = ImageQt(image)
            if self.outpaint.canvas.selected_item is not None:
                for items in self.outpaint.canvas.rectlist:
                    if items.id == self.outpaint.canvas.selected_item:
                        items.image = qimage


            self.painter.drawImage(QRect(0, 0, self.image.im.size[0], self.image.im.size[1]), qimage)

        self.dynaimage.w.label.setPixmap(
            self.ipixmap.scaled(self.image.im.size[0], self.image.im.size[1], Qt.AspectRatioMode.KeepAspectRatio))
        self.painter.end()

    @Slot()
    def reenableRunButton(self):
        try:
            self.w.prompt.w.runButton.setEnabled(True)
            self.prompt_fetcher.w.dreamPrompt.setEnabled(True)
        except:
            pass
        try:
            self.stop_timer()
        except:
            pass

    # timer
    def start_timer(self, *args, **kwargs):
        self.ftimer.timeout.connect(self.imageCallback_func)

        self.videoPreview = True
        self.ftimer.start(80)

    def stop_timer(self):
        self.ftimer.stop()
        self.videoPreview = False

    # callback
    def imageCallback_signal(self, image, *args, **kwargs):

        if self.choice == "Text to Image":
            if self.deforum.sample_number > 1:
                #self.image_path = image
                self.signals.add_image_to_thumbnail_signal.emit(image)
            else:
                self.currentFrames.append(image)
                self.renderedFrames += 1
                self.image = image
                self.signals.txt2img_image_cb.emit()
        elif self.choice == "Outpaint":
            self.currentFrames.append(image)
            self.renderedFrames += 1
            self.image = image
            self.signals.txt2img_image_cb.emit()

    # text2img
    def run_txt2img_lm(self, progress_callback=None):

        self.w.prompt.w.runButton.setEnabled(False)
        self.prompt_fetcher.w.dreamPrompt.setEnabled(False)

        self.currentFrames = []
        self.renderedFrames = 0

        self.w.statusBar().showMessage("Loading model...")
        # self.load_upscalers()

        self.updateRate = self.w.sizer_count.w.previewSlider.value()

        prompt_list = self.w.prompt.w.textEdit.toPlainText()
        prompt_list = prompt_list.split('\n')
        # self.w.setCentralWidget(self.dynaimage.w)
        steps = self.w.sampler.w.steps.value()
        samples = self.w.sizer_count.w.samplesSlider.value()
        batch_size = self.w.sizer_count.w.batchSizeSlider.value()

        self.onePercent = 100 / (steps * samples * batch_size * len(prompt_list))

        """The full list of arguments to Generate() are:
        gr = Generate(
                  weights     = path to model weights ('models/ldm/stable-diffusion-v1/model.ckpt')
                  config     = path to model configuraiton ('configs/stable-diffusion/v1-inference.yaml')
                  iterations  = <integer>     // how many times to run the sampling (1)
                  steps       = <integer>     // 50
                  seed        = <integer>     // current system time
                  sampler_name= ['ddim', 'k_dpm_2_a', 'k_dpm_2', 'k_euler_a', 'k_euler', 'k_heun', 'k_lms', 'plms']  // k_lms
                  grid        = <boolean>     // false
                  width       = <integer>     // image width, multiple of 64 (512)
                  height      = <integer>     // image height, multiple of 64 (512)
                  cfg_scale   = <float>       // condition-free guidance scale (7.5)
                  )

                    """

        # self.vpainter[tins] = QPainter(self.tpixmap)

        # self.vpainter[iins] = QPainter(dpixmap)

        # self.vpainter[iins].begin(dpixmap)

        self.progress = 0.0
        self.update = 0
        for i in range(batch_size):
            for prompt in prompt_list:
                print(prompt)
                self.torch_gc()
                results = self.gr.prompt2image(prompt=prompt,
                                               outdir=gs.system.txt2imgOut,
                                               cfg_scale=self.w.sampler.w.scale.value() / 100,
                                               width=self.w.sizer_count.w.widthSlider.value(),
                                               height=self.w.sizer_count.w.heightSlider.value(),
                                               iterations=self.w.sizer_count.w.samplesSlider.value(),
                                               steps=self.w.sampler.w.steps.value(),
                                               seamless=self.w.sampler.w.seamless.isChecked(),
                                               sampler_name=self.w.sampler.w.sampler.currentText(),
                                               seed=random.randint(0, 2**32 - 1) if self.w.sampler.w.seed.text() == '' else int(self.w.sampler.w.seed.text()),
                                               upscale=self.w.sizer_count.w.upScale.isChecked(),
                                               upscale_scale=self.w.sizer_count.w.upscaleScale.value(),
                                               upscale_strength=self.w.sizer_count.w.upscaleStrength.value(),
                                               use_gfpgan=self.w.sizer_count.w.useGfpgan.isChecked(),
                                               gfpgan_strength=self.w.sizer_count.w.gfpganSlider.value() / 100,
                                               strength=0.0,
                                               full_precision=self.w.sampler.w.fullPrecision.isChecked(),
                                               step_callback=self.deforumstepCallback_signal,
                                               image_callback=self.imageCallback_signal)
                for row in results:
                    print(f'filename={row[0]}')
                    print(f'seed    ={row[1]}')
                    filename = random.randint(10000, 99999)
                    output = f'{gs.system.txt2imgOut}/{filename}.png'
                    row[0].save(output)
                    self.image_path = output
                    self.signals.deforum_image_cb.emit()
                self.torch_gc()
            self.torch_gc()
        self.signals.reenable_runbutton.emit()
        # self.stop_painters()

    def txt2img_lm_thread(self):
        # Pass the function to execute
        worker = Worker(self.run_txt2img_lm)
        # Execute
        self.threadpool.start(worker)

    # gallery ??
    def tileImageClicked(self, item):

        vins = random.randint(10000, 99999)
        imageSize = item.icon().actualSize(QSize(10000, 10000))
        qimage = QImage(item.icon().pixmap(imageSize).toImage())
        self.newPixmap[vins] = QPixmap(QSize(2048, 2048))

        self.vpainter[vins] = QPainter(self.newPixmap[vins])

        newItem = QGraphicsPixmapItem()

        self.vpainter[vins].beginNativePainting()

        self.vpainter[vins].drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
        self.vpainter[vins].drawImage(QRect(QPoint(512, 0), QSize(qimage.size())), qimage)
        self.vpainter[vins].drawImage(QRect(QPoint(0, 512), QSize(qimage.size())), qimage)
        self.vpainter[vins].drawImage(QRect(QPoint(512, 512), QSize(qimage.size())), qimage)

        newItem.setPixmap(self.newPixmap[vins])

        for items in self.w.preview.w.scene.items():
            self.w.preview.w.scene.removeItem(items)
        self.w.preview.w.scene.addItem(newItem)
        self.w.preview.w.graphicsView.fitInView(newItem, Qt.AspectRatioMode.KeepAspectRatio)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.vpainter[vins].endNativePainting()

    def viewImageClicked(self, item):

        # vins = random.randint(10000, 99999)
        imageSize = item.icon().actualSize(QSize(10000, 10000))
        qimage = QImage(item.icon().pixmap(imageSize).toImage())
        pixmap = QPixmap(imageSize)
        painter = QPainter()
        painter.begin(pixmap)
        painter.drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
        painter.end()
        self.dynaimage.w.label.setPixmap(pixmap)
        #for items in self.outpaint.canvas.rectlist:
        #    print(f"adding image{qimage}")
        #    items.image = qimage

        # self.vpainter[vins] = QPainter()
        # newItem = QGraphicsPixmapItem()
        # self.vpainter[vins].begin(self.outpaint.canvas.pixmap)
        # self.vpainter[vins].drawImage(QRect(QPoint(0, 0), QSize(qimage.size())), qimage)
        # newItem.setPixmap(self.newPixmap[vins])

        # for items in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(items)

        # self.outpaint.canvas.setPixmap(self.outpaint.canvas.pixmap)
        # self.w.preview.w.scene.addItem(newItem)
        # self.w.preview.w.graphicsView.fitInView(newItem, Qt.AspectRatioMode.KeepAspectRatio)
        # self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # self.vpainter[vins].end()

    def zoom_IN(self):
        self.w.preview.w.graphicsView.scale(1.25, 1.25)

    def zoom_OUT(self):
        self.w.preview.w.graphicsView.scale(0.75, 0.75)

    # settings

    def create_out_folders(self):
        os.makedirs(gs.system.galleryMainPath, exist_ok=True)
        os.makedirs(gs.system.txt2imgOut, exist_ok=True)
        os.makedirs(gs.system.img2imgTmp, exist_ok=True)
        os.makedirs(gs.system.img2imgOut, exist_ok=True)
        os.makedirs(gs.system.txt2vidSingleFrame, exist_ok=True)
        os.makedirs(gs.system.txt2vidOut, exist_ok=True)
        os.makedirs(gs.system.vid2vidTmp, exist_ok=True)
        os.makedirs(gs.system.vid2vidSingleFrame, exist_ok=True)
        os.makedirs(gs.system.vid2vidOut, exist_ok=True)

    def load_default_diffusion_settings(self):
        self.load_settings(True)

    def load_settings(self, default=False):
        if not default:
            settings.load_settings_json()
        else:
            settings.load_default_settings_json()

        self.load_last_prompt()

        self.animKeys.w.angle.setText(gs.diffusion.angle)
        self.animKeys.w.zoom.setText(gs.diffusion.zoom)
        self.animKeys.w.trans_x.setText(gs.diffusion.trans_x)
        self.animKeys.w.trans_y.setText(gs.diffusion.trans_y)
        self.animKeys.w.trans_z.setText(gs.diffusion.trans_z)
        self.animKeys.w.rot_x.setText(gs.diffusion.rot_x)
        self.animKeys.w.rot_y.setText(gs.diffusion.rot_y)
        self.animKeys.w.rot_z.setText(gs.diffusion.rot_z)
        self.animKeys.w.flip2dPerspective.setChecked(gs.diffusion.flip2dPerspective)
        self.animKeys.w.persp_theta.setText(gs.diffusion.persp_theta)
        self.animKeys.w.persp_phi.setText(gs.diffusion.persp_phi)
        self.animKeys.w.persp_gamma.setText(gs.diffusion.persp_gamma)
        self.animKeys.w.persp_fv.setText(gs.diffusion.persp_fv)
        self.animKeys.w.noise_sched.setText(gs.diffusion.noise_sched)
        self.animKeys.w.strength_sched.setText(gs.diffusion.strength_sched)
        self.animKeys.w.contrast_sched.setText(gs.diffusion.contrast_sched)
        self.animKeys.w.cadenceSlider.setValue(gs.diffusion.cadence)

        self.animKeys.w.saveStepSample.setChecked(gs.diffusion.saveStepSample)
        self.animKeys.w.showStepSample.setChecked(gs.diffusion.showStepSample)
        self.animKeys.w.promptWeighting.setChecked(gs.diffusion.promptWeighting)
        self.animKeys.w.logPromptWeight.setChecked(gs.diffusion.logPromptWeight)
        self.animKeys.w.saveSamples.setChecked(gs.diffusion.saveSamples)
        self.animKeys.w.saveSettings.setChecked(gs.diffusion.saveSettings)
        self.animKeys.w.displaySamples.setChecked(gs.diffusion.displaySamples)
        self.animKeys.w.makeGrid.setChecked(gs.diffusion.makeGrid)
        self.animKeys.w.useInit.setChecked(gs.diffusion.useInit)
        self.animKeys.w.strength0.setChecked(gs.diffusion.strength0)
        self.animKeys.w.useMask.setChecked(gs.diffusion.useMask)
        self.animKeys.w.useAlphaMask.setChecked(gs.diffusion.useAlphaMask)
        self.animKeys.w.overlayMask.setChecked(gs.diffusion.overlayMask)
        self.animKeys.w.useDepthWarp.setChecked(gs.diffusion.useDepthWarp)
        self.animKeys.w.saveDepthMask.setChecked(gs.diffusion.saveDepthMask)
        self.animKeys.w.useMaskVideo.setChecked(gs.diffusion.useMaskVideo)
        self.animKeys.w.resumeTimestring.setChecked(gs.diffusion.resumeTimestring)
        self.animKeys.w.saveStepSample.setChecked(gs.diffusion.saveStepSample)
        self.animKeys.w.saveStepSample.setChecked(gs.diffusion.saveStepSample)
        self.animKeys.w.colorCoherence.setCurrentIndex(gs.diffusion.colorCoherence)
        self.animKeys.w.border.setCurrentIndex(gs.diffusion.border)

        self.w.sizer_count.w.heightSlider.setValue(gs.diffusion.H)
        self.w.sizer_count.w.widthSlider.setValue(gs.diffusion.W)
        self.w.sizer_count.w.samplesSlider.setValue(gs.diffusion.n_samples)
        self.w.sizer_count.w.batchSizeSlider.setValue(gs.diffusion.batch_size)
        self.w.sampler.w.scale.setValue(gs.diffusion.scale * 100)
        self.w.sampler.w.steps.setValue(gs.diffusion.steps)
        self.w.sizer_count.w.upScale.setChecked(gs.diffusion.upScale)
        self.w.sizer_count.w.upscaleScale.setValue(gs.diffusion.upscale_scale)
        self.w.sizer_count.w.upscaleStrength.setValue(gs.diffusion.upscale_strength)
        self.w.sizer_count.w.useGfpgan.setChecked(gs.diffusion.use_gfpgan)
        self.w.sizer_count.w.gfpganSlider.setValue(gs.diffusion.gfpgan_strength)

        self.w.sampler.w.fullPrecision.setChecked(gs.diffusion.fullPrecision)
        self.w.sampler.w.seamless.setChecked(gs.diffusion.seamless)
        self.w.sampler.w.seed.setText(gs.diffusion.seed)

        self.w.sampler.w.processType.setCurrentIndex(gs.diffusion.processType)
        self.w.sampler.w.sampler.setCurrentIndex(gs.diffusion.sampler)
        self.w.sampler.w.sampleMode.setCurrentIndex(gs.diffusion.sampleMode)
        self.w.sampler.w.seedBehavior.setCurrentIndex(gs.diffusion.seedBehavior)
        self.w.sampler.w.paddingMode.setCurrentIndex(gs.diffusion.paddingMode)

        self.animSliders.w.frames.setValue(gs.diffusion.frames)
        self.animSliders.w.ddim_eta.setValue(gs.diffusion.ddim_eta)
        self.animSliders.w.strength.setValue(gs.diffusion.strength)
        self.animSliders.w.mask_contrast.setValue(gs.diffusion.mask_contrast)
        self.animSliders.w.mask_brightness.setValue(gs.diffusion.mask_brightness)
        self.animSliders.w.mask_blur.setValue(gs.diffusion.mask_blur)
        self.animSliders.w.fov.setValue(gs.diffusion.fov)
        self.animSliders.w.midas_weight.setValue(gs.diffusion.midas_weight)
        self.animSliders.w.near_plane.setValue(gs.diffusion.near_plane)
        self.animSliders.w.far_plane.setValue(gs.diffusion.far_plane)

        self.animSliders.w.useInit.setChecked(gs.diffusion.useInit)
        self.animSliders.w.adabins.setChecked(gs.diffusion.adabins)
        self.animSliders.w.clearLatent.setChecked(gs.diffusion.clearLatent)
        self.animSliders.w.clearSample.setChecked(gs.diffusion.clearSample)

        self.path_setup.w.galleryMainPath.setText(gs.system.galleryMainPath)
        self.path_setup.w.txt2imgOut.setText(gs.system.txt2imgOut)
        self.path_setup.w.img2imgTmp.setText(gs.system.img2imgTmp)
        self.path_setup.w.img2imgOut.setText(gs.system.img2imgOut)
        self.path_setup.w.txt2vidSingleFrame.setText(gs.system.txt2vidSingleFrame)
        self.path_setup.w.txt2vidOut.setText(gs.system.txt2vidOut)
        self.path_setup.w.vid2vidTmp.setText(gs.system.vid2vidTmp)
        self.path_setup.w.vid2vidSingleFrame.setText(gs.system.vid2vidSingleFrame)
        self.path_setup.w.vid2vidOut.setText(gs.system.vid2vidOut)

        self.path_setup.w.adabinsPath.setText(gs.system.adabinsPath)
        self.path_setup.w.midasPath.setText(gs.system.midasPath)
        self.path_setup.w.sdClipPath.setText(gs.system.sdClipPath)
        self.path_setup.w.sdPath.setText(gs.system.sdPath)
        self.path_setup.w.sdInference.setText(gs.system.sdInference)
        self.path_setup.w.gfpganPath.setText(gs.system.gfpganPath)
        self.path_setup.w.realesrganPath.setText(gs.system.realesrganPath)
        self.path_setup.w.realesrganAnimeModelPath.setText(gs.system.realesrganAnimeModelPath)
        self.path_setup.w.ffmpegPath.setText(gs.system.ffmpegPath)
        self.path_setup.w.settingsPath.setText(gs.system.settingsPath)

        self.path_setup.w.gfpganCpu.setChecked(gs.system.gfpganCpu)
        self.path_setup.w.realesrganCpu.setChecked(gs.system.realesrganCpu)
        self.path_setup.w.extraModelsCpu.setChecked(gs.system.extraModelsCpu)
        self.path_setup.w.extraModelsGpu.setChecked(gs.system.extraModelsGpu)

        self.path_setup.w.gpu.setText(str(gs.system.gpu))
        self.create_out_folders()

    def load_last_prompt(self):
        data = ''
        try:
            with open('configs/ainodes/last_prompt.txt', 'r') as file:
                data = file.read().replace('\n', '')
        except:
            pass
        self.w.prompt.w.textEdit.setPlainText(data)

    def save_last_prompt(self):
        f = open('configs/ainodes/last_prompt.txt', 'w')
        f.write(self.w.prompt.w.textEdit.toPlainText())
        f.close()

    def save_diffusion_settings(self):
        gs.diffusion.angle = self.animKeys.w.angle.toPlainText()
        gs.diffusion.angle = self.animKeys.w.angle.toPlainText()
        gs.diffusion.zoom = self.animKeys.w.zoom.toPlainText()
        gs.diffusion.trans_x = self.animKeys.w.trans_x.toPlainText()
        gs.diffusion.trans_y = self.animKeys.w.trans_y.toPlainText()
        gs.diffusion.trans_z = self.animKeys.w.trans_z.toPlainText()
        gs.diffusion.rot_x = self.animKeys.w.rot_x.toPlainText()
        gs.diffusion.rot_y = self.animKeys.w.rot_y.toPlainText()
        gs.diffusion.rot_z = self.animKeys.w.rot_z.toPlainText()
        gs.diffusion.flip2dPerspective = self.animKeys.w.flip2dPerspective.setChecked()
        gs.diffusion.persp_theta = self.animKeys.w.persp_theta.toPlainText()
        gs.diffusion.persp_phi = self.animKeys.w.persp_phi.toPlainText()
        gs.diffusion.persp_gamma = self.animKeys.w.persp_gamma.toPlainText()
        gs.diffusion.persp_fv = self.animKeys.w.persp_fv.toPlainText()
        gs.diffusion.noise_sched = self.animKeys.w.noise_sched.toPlainText()
        gs.diffusion.strength_sched = self.animKeys.w.strength_sched.toPlainText()
        gs.diffusion.contrast_sched = self.animKeys.w.contrast_sched.toPlainText()
        gs.diffusion.cadence = self.animKeys.w.cadenceSlider.value()

        gs.diffusion.saveStepSample = self.animKeys.w.saveStepSample.isChecked()
        gs.diffusion.showStepSample = self.animKeys.w.showStepSample.isChecked()
        gs.diffusion.promptWeighting = self.animKeys.w.promptWeighting.isChecked()
        gs.diffusion.logPromptWeight = self.animKeys.w.logPromptWeight.isChecked()
        gs.diffusion.saveSamples = self.animKeys.w.saveSamples.isChecked()
        gs.diffusion.saveSettings = self.animKeys.w.saveSettings.isChecked()
        gs.diffusion.displaySamples = self.animKeys.w.displaySamples.isChecked()
        gs.diffusion.makeGrid = self.animKeys.w.makeGrid.isChecked()
        gs.diffusion.useInit = self.animKeys.w.useInit.isChecked()
        gs.diffusion.strength0 = self.animKeys.w.strength0.isChecked()
        gs.diffusion.useMask = self.animKeys.w.useMask.isChecked()
        gs.diffusion.useAlphaMask = self.animKeys.w.useAlphaMask.isChecked()
        gs.diffusion.overlayMask = self.animKeys.w.overlayMask.isChecked()
        gs.diffusion.useDepthWarp = self.animKeys.w.useDepthWarp.isChecked()
        gs.diffusion.saveDepthMask = self.animKeys.w.saveDepthMask.isChecked()
        gs.diffusion.useMaskVideo = self.animKeys.w.useMaskVideo.isChecked()
        gs.diffusion.resumeTimestring = self.animKeys.w.resumeTimestring.isChecked()
        gs.diffusion.saveStepSample = self.animKeys.w.saveStepSample.isChecked()
        gs.diffusion.saveStepSample = self.animKeys.w.saveStepSample.isChecked()
        gs.diffusion.colorCoherence = self.animKeys.w.colorCoherence.currentIndex()
        gs.diffusion.border = self.animKeys.w.border.currentIndex()

        gs.diffusion.H = self.w.sizer_count.w.heightSlider.value()
        gs.diffusion.W = self.w.sizer_count.w.widthSlider.value()
        gs.diffusion.n_samples = self.w.sizer_count.w.samplesSlider.value()
        gs.diffusion.batch_size = self.w.sizer_count.w.batchSizeSlider.value()
        gs.diffusion.scale = self.w.sampler.w.scale.value() / 100
        gs.diffusion.steps = self.w.sampler.w.steps.value()
        gs.diffusion.upScale = self.w.sizer_count.w.upScale.isChecked()
        gs.diffusion.upscale_scale = self.w.sizer_count.w.upscaleScale.value()
        gs.diffusion.upscale_strength = self.w.sizer_count.w.upscaleStrength.value()
        gs.diffusion.use_gfpgan = self.w.sizer_count.w.useGfpgan.isChecked()
        gs.diffusion.gfpgan_strength = self.w.sizer_count.w.gfpganSlider.value()

        gs.diffusion.scale = self.w.sampler.w.scale.value() / 100
        gs.diffusion.steps = self.w.sampler.w.steps.value()
        gs.diffusion.fullPrecision = self.w.sampler.w.fullPrecision.isChecked()
        gs.diffusion.seamless = self.w.sampler.w.seamless.isChecked()
        gs.diffusion.seed = self.w.sampler.w.seed.text()
        gs.diffusion.sampler = self.w.sampler.w.sampler.currentIndex()
        gs.diffusion.sampleMode = self.w.sampler.w.sampleMode.currentIndex()
        gs.diffusion.seedBehavior = self.w.sampler.w.seedBehavior.currentIndex()
        gs.diffusion.processType = self.w.sampler.w.processType.currentIndex()
        gs.diffusion.paddingMode = self.w.sampler.w.paddingMode.currentIndex()

        gs.diffusion.frames = self.animSliders.w.frames.value()
        gs.diffusion.ddim_eta = self.animSliders.w.ddim_eta.value()
        gs.diffusion.strength = self.animSliders.w.strength.value()
        gs.diffusion.mask_contrast = self.animSliders.w.mask_contrast.value()
        gs.diffusion.mask_brightness = self.animSliders.w.mask_brightness.value()
        gs.diffusion.mask_blur = self.animSliders.w.mask_blur.value()
        gs.diffusion.fov = self.animSliders.w.fov.value()
        gs.diffusion.midas_weight = self.animSliders.w.midas_weight.value()
        gs.diffusion.near_plane = self.animSliders.w.near_plane.value()
        gs.diffusion.far_plane = self.animSliders.w.far_plane.value()

        gs.diffusion.useInit = self.animSliders.w.useInit.isChecked()
        gs.diffusion.adabins = self.animSliders.w.adabins.isChecked()
        gs.diffusion.clearLatent = self.animSliders.w.clearLatent.isChecked()
        gs.diffusion.clearSample = self.animSliders.w.clearSample.isChecked()

        settings.save_settings_json()

    def save_system_settings(self):
        gs.system.galleryMainPath = self.path_setup.w.galleryMainPath.text()
        gs.system.txt2imgOut = self.path_setup.w.txt2imgOut.text()
        gs.system.img2imgTmp = self.path_setup.w.img2imgTmp.text()
        gs.system.img2imgOut = self.path_setup.w.img2imgOut.text()
        gs.system.txt2vidSingleFrame = self.path_setup.w.txt2vidSingleFrame.text()
        gs.system.txt2vidOut = self.path_setup.w.txt2vidOut.text()
        gs.system.vid2vidTmp = self.path_setup.w.vid2vidTmp.text()
        gs.system.vid2vidSingleFrame = self.path_setup.w.vid2vidSingleFrame.text()
        gs.system.vid2vidOut = self.path_setup.w.vid2vidOut.text()

        gs.system.adabinsPath = self.path_setup.w.adabinsPath.text()
        gs.system.midasPath = self.path_setup.w.midasPath.text()
        gs.system.sdClipPath = self.path_setup.w.sdClipPath.text()
        gs.system.sdPath = self.path_setup.w.sdPath.text()
        gs.system.sdInference = self.path_setup.w.sdInference.text()
        gs.system.gfpganPath = self.path_setup.w.gfpganPath.text()
        gs.system.realesrganPath = self.path_setup.w.realesrganPath.text()
        gs.system.realesrganAnimeModelPath = self.path_setup.w.realesrganAnimeModelPath.text()
        gs.system.ffmpegPath = self.path_setup.w.ffmpegPath.text()
        gs.system.settingsPath = self.path_setup.w.settingsPath.text()

        gs.system.gfpganCpu = self.path_setup.w.gfpganCpu.isChecked()
        gs.system.realesrganCpu = self.path_setup.w.realesrganCpu.isChecked()
        gs.system.extraModelsCpu = self.path_setup.w.extraModelsCpu.isChecked()
        gs.system.extraModelsGpu = self.path_setup.w.extraModelsGpu.isChecked()

        gs.system.gpu = int(self.path_setup.w.gpu.text())
        settings.save_settings_json()

    # threading

    # gets called by the dream button(s)
    def taskSwitcher(self):
        self.save_last_prompt()
        self.choice = self.w.sampler.w.processType.currentText()
        self.sample_number = self.w.sizer_count.w.samplesSlider.value()


        if self.choice == "Text to Video":
                self.deforum_thread()
        elif self.choice == "Text to Image LM":
            self.txt2img_lm_thread()
        elif self.choice == 'Text to Image':
            self.deforum_txt2img_thread()
        elif self.choice == 'Outpaint':
            self.deforum_outpaint_thread()

    # dont know yet
    def load_history(self):
        self.w.thumbnails.thumbs.clear()
        for image in gs.album:
            self.w.thumbnails.thumbs.addItem(QListWidgetItem(QIcon(image), str(image)))

    def viewThread(self, item):
        self.viewImageClicked(item)

    def prevFrame(self):
        if self.now > 0:
            self.now -= 2
            advance = False
            self.videoPreview = True
            self.imageCallback_func(advance)
            self.videoPreview = False

    def nextFrame(self):
        if self.now < self.renderedFrames:
            self.now += 1
            advance = False
            self.videoPreview = True
            self.imageCallback_func(advance)

    def stop_painters(self):
        try:
            self.vpainter["iins"].end()
            self.livePainter.end()
        except Exception as e:
            print(f"Exception: {e}")
            pass

    def testThread(self, data1=None, data2=None):

        self.updateRate = self.w.sizer_count.w.previewSlider.value()

        self.progress = self.progress + self.onePercent
        self.w.progressBar.setValue(self.progress)
        # Pass the function to execute
        self.liveWorker2 = Worker(self.liveUpdate(data1, data2))

        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #    self.liveUpdate(data1, data2)

        self.threadpool.start(self.liveWorker2)

    def liveUpdate(self, data1=None, data2=None):
        self.update += 1
        self.w.statusBar().showMessage(f"Generating... (step {data2} of {self.steps})")
        updateRate = self.w.sizer_count.w.previewSlider.value()
        if self.update >= updateRate:
            try:
                self.update = 0
                self.test_output(data1, data2)
            except Exception as e:
                print(f"Exception: {e}")
                self.update = 0
            finally:
                return

    def test_output(self, data1, data2):
        tpixmap = QPixmap(self.w.sizer_count.w.widthSlider.value(), self.w.sizer_count.w.heightSlider.value())
        self.livePainter.begin(tpixmap)
        x_samples = torch.clamp((data1 + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            print(
                f'we got {len(x_samples)} Tensors but Tensor Preview will show only one')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )
        # self.x_sample = cv2.cvtColor(self.x_sample.astype(np.uint8), cv2.COLOR_RGB2BGR)
        x_sample = x_sample.astype(np.uint8)
        dPILimg = Image.fromarray(x_sample)
        dqimg = ImageQt(dPILimg)
        self.livePainter.drawImage(
            QRect(0, 0, self.w.sizer_count.w.widthSlider.value(), self.w.sizer_count.w.heightSlider.value()), dqimg)
        self.w.dynaview.w.label.setPixmap(
            tpixmap.scaled(self.w.sizer_count.w.widthSlider.value(), self.w.sizer_count.w.heightSlider.value(),
                           Qt.AspectRatioMode.KeepAspectRatio))
        self.livePainter.end()

    def pass_object(self, progress_callback=None):
        pass

    def get_pic(self, clear=False):  # from self.image_path
        # for item in self.w.preview.w.scene.items():
        #    self.w.preview.w.scene.removeItem(item)

        image_qt = QImage(self.image_path)

        self.w.preview.pic = QGraphicsPixmapItem()
        self.w.preview.pic.setPixmap(QPixmap.fromImage(image_qt))
        if clear == True:
            self.w.preview.w.scene.clear()
        self.w.preview.w.scene.addItem(self.w.preview.pic)

        self.w.preview.w.graphicsView.fitInView(self.w.preview.pic, Qt.AspectRatioMode.KeepAspectRatio)
        self.w.preview.w.graphicsView.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # gs.obj_to_delete = self.w.preview.pic

        # KEYFRAME FUNCTIONS

    def showTypeKeyframes(self):
        valueType = self.animKeyEditor.w.comboBox.currentText()
        print(valueType)
        self.timeline.timeline.selectedValueType = valueType
        self.updateAnimKeys()

    def sort_keys(self, e):
        return e.position

    def updateAnimKeys(self):
        tempString = ""
        valueType = self.animKeyEditor.w.comboBox.currentText()
        self.timeline.timeline.keyFrameList.sort(key=self.sort_keys)
        for item in self.timeline.timeline.keyFrameList:
            if item.valueType == valueType:
                if tempString == "":
                    tempString = f'{item.position}:({item.value})'
                else:
                    tempString = f'{tempString}, {item.position}:({item.value})'
        selection = self.animKeyEditor.w.comboBox.currentText()
        if tempString != "":
            if "Contrast" in selection:
                self.animKeys.w.contrast_sched.setText(tempString)
            if "Noise" in selection:
                self.animKeys.w.noise_sched.setText(tempString)
            if "Strength" in selection:
                self.animKeys.w.strength_sched.setText(tempString)
            if "Rotation X" in selection:
                self.animKeys.w.rot_x.setText(tempString)
            if "Rotation Y" in selection:
                self.animKeys.w.rot_y.setText(tempString)
            if "Rotation Z" in selection:
                self.animKeys.w.rot_z.setText(tempString)
            if "Translation X" in selection:
                self.animKeys.w.trans_x.setText(tempString)
            if "Translation Y" in selection:
                self.animKeys.w.trans_y.setText(tempString)
            if "Translation Z" in selection:
                self.animKeys.w.trans_z.setText(tempString)
            if "Angle" in selection:
                self.animKeys.w.angle.setText(tempString)
            if "Zoom" in selection:
                self.animKeys.w.zoom.setText(tempString)
        # perspective_flip_theta = self.animKeys.w.persp_theta.toPlainText()
        # perspective_flip_phi = self.animKeys.w.persp_phi.toPlainText()
        # perspective_flip_gamma = self.animKeys.w.persp_gamma.toPlainText()
        # perspective_flip_fv = self.animKeys.w.persp_fv.toPlainText()

    @Slot()
    def updateKeyFramesFromTemp(self):
        self.updateAnimKeys()

    def addCurrentFrame(self):
        matchFound = False
        value = self.animKeyEditor.w.valueText.value()
        valueType = self.animKeyEditor.w.comboBox.currentText()
        position = int(self.timeline.timeline.pointerTimePos)
        keyframe = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        keyframe[position] = KeyFrame(uid, valueType, position, value)
        for items in self.timeline.timeline.keyFrameList:
            if items.valueType == valueType:
                if items.position == position:
                    items.value = value
                    matchFound = True
        if matchFound == False:
            self.timeline.timeline.keyFrameList.append(keyframe[position])
        self.timeline.timeline.update()
        self.updateAnimKeys()

    # updates
    def update_timeline(self):
        self.timeline.timeline.duration = self.animSliders.w.frames.value()
        self.timeline.timeline.update()

    def updateThumbsZoom(self):
        while gs.callbackBusy == True:
            time.sleep(0.1)
        try:
            if gs.callbackBusy == False:
                size = self.w.thumbnails.thumbsZoom.value()
                self.w.thumbnails.thumbs.setGridSize(QSize(size, size))
                self.w.thumbnails.thumbs.setIconSize(QSize(size, size))
        except Exception as e:
            print(f"Exception: {e}")
            pass

    def update_scaleNumber(self):
        float = self.w.sampler.w.scale.value() / 100
        self.w.sampler.w.scaleNumber.display(str(float))

    def update_gfpganNumber(self):
        float = self.w.sizer_count.w.gfpganSlider.value() / 10
        self.w.sizer_count.w.gfpganNumber.display(str(float))

        # EVENT FILTERS

    def eventFilter(self, source, event):
        if event.type() == QEvent.ContextMenu and source is self.w.thumbnails.thumbs:
            menu = QMenu()
            tileAction = QAction("Tile", self)
            menu.addAction(tileAction)
            menu.addAction('Action 2')
            menu.addAction('Action 3')

            if menu.exec_(event.globalPos()):
                item = source.itemAt(event.pos())
                self.tileImageClicked(item)
                # print(item.text())
            return True
        return super().eventFilter(source, event)

    def closeEvent(self, event):
        print(f"close event{event.type()}")
        try:
            del self.nodeWindow
        except:
            pass
        sys.exit(0)


def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    import numexpr
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)
            # if it's only a number, leave the rest for the default interpolation
            if value_is_number:
                t = i
                key_frame_series[i] = value
        if not value_is_number:
            t = i
            key_frame_series[i] = numexpr.evaluate(value)
    key_frame_series = key_frame_series.astype(float)

    if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
        interp_method = 'Quadratic'
    if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
        interp_method = 'Linear'

    key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
    key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
    key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
    if integer:
        return key_frame_series.astype(int)
    return key_frame_series
