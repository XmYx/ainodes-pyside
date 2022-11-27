import os
import time
import random
from datetime import datetime
from uuid import uuid4
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QFile, QIODevice, QEasingCurve, Slot, QRect, QThreadPool, QDir, Signal, QObject
from PySide6.QtWidgets import QMainWindow, QToolBar, QPushButton, QGraphicsColorizeEffect, QListWidgetItem, QFileDialog, \
    QLabel
from PySide6.QtGui import QAction, QIcon, QColor, QPixmap, QPainter, Qt
from PySide6 import QtCore, QtWidgets
from backend.deforum.six.animation import check_is_number
from einops import rearrange

from backend.worker import Worker
from frontend import plugin_loader
from frontend.ui_model_chooser import ModelChooser_UI

from backend.prompt_ai.prompt_gen import AiPrompt
from frontend.ui_paint import PaintUI, spiralOrder, random_path
from frontend.ui_classes import Thumbnails, PathSetup, ThumbsUI, AnimKeyEditor
from frontend.unicontrol import UniControl
import backend.settings as settings
from backend.singleton import singleton
from frontend.ui_krea import Krea
from frontend.ui_lexica import LexicArt
from frontend.ui_prompt_fetcher import PromptFetcher_UI, FetchPrompts
from backend.devices import choose_torch_device
from frontend.ui_timeline import Timeline, KeyFrame


gs = singleton
settings.load_settings_json()
# we had to load settings first before we can do this import
from frontend.ui_image_lab import ImageLab
from frontend.ui_deforum import Deforum_UI
from frontend.session_params import SessionParams
from backend.shared import save_last_prompt

# please don't remove it totally, just remove what we know is not used
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



class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.signals = Callbacks()
        self.load_last_prompt()
        self.canvas = PaintUI(self)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("aiNodes - Still Mode")
        self.timeline = Timeline()
        self.animKeyEditor = AnimKeyEditor()

        self.resize(1280, 800)
        self.unicontrol = UniControl(self)
        self.sessionparams = SessionParams(self)
        self.sessionparams.create_params()
        self.thumbs = ThumbsUI()
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.unicontrol.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.animKeyEditor.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.thumbs.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.timeline)
        self.tabifyDockWidget(self.timeline, self.thumbs.w.dockWidget)

        self.create_main_toolbar()
        self.create_secondary_toolbar()

        self.sessionparams.add_state_to_history()
        self.update_ui_from_params()

        self.currentFrames = []
        self.renderedFrames = 0

        self.threadpool = QThreadPool()
        self.deforum_ui = Deforum_UI(self)


        self.y = 0
        self.lastheight = None
        self.height = gs.diffusion.H

        self.lexicart = LexicArt()
        self.krea = Krea()
        self.prompt_fetcher = FetchPrompts()
        self.prompt_fetcher_ui = PromptFetcher_UI(self)

        self.path_setup = PathSetup()
        self.image_lab = ImageLab()
        self.image_lab_ui = self.image_lab.imageLab
        self.model_chooser = ModelChooser_UI(self)
        self.unicontrol.w.dockWidget.setWindowTitle("Parameters")
        self.path_setup.w.dockWidget.setWindowTitle("Model / Paths")
        self.image_lab_ui.w.dockWidget.setWindowTitle("Image Lab")
        self.lexicart.w.dockWidget.setWindowTitle("Lexica Art")
        self.krea.w.dockWidget.setWindowTitle("Krea")
        self.prompt_fetcher.w.dockWidget.setWindowTitle("Prompt Fetcher")

        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.image_lab_ui.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.path_setup.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.lexicart.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.krea.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.prompt_fetcher.w.dockWidget)

        self.tabifyDockWidget(self.lexicart.w.dockWidget, self.krea.w.dockWidget)
        self.tabifyDockWidget(self.krea.w.dockWidget, self.prompt_fetcher.w.dockWidget)
        self.tabifyDockWidget(self.prompt_fetcher.w.dockWidget, self.image_lab_ui.w.dockWidget)
        self.tabifyDockWidget(self.image_lab_ui.w.dockWidget, self.path_setup.w.dockWidget)
        self.tabifyDockWidget(self.path_setup.w.dockWidget, self.unicontrol.w.dockWidget)
        self.hide_default()
        self.mode = 'txt2img'
        self.init_plugin_loader()
        self.connections()
        self.list_files()
        self.resize(1280, 800)


    def connections(self):
        self.deforum_ui.signals.txt2img_image_cb.connect(self.image_preview_func)
        self.deforum_ui.signals.deforum_step.connect(self.tensor_preview_schedule)

        self.canvas.W.valueChanged.connect(self.canvas.canvas.change_resolution)
        self.canvas.H.valueChanged.connect(self.canvas.canvas.change_resolution)

        self.canvas.canvas.signals.update_selected.connect(self.show_outpaint_details)
        self.canvas.canvas.signals.update_params.connect(self.create_params)
        self.canvas.canvas.signals.outpaint_signal.connect(self.deforum_ui.deforum_outpaint_thread)
        self.canvas.canvas.signals.txt2img_signal.connect(self.deforum_six_txt2img_thread)

        #self.canvas.canvas.signals.update_selected.connect(self.show_outpaint_details)
        #self.canvas.canvas.signals.update_params.connect(self.create_params)
        #self.canvas.canvas.signals.outpaint_signal.connect(self.deforum_ui.deforum_outpaint_thread)
        self.canvas.canvas.signals.txt2img_signal.connect(self.deforum_six_txt2img_thread)

        self.unicontrol.w.dream.clicked.connect(self.taskswitcher)
        self.unicontrol.w.H.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)
        self.unicontrol.w.W.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)
        self.unicontrol.w.lucky.clicked.connect(self.show_default)


        self.unicontrol.w.redo.clicked.connect(self.redo_current_outpaint)
        self.unicontrol.w.delete_2.clicked.connect(self.delete_outpaint_frame)
        self.unicontrol.w.preview_batch.clicked.connect(self.preview_batch_outpaint)
        #self.outpaint_controls.w.createBatch.clicked.connect(self.prepare_batch_outpaint_thread)
        self.unicontrol.w.run_batch.clicked.connect(self.run_prepared_outpaint_batch_thread)
        self.unicontrol.w.run_hires.clicked.connect(self.run_hires_batch_thread)
        self.unicontrol.w.prep_hires.clicked.connect(self.run_create_outpaint_img2img_batch)
        self.unicontrol.w.update_params.clicked.connect(self.update_params)

        self.unicontrol.w.W.valueChanged.connect(self.update_outpaint_parameters)
        self.unicontrol.w.H.valueChanged.connect(self.update_outpaint_parameters)
        self.unicontrol.w.mask_offset.valueChanged.connect(self.outpaint_offset_signal)
        self.unicontrol.w.mask_offset.valueChanged.connect(self.canvas.canvas.set_offset(int(self.unicontrol.w.mask_offset.value())))  # todo does this work?
        self.unicontrol.w.rect_overlap.valueChanged.connect(self.outpaint_rect_overlap)

        self.timeline.timeline.keyFramesUpdated.connect(self.updateKeyFramesFromTemp)
        self.animKeyEditor.w.comboBox.currentTextChanged.connect(self.showTypeKeyframes)
        self.animKeyEditor.w.keyButton.clicked.connect(self.addCurrentFrame)


        self.image_lab.signals.upscale_start.connect(self.upscale_start)
        self.image_lab.signals.upscale_stop.connect(self.upscale_stop)
        self.image_lab.signals.upscale_counter.connect(self.upscale_count)
        self.image_lab.signals.img_to_txt_start.connect(self.img_to_text_start)
        self.image_lab.signals.watermark_start.connect(self.watermark_start)
        self.image_lab.signals.model_merge_start.connect(self.model_merge_start)
        self.image_lab.signals.ebl_model_merge_start.connect(self.ebl_model_merge_start)
        self.image_lab.signals.run_aestetic_prediction.connect(self.run_aestetic_prediction_thread)
        self.image_lab.signals.run_interrogation.connect(self.run_interrogation_thread)

        self.prompt_fetcher_ui.signals.run_ai_prompt.connect(self.ai_prompt_thread)
        self.prompt_fetcher_ui.signals.run_img_to_prompt.connect(self.image_to_prompt_thread)
        self.prompt_fetcher_ui.signals.get_lexica_prompts.connect(self.get_lexica_prompts_thread)
        self.prompt_fetcher_ui.signals.got_image_to_prompt.connect(self.prompt_fetcher_ui.set_img_to_prompt_text)
        self.prompt_fetcher_ui.signals.got_lexica_prompts.connect(self.prompt_fetcher_ui.set_lexica_prompts)
        self.prompt_fetcher_ui.signals.get_krea_prompts.connect(self.get_krea_prompts_thread)
        self.prompt_fetcher_ui.signals.got_krea_prompts.connect(self.prompt_fetcher_ui.set_krea_prompts)

    def taskswitcher(self):
        save_last_prompt(self.unicontrol.w.prompts.toHtml(), self.unicontrol.w.prompts.toPlainText())
        if self.unicontrol.w.use_inpaint.isChecked() == True:
            self.canvas.canvas.reusable_outpaint(self.canvas.canvas.selected_item)
            self.deforum_ui.deforum_outpaint_thread()
        else:
            self.deforum_six_txt2img_thread()

    def path_setup_temp(self):
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

    def still_mode(self):
        pass

    def anim_mode(self):
        pass

    def node_mode(self):
        pass

    def gallery_mode(self):
        pass

    def settings_mode(self):
        pass

    def help_mode(self):
        pass

    @Slot()
    def run_interrogation_thread(self):
        worker = Worker(self.image_lab.run_interrogation)
        self.threadpool.start(worker)

    @Slot()
    def ai_prompt_thread(self):
        self.aiPrompt = AiPrompt()
        self.aiPrompt.signals.ai_prompt_ready.connect(self.prompt_fetcher_ui.set_ai_prompt)
        self.aiPrompt.signals.status_update.connect(self.set_status_bar)
        worker = Worker(self.aiPrompt.get_prompts, self.prompt_fetcher.w.input.toPlainText())
        self.threadpool.start(worker)

    @Slot()
    def image_to_prompt_thread(self):
        worker = Worker(self.prompt_fetcher_ui.get_img_to_prompt)
        self.threadpool.start(worker)

    @Slot()
    def use_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        self.prompt.w.textEdit.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))
    @Slot()
    def dream_prompt(self):
        prompt = self.prompt_fetcher.w.output.textCursor().selectedText()
        self.prompt.w.textEdit.setPlainText(prompt.replace(u'\u2029\u2029', '\n'))
        self.taskSwitcher()

    @Slot()
    def get_lexica_prompts_thread(self):
        worker = Worker(self.prompt_fetcher_ui.get_lexica_prompts)
        self.threadpool.start(worker)

    @Slot()
    def get_krea_prompts_thread(self):
        worker = Worker(self.prompt_fetcher_ui.get_krea_prompts)
        self.threadpool.start(worker)

    @Slot()
    def set_status_bar(self, txt):
        self.statusBar().showMessage(txt)

    def upscale_start(self):
        self.signals.setStatusBar.emit("Upscale started...")
        self.upscale_thread()

    def upscale_stop(self):
        self.signals.setStatusBar.emit("Upscale finished...")

    def upscale_count(self, num):
        self.signals.setStatusBar.emit(f"Upscaled {str(num)} image(s)...")

    @Slot()
    def upscale_thread(self):
        worker = Worker(self.image_lab.run_upscale)
        self.threadpool.start(worker)
    @Slot()
    def img_to_text_start(self):
        worker = Worker(self.image_lab.run_img2txt)
        self.threadpool.start(worker)
    @Slot()
    def watermark_start(self):
        worker = Worker(self.image_lab.run_watermark)
        self.threadpool.start(worker)
    @Slot()
    def model_merge_start(self):
        worker = Worker(self.image_lab.model_merge_start)
        self.threadpool.start(worker)
    @Slot()
    def ebl_model_merge_start(self):
        worker = Worker(self.image_lab.ebl_model_merge_start)
        self.threadpool.start(worker)

    @Slot()
    def run_aestetic_prediction_thread(self):
        worker = Worker(self.image_lab.run_aestetic_prediction)
        self.threadpool.start(worker)

    @Slot()
    def run_interrogation_thread(self):
        worker = Worker(self.image_lab.run_interrogation)
        self.threadpool.start(worker)

    @Slot()
    def ai_prompt_thread(self):
        self.aiPrompt = AiPrompt()
        self.aiPrompt.signals.ai_prompt_ready.connect(self.prompt_fetcher_ui.set_ai_prompt)
        self.aiPrompt.signals.status_update.connect(self.set_status_bar)
        worker = Worker(self.aiPrompt.get_prompts, self.prompt_fetcher.w.input.toPlainText())
        self.threadpool.start(worker)

    @Slot()
    def image_to_prompt_thread(self):
        worker = Worker(self.prompt_fetcher_ui.get_img_to_prompt)
        self.threadpool.start(worker)



    def update_ui_from_params(self):
        for key, value in self.sessionparams.params.items():
            try:
                #We have to add check for Animation Mode as thats a radio checkbox with values 'anim2d', 'anim3d', 'animVid'
                #add colormatch_image (it will be with a fancy preview)
                type = str(getattr(self.unicontrol.w, key))

                if 'QSpinBox' in type or 'QDoubleSpinBox' in type:
                    getattr(self.unicontrol.w, key).setValue(value)
                elif  'QTextEdit' in type or 'QLineEdit' in type:
                    getattr(self.unicontrol.w, key).setText(str(value))
                elif 'QCheckBox' in type:
                    if value == True:
                        getattr(self.unicontrol.w, key).setCheckState(QtCore.Qt.Checked)


            except Exception as e:
                print(e)
                continue


    def init_plugin_loader(self):
        self.unicontrol.w.loadbutton.clicked.connect(self.load_plugin)
        self.unicontrol.w.unloadbutton.clicked.connect(self.unload_plugin)
        self.plugins = plugin_loader.PluginLoader(self)
        list = self.plugins.list_plugins()
        for i in list:
            self.unicontrol.w.plugins.addItem(i)

    def load_plugin(self):
        plugin_name = self.unicontrol.w.plugins.currentText()
        self.plugins.load_plugin(f"plugins.{plugin_name}.{plugin_name}")

    def unload_plugin(self):
        plugin_name = self.unicontrol.w.plugins.currentText()
        self.plugins.unload_plugin(plugin_name)

    def create_main_toolbar(self):
        self.toolbar = QToolBar('Outpaint Tools')
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
        still_mode = QAction(QIcon_from_svg('frontend/icons/instagram.svg'), 'Still', self)
        anim_mode = QAction(QIcon_from_svg('frontend/icons/film.svg'), 'Anim', self)
        node_mode = QAction(QIcon_from_svg('frontend/icons/image.svg'), 'Nodes', self)
        gallery_mode = QAction(QIcon_from_svg('frontend/icons/image.svg'), 'Gallery', self)
        settings_mode = QAction(QIcon_from_svg('frontend/icons/image.svg'), 'Settings', self)
        help_mode = QAction(QIcon_from_svg('frontend/icons/help-circle.svg'), 'Help', self)

        self.toolbar.addAction(still_mode)
        self.toolbar.addAction(anim_mode)
        self.toolbar.addAction(node_mode)
        self.toolbar.addAction(gallery_mode)
        self.toolbar.addAction(settings_mode)
        self.toolbar.addAction(help_mode)

    def create_secondary_toolbar(self):
        self.secondary_toolbar = QToolBar('Outpaint Tools')
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.secondary_toolbar)
        select_mode = QAction(QIcon_from_svg('frontend/icons/mouse-pointer.svg'), 'Still', self)
        drag_mode = QAction(QIcon_from_svg('frontend/icons/wind.svg'), 'Anim', self)
        add_mode = QAction(QIcon_from_svg('frontend/icons/plus.svg'), 'Nodes', self)
        save_canvas = QAction(QIcon_from_svg('frontend/icons/save.svg'), 'Save as Json', self)
        save_canvas_png = QAction(QIcon_from_svg('frontend/icons/save.svg'), 'Save as PNG', self)
        clear_canvas = QAction(QIcon_from_svg('frontend/icons/frown.svg'), 'Clear Canvas', self)
        load_canvas = QAction(QIcon_from_svg('frontend/icons/folder.svg'), 'Load from Json', self)

        self.secondary_toolbar.addAction(select_mode)
        self.secondary_toolbar.addAction(drag_mode)
        self.secondary_toolbar.addAction(add_mode)
        self.secondary_toolbar.addAction(save_canvas)
        self.secondary_toolbar.addAction(save_canvas_png)
        self.secondary_toolbar.addAction(clear_canvas)
        self.secondary_toolbar.addAction(load_canvas)

        select_mode.triggered.connect(self.canvas.canvas.select_mode)
        drag_mode.triggered.connect(self.canvas.canvas.drag_mode)
        add_mode.triggered.connect(self.canvas.canvas.add_mode)

        save_canvas.triggered.connect(self.canvas.canvas.save_rects_as_json)
        load_canvas.triggered.connect(self.canvas.canvas.load_rects_from_json)
        clear_canvas.triggered.connect(self.canvas.canvas.reset)
        save_canvas_png.triggered.connect(self.canvas.canvas.save_canvas)



    def hide_default(self):
        self.toolbar.setVisible(False)
        self.secondary_toolbar.setVisible(False)

        self.unicontrol.w.enable_negative_prompt.setVisible(False)
        self.unicontrol.w.negative_prompts.setVisible(False)
        self.unicontrol.w.prompt_weighting.setVisible(False)
        self.unicontrol.w.toggle_sampler.setVisible(False)
        self.unicontrol.w.toggle_outpaint.setVisible(False)
        self.unicontrol.w.toggle_animations.setVisible(False)
        self.unicontrol.w.toggle_plotting.setVisible(False)
        self.unicontrol.w.toggle_aesthetics.setVisible(False)
        self.unicontrol.w.toggle_embeddings.setVisible(False)
        self.unicontrol.w.toggle_plugins.setVisible(False)

        #self.unicontrol.w.showHideAll.setVisible(False)
        self.unicontrol.w.H.setVisible(False)
        self.unicontrol.w.H_slider.setVisible(False)
        self.unicontrol.w.W.setVisible(False)
        self.unicontrol.w.W_slider.setVisible(False)
        self.unicontrol.w.cfglabel.setVisible(False)
        self.unicontrol.w.heightlabel.setVisible(False)
        self.unicontrol.w.widthlabel.setVisible(False)
        self.unicontrol.w.steps.setVisible(False)
        self.unicontrol.w.steps_slider.setVisible(False)
        self.unicontrol.w.scale.setVisible(False)
        self.unicontrol.w.scale_slider.setVisible(False)
        self.unicontrol.w.stepslabel.setVisible(False)
        self.path_setup.w.dockWidget.setVisible(False)
        self.animKeyEditor.w.dockWidget.setVisible(False)
        self.image_lab_ui.w.dockWidget.setVisible(False)
        self.lexicart.w.dockWidget.setVisible(False)
        self.krea.w.dockWidget.setVisible(False)
        self.prompt_fetcher.w.dockWidget.setVisible(False)

        if self.unicontrol.samHidden == False:
            self.unicontrol.hideSampler_anim()
        if self.unicontrol.aesHidden == False:
            self.unicontrol.hideAesthetic_anim()
        if self.unicontrol.aniHidden == False:
            self.unicontrol.hideAnimation_anim()

        if self.unicontrol.ploHidden == False:
            self.unicontrol.hidePlotting_anim()

        self.thumbs.w.dockWidget.setVisible(False)

        self.default_hidden = True

    def show_default(self):
        if self.default_hidden == True:
            self.toolbar.setVisible(True)
            self.secondary_toolbar.setVisible(True)

            self.unicontrol.w.enable_negative_prompt.setVisible(True)
            self.unicontrol.w.prompt_weighting.setVisible(True)
            self.unicontrol.w.toggle_sampler.setVisible(True)
            self.unicontrol.w.toggle_outpaint.setVisible(True)
            self.unicontrol.w.toggle_animations.setVisible(True)
            self.unicontrol.w.toggle_plotting.setVisible(True)
            self.unicontrol.w.toggle_aesthetics.setVisible(True)
            self.unicontrol.w.toggle_embeddings.setVisible(True)
            self.unicontrol.w.toggle_plugins.setVisible(True)
            self.unicontrol.w.H.setVisible(True)
            self.unicontrol.w.H_slider.setVisible(True)
            self.unicontrol.w.W.setVisible(True)
            self.unicontrol.w.W_slider.setVisible(True)
            self.unicontrol.w.cfglabel.setVisible(True)
            self.unicontrol.w.heightlabel.setVisible(True)
            self.unicontrol.w.widthlabel.setVisible(True)
            self.unicontrol.w.steps.setVisible(True)
            self.unicontrol.w.steps_slider.setVisible(True)
            self.unicontrol.w.scale.setVisible(True)
            self.unicontrol.w.scale_slider.setVisible(True)
            self.unicontrol.w.stepslabel.setVisible(True)
            self.path_setup.w.dockWidget.setVisible(True)
            self.image_lab_ui.w.dockWidget.setVisible(True)
            self.lexicart.w.dockWidget.setVisible(True)
            self.krea.w.dockWidget.setVisible(True)
            self.prompt_fetcher.w.dockWidget.setVisible(True)
            self.thumbs.w.dockWidget.setVisible(True)
            self.animKeyEditor.w.dockWidget.setVisible(True)

            self.default_hidden = False
        else:
            self.hide_default()

    def thumbnails_Animation(self):
        self.thumbsShow = QtCore.QPropertyAnimation(self.thumbnails, b"maximumHeight")
        self.thumbsShow.setDuration(2000)
        self.thumbsShow.setStartValue(0)
        self.thumbsShow.setEndValue(self.height() / 4)
        self.thumbsShow.setEasingCurve(QEasingCurve.Linear)
        self.thumbsShow.start()

    def load_last_prompt(self):
        data = ''
        try:
            with open('configs/ainodes/last_prompt.txt', 'r') as file:
                data = file.read().replace('\n', '')
        except:
            pass
        gs.diffusion.prompt = data
        #self.prompt.w.textEdit.setHtml(data)

    def deforum_six_txt2img_thread(self):
        self.update = 0
        height = self.height
        #for debug
        #self.deforum_ui.run_deforum_txt2img()
        self.params = self.sessionparams.update_params()
        self.sessionparams.add_state_to_history()
        #Prepare next rectangle, widen canvas:
        worker = Worker(self.deforum_ui.run_deforum_six_txt2img)
        self.threadpool.start(worker)


    def image_preview_signal(self, image, *args, **kwargs):
        self.image = image
        self.deforum_ui.signals.add_image_to_thumbnail_signal.emit(image)
        self.deforum_ui.signals.txt2img_image_cb.emit()
        self.currentFrames.append(image)
        self.renderedFrames += 1

    @Slot()
    def image_preview_func(self, image=None, seed=None, upscaled=False, use_prefix=None, first_seed=None, advance=True):
        x = 0
        y = 0
        if self.canvas.canvas.rectlist != []:
            for i in self.canvas.canvas.rectlist:
                if i.id == self.canvas.canvas.selected_item:
                    self.lastheight = self.height
                    #Calculate next image's X coordinate
                    x = i.x + i.w + 20
                    if i.h > self.height:
                        self.height = i.h
                        self.lastheight = self.height
                    if self.canvas.canvas.pixmap.width() < 3000:
                        w = x  + self.unicontrol.w.W.value() + 25
                        if w > self.w:
                            self.w = w
                    else:
                        w = self.w
                    if x > 3000:
                        self.y = self.y + self.unicontrol.w.H.value() + 20
                        x = 0
                        self.lastheight = self.lastheight + i.h + 20
                        self.height = self.lastheight
                        w = w
                        if w > self.w:
                            self.w = w
                    if self.lastheight is not None:
                        if self.lastheight < self.height + i.h + 20:
                            self.lastheight = self.height + i.h + 20
                            if self.sessionparams.params.advanced == False:
                                self.canvas.canvas.resize_canvas(w=self.w, h=self.lastheight + self.unicontrol.w.H.value())
                    y = self.y



            if x != 0 or y > 0:
                if self.sessionparams.params.advanced == False:
                    self.canvas.canvas.w = self.unicontrol.w.W.value()
                    self.canvas.canvas.h = self.unicontrol.w.H.value()
                    self.canvas.canvas.addrect_atpos(x=x, y=self.y, params=self.sessionparams.params)
                    print(f"resizing canvas to {self.height}")
                    self.canvas.canvas.resize_canvas(w=self.w, h=self.height)
        elif self.sessionparams.params.advanced == False or self.canvas.canvas.selected_item == None:
            w = self.unicontrol.w.W.value()
            h = self.unicontrol.w.H.value()
            self.canvas.canvas.w = w
            self.canvas.canvas.h = h
            self.canvas.canvas.addrect_atpos(x=0, y=0)
            self.height = self.unicontrol.w.H.value()
            #print(f"this should only haappen once {self.height}")
            self.canvas.canvas.resize_canvas(w=self.w, h=self.height)

        self.lastheight = self.height

        qimage = ImageQt(self.image.convert("RGBA"))
        for items in self.canvas.canvas.rectlist:
            if items.id == self.canvas.canvas.selected_item:
                if items.images is not None:
                    templist = items.images
                else:
                    templist = []
                items.PILImage = self.image
                templist.append(qimage)
                items.images = templist
                if items.index == None:
                    items.index = 0
                else:
                    items.index = items.index + 1
                items.image = items.images[items.index]
                self.canvas.canvas.newimage = True
                items.timestring = time.time()

                self.canvas.canvas.update()


    def tensor_preview_signal(self, data, data2):
        self.data = data

        if data2 is not None:
            self.data2 = data2
        else:
            self.data2 = None
        self.deforum_ui.signals.deforum_step.emit()

    def tensor_preview_schedule(self):
        x_samples = torch.clamp((self.data + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            print(
                f'we got {len(x_samples)} Tensors but Tensor Preview will show only one')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )

        x_sample = x_sample.astype(np.uint8)
        dPILimg = Image.fromarray(x_sample)
        dqimg = ImageQt(dPILimg)
        self.canvas.canvas.tensor_preview_item = dqimg
        self.canvas.canvas.tensor_preview()

    def tensor_draw_function(self, data1, data2):
        #tpixmap = QPixmap(self.sizer_count.w.widthSlider.value(), self.sizer_count.w.heightSlider.value())
        #self.livePainter.begin(tpixmap)
        x_samples = torch.clamp((self.data + 1.0) / 2.0, min=0.0, max=1.0)
        if len(x_samples) != 1:
            print(
                f'we got {len(x_samples)} Tensors but Tensor Preview will show only one')
        x_sample = 255.0 * rearrange(
            x_samples[0].cpu().numpy(), 'c h w -> h w c'
        )

        x_sample = x_sample.astype(np.uint8)
        dPILimg = Image.fromarray(x_sample)
        dqimg = ImageQt(dPILimg)
        self.canvas.canvas.tensor_preview_item = dqimg
        self.canvas.canvas.tensor_preview()

    def outpaint_offset_signal(self):

        value = int(self.unicontrol.w.offset_slider.value())
        self.canvas.canvas.set_offset(value)

    @Slot()
    def update_outpaint_parameters(self):
        W = self.unicontrol.w.W.value()
        H = self.unicontrol.w.H.value()
        #W, H = map(lambda x: x - x % 64, (W, H))
        self.unicontrol.w.W.setValue(W)
        self.unicontrol.w.H.setValue(H)

        self.canvas.canvas.w = W
        self.canvas.canvas.h = H

    def prep_rect_params(self, prompt=None):
        #prompt = str(prompt)
        #steps = self.unicontrol.w.stepsSlider.value()
        params = {"prompts": self.unicontrol.w.prompts.toPlainText(),
                  "seed":random.randint(0, 2**32 - 1) if self.unicontrol.w.seed.text() == '' else int(self.unicontrol.w.seed.text()),
                  "strength": self.unicontrol.w.strength.value(),
                  "scale":self.unicontrol.w.scale.value(),
                  "mask_blur":int(self.unicontrol.w.mask_blur.value()),
                  "reconstruction_blur":int(self.unicontrol.w.reconstruction_blur.value()),
                  "use_inpaint":self.unicontrol.w.use_inpaint.isChecked(),
                  "mask_offset":self.unicontrol.w.mask_offset.value(),
                  "steps":self.unicontrol.w.steps.value(),
                  "H":self.unicontrol.w.H.value(),
                  "W":self.unicontrol.w.W.value(),
                  "ddim_eta":self.unicontrol.w.ddim_eta.value()
                  }

        return params

    @Slot(str)
    def update_params(self, uid=None, params=None):
        if self.canvas.canvas.selected_item is not None:
            for i in self.canvas.canvas.rectlist:
                if uid is not None:
                    if i.id == uid:
                        if params == None:
                                params = self.get_params()
                        i.params = params
                else:
                    if i.id == self.canvas.canvas.selected_item:
                        params = self.get_params()
                        i.params = params

    @Slot(str)
    def create_params(self, uid=None):
        for i in self.canvas.canvas.rectlist:
            if i.id == uid:
                params = self.prep_rect_params()
                i.params = params


    def get_params(self):
        params = self.sessionparams.params()
        #print(f"Created Params")
        return params

    @Slot()
    def show_outpaint_details(self):

        if self.canvas.canvas.selected_item is not None:
            self.thumbs.w.thumbnails.clear()
            for items in self.canvas.canvas.rectlist:
                if items.id == self.canvas.canvas.selected_item:

                    if items.params != {}:
                        #print(f"showing strength of {items.params['strength'] * 100}")
                        self.unicontrol.w.steps.setValue(items.params['steps'])
                        self.unicontrol.w.steps_slider.setValue(items.params['steps'])
                        self.unicontrol.w.scale.setValue(items.params['scale'] * 10)
                        self.unicontrol.w.scale_slider.setValue(items.params['scale'] * 10)
                        self.unicontrol.w.strength.setValue(int(items.params['strength'] * 100))
                        self.unicontrol.w.strength_slider.setValue(int(items.params['strength'] * 100))
                        self.unicontrol.w.reconstruction_blur.setValue(items.params['reconstruction_blur'])
                        self.unicontrol.w.mask_blur.setValue(items.params['mask_blur'])
                        self.unicontrol.w.prompts.setText(items.params['prompts'])
                        self.unicontrol.w.seed.setText(str(items.params['seed']))
                        self.unicontrol.w.mask_offset.setValue(items.params['mask_offset'])

                    if items.images is not []:
                        for i in items.images:
                            if i is not None:
                                image = i.copy(0, 0, i.width(), i.height())
                                pixmap = QPixmap.fromImage(image)
                                self.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap), f"{items.index}"))

    def redo_current_outpaint(self):
        self.canvas.canvas.redo_outpaint(self.canvas.canvas.selected_item)

    def delete_outpaint_frame(self):
        #self.canvas.canvas.undoitems = []
        if self.canvas.canvas.selected_item is not None:
            x = 0
            for i in self.canvas.canvas.rectlist:
                if i.id == self.canvas.canvas.selected_item:
                    self.canvas.canvas.undoitems.append(i)
                    self.canvas.canvas.rectlist.pop(x)
                    pass
                x += 1

        self.canvas.canvas.update()
        self.canvas.canvas.pixmap.fill(Qt.transparent)
        self.canvas.canvas.newimage = True

    def test_save_outpaint(self):

        self.canvas.canvas.pixmap = self.canvas.canvas.pixmap.copy(QRect(64, 32, 512, 512))

        self.canvas.canvas.setPixmap(self.canvas.canvas.pixmap)
        self.canvas.canvas.update()

    @Slot()
    def stop_processing(self):
        self.stopprocessing = True

    def sort_rects(self, e):
        return e.order

    def run_batch_outpaint(self, progress_callback=False):
        self.stopprocessing = False
        self.callbackbusy = False
        self.sleepytime = 0.0
        self.choice = "Outpaint"
        self.create_outpaint_batch()

    def create_outpaint_batch(self, gobig_img_path=None):
        self.sessionparams.params.advanced = True
        self.callbackbusy = True
        x = 0
        self.busy = False
        offset = self.unicontrol.w.mask_offset.value()
        #self.preview_batch_outpaint()
        if gobig_img_path is not None:
            pil_image = Image.open(gobig_img_path).resize((self.canvas.W.value(),self.canvas.H.value()), Image.Resampling.LANCZOS).convert("RGBA")
            qimage = ImageQt(pil_image)
            chops_x = int(qimage.width() / self.canvas.canvas.w)
            chops_y = int(qimage.width() / self.canvas.canvas.h)
            self.preview_batch_outpaint(with_chops=chops_x, chops_y=chops_y)

        for items in self.canvas.canvas.tempbatch:
            if type(items) == list:
                for item in items:

                    if gobig_img_path is not None:
                        rect = QRect(item['x'], item['y'], self.canvas.canvas.w, self.canvas.canvas.h)
                        image = qimage.copy(rect)
                        index = None
                        self.hires_source = pil_image

                    else:
                        image = None
                        index = None
                        self.hires_source = None
                    offset = offset + 512
                    params = self.prep_rect_params(item["prompt"])

                    self.canvas.canvas.addrect_atpos(prompt=item["prompt"], x=item['x'], y=item['y'], image=image, index=index, order=item["order"], params=params)

                    #x = self.iterate_further(x)
                    x += 1
                    while self.busy == True:
                        time.sleep(0.25)
            elif type(items) == dict:

                if gobig_img_path is not None:
                    rect = QRect(item['x'], item['y'], self.canvas.canvas.w, self.canvas.canvas.h)
                    image = qimage.copy(rect)
                    index = None
                    self.hires_source = pil_image

                else:
                    image = None
                    index = None
                    self.hires_source = None
                offset = offset + 512

                params = self.prep_rect_params(items["prompt"])

                self.canvas.canvas.addrect_atpos(prompt=items["prompt"], x=items['x'], y=items['y'], image=image, index=index, order=items["order"], params=params)

                #x = self.iterate_further(x)
                x += 1
                while self.busy == True:
                    time.sleep(0.25)
        self.callbackbusy = False

    def run_hires_batch(self, progress_callback=None):
        self.sessionparams.params.advanced = True
        #multi = self.unicontrol.w.multiBatch.isChecked()
        #batch_n = self.unicontrol.w.multiBatchvalue.value()
        multi = False
        batch_n = 1
        self.stopprocessing = False
        self.callbackbusy = False
        self.sleepytime = 0.0
        self.choice = "Outpaint"

        for i in range(batch_n):
            while self.callbackbusy == True:
                time.sleep(0.5)
            time.sleep(1)
            betterslices = []
            og_size = (512, 512)
            tiles = (self.canvas.canvas.cols - 1) * (self.canvas.canvas.rows - 1)
            for x in range(int(tiles)):
                if self.stopprocessing == False:
                    self.run_hires_step_x(x)
                    betterslices.append((self.image.convert('RGBA'), self.canvas.canvas.rectlist[x].x, self.canvas.canvas.rectlist[x].y))
                else:
                    break

            source_image = self.hires_source
            alpha = Image.new("L", og_size, color=0xFF)
            alpha_gradient = ImageDraw.Draw(alpha)
            a = 0
            i = 0
            overlap = self.unicontrol.w.offsetSlider.value()
            shape = (og_size, (0, 0))
            while i < overlap:
                alpha_gradient.rectangle(shape, fill=a)
                a += 4
                i += 1
                shape = ((og_size[0] - i, og_size[1] - i), (i, i))
            mask = Image.new("RGBA", og_size, color=0)
            mask.putalpha(alpha)
            finished_slices = []
            for betterslice, x, y in betterslices:
                finished_slice = addalpha(betterslice, mask)
                finished_slices.append((finished_slice, x, y))
            # # Once we have all our images, use grid_merge back onto the source, then save
            final_output = grid_merge(
                source_image.convert("RGBA"), finished_slices
            ).convert("RGBA")
            final_output.save('output/test_hires.png')
            #base_filename = f"{base_filename}d"
            print(f"All time wasted: {self.sleepytime} seconds.")
            self.hires_source = final_output
            self.deforum_ui.signals.prepare_hires_batch.emit('output/test_hires.png')

    def run_hires_step_x(self, x):
        self.choice = 'Outpaint'
        image = self.canvas.canvas.rectlist[x].image
        image.save('output/temp/temp.png', "PNG")
        self.canvas.canvas.selected_item = self.canvas.canvas.rectlist[x].id


        self.deforum_ui.run_deforum_six_txt2img()
        while self.callbackbusy == True:
            time.sleep(0.25)
            self.sleepytime += 0.25
        time.sleep(0.25)
        self.sleepytime += 0.25
        x += 1

        self.busy = False
        return x

    def run_prepared_outpaint_batch(self, progress_callback=None):
        self.stopprocessing = False
        self.callbackbusy = False
        self.sleepytime = 0.0
        self.choice = "Outpaint"
        self.sessionparams.params.advanced = True

        #multi = self.unicontrol.w.multiBatch.isChecked()
        #batch_n = self.unicontrol.w.multiBatchvalue.value()

        multi = False
        batch_n = 1

        tiles = len(self.canvas.canvas.rectlist)
        print(f"Tiles to Outpaint:{tiles}")

        if multi == True:
            for i in range(batch_n):
                if i != 0:
                    filename = str(random.randint(1111111,9999999))
                    self.canvas.canvas.save_rects_as_json(filename=filename)
                    self.canvas.canvas.save_canvas()
                    self.canvas.canvas.rectlist.clear()
                    self.create_outpaint_batch()
                for x in range(tiles):
                    #print(x)
                    if self.stopprocessing == False:
                        self.run_outpaint_step_x(x)
                    else:
                        break
        else:
            for x in range(tiles):
                if self.stopprocessing == False:
                    print(f"running step {x}")
                    self.run_outpaint_step_x(x)
                else:
                    break
            #self.canvas.canvas.save_canvas()

            print(f"All time wasted: {self.sleepytime} seconds.")

    def run_outpaint_step_x(self, x):

        #print("it should not do anything....")

        self.busy = True
        self.canvas.canvas.reusable_outpaint(self.canvas.canvas.rectlist[x].id)
        while self.canvas.canvas.busy == True:
            time.sleep(0.25)
            self.sleepytime += 0.25
        params = self.canvas.canvas.rectlist[x].params

        self.deforum_ui.run_deforum_outpaint(params)
        while self.callbackbusy == True:
            time.sleep(0.25)
            self.sleepytime += 0.25
        time.sleep(0.25)
        self.sleepytime += 0.25
        x += 1

        self.busy = False
        return x

    def preview_batch_outpaint(self, with_chops=None, chops_y=None):
        if with_chops is None:
            self.canvas.canvas.cols = self.unicontrol.w.batch_columns_slider.value()
            self.canvas.canvas.rows = self.unicontrol.w.batch_rows_slider.value()
        else:
            self.canvas.canvas.cols = with_chops
            self.canvas.canvas.rows = chops_y
        self.canvas.canvas.offset = self.unicontrol.w.rect_overlap.value()
        self.canvas.canvas.maskoffset = self.unicontrol.w.mask_offset.value()
        randomize = self.unicontrol.w.randomize.isChecked()
        spiral = self.unicontrol.w.spiral.isChecked()
        reverse = self.unicontrol.w.reverse.isChecked()
        startOffsetX = self.unicontrol.w.start_offset_x_slider.value()
        startOffsetY = self.unicontrol.w.start_offset_y_slider.value()
        prompts = self.unicontrol.w.prompts.toPlainText()
        #keyframes = self.prompt.w.keyFrames.toPlainText()
        keyframes = ""
        self.canvas.canvas.create_tempBatch(prompts, keyframes, startOffsetX, startOffsetY, randomize)
        templist = []
        if spiral:
            #self.canvas.canvas.tempbatch = random_path(self.canvas.canvas.tempbatch, self.canvas.canvas.cols)
            self.canvas.canvas.tempbatch = spiralOrder(self.canvas.canvas.tempbatch)
        if reverse:
            self.canvas.canvas.tempbatch.reverse()

        self.canvas.canvas.draw_tempBatch(self.canvas.canvas.tempbatch)

    def outpaint_rect_overlap(self):
        self.canvas.canvas.rectPreview = self.unicontrol.w.enable_overlap.isChecked()
        if self.canvas.canvas.rectPreview == False:
            self.canvas.canvas.newimage = True
            self.canvas.canvas.redraw()
        elif self.canvas.canvas.rectPreview == True:
            self.canvas.canvas.visualize_rects()

    def prepare_batch_outpaint_thread(self):
        #self.prompt.w.stopButton.clicked.connect(self.stop_processing)
        self.stopprocessing = False
        #self.save_last_prompt()
        #if self.canvas.canvas.tempbatch == [] or self.canvas.canvas.tempbatch is None:
        #    self.preview_batch_outpaint()
        #    #self.create_outpaint_batch()
        worker = Worker(self.run_batch_outpaint)
        self.threadpool.start(worker)

    @Slot(str)
    def run_create_outpaint_img2img_batch(self, input=None):
        if input != False:
            data = input
        else:
            data = self.getfile()
        self.create_outpaint_batch(gobig_img_path=data)

    def run_prepared_outpaint_batch_thread(self):
        if self.canvas.canvas.rectlist == []:
            self.create_outpaint_batch()
        worker = Worker(self.run_prepared_outpaint_batch)
        self.threadpool.start(worker)

    def run_hires_batch_thread(self):
        worker = Worker(self.run_hires_batch)
        self.threadpool.start(worker)

    def getfile(self, file_ext='', text='', button_caption='', button_type=0, title='Load', save=False):
        filter = {
            '': '',
            'txt': 'File (*.txt)',
            'dbf': 'Table/DBF (*.dbf)',
        }.get(file_ext, '*.' + file_ext)

        filter = QDir.Files
        t = QFileDialog()
        t.setFilter(filter)
        if save:
            t.setAcceptMode(QFileDialog.AcceptSave)
        #t.selectFilter(filter or 'All Files (*.*);;')
        if text:
            (next(x for x in t.findChildren(QLabel) if x.text() == 'File &name:')).setText(text)
        if button_caption:
            t.setLabelText(QFileDialog.Accept, button_caption)
        if title:
            t.setWindowTitle(title)
            t.exec_()
        return t.selectedFiles()[0]

    def list_files(self, index=0):
        paths = []
        self.unicontrol.w.aesthetic_embedding.clear()
        self.unicontrol.w.aesthetic_embedding.addItem("None")
        for _, _, files in os.walk(gs.system.aesthetic_gradients):
            for file in files:
                self.unicontrol.w.aesthetic_embedding.addItem(str(file))
        #self.set_txt2img.w.gradientList.setItemText(index)

    def select_gradient(self, gradient):
        if self.unicontrol.w.aesthetic_embedding.itemText(gradient) != "None":
            gs.aesthetic_embedding_path = os.path.join(gs.system.aesthetic_gradients, self.unicontrol.w.aesthetic_embedding.itemText(gradient))
        else:
            gs.aesthetic_embedding_path = None

    #Timeline functions
    def showTypeKeyframes(self):
        valueType = self.animKeyEditor.w.comboBox.currentText()

        self.timeline.timeline.selectedValueType = valueType
        self.updateAnimKeys()
        self.timeline.timeline.update()

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
                self.unicontrol.w.contrast_schedule.setText(tempString)
            if "Noise" in selection:
                self.unicontrol.w.noise_schedule.setText(tempString)
            if "Strength" in selection:
                self.unicontrol.w.strength_schedule.setText(tempString)
            if "Rotation X" in selection:
                self.unicontrol.w.rotation_3d_x.setText(tempString)
            if "Rotation Y" in selection:
                self.unicontrol.w.rotation_3d_y.setText(tempString)
            if "Rotation Z" in selection:
                self.unicontrol.w.rotation_3d_z.setText(tempString)
            if "Translation X" in selection:
                self.unicontrol.w.translation_x.setText(tempString)
            if "Translation Y" in selection:
                self.unicontrol.w.translation_y.setText(tempString)
            if "Translation Z" in selection:
                self.unicontrol.w.translation_z.setText(tempString)
            if "Angle" in selection:
                self.unicontrol.w.angle.setText(tempString)
            if "Zoom" in selection:
                self.unicontrol.w.zoom.setText(tempString)

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

    def update_timeline(self):
        self.timeline.timeline.duration = self.animSliders.w.frames.value()
        self.timeline.timeline.update()


def QIcon_from_svg(svg_filepath, color='white'):
    img = QPixmap(svg_filepath)
    qp = QPainter(img)
    qp.setCompositionMode(QPainter.CompositionMode_SourceIn)
    qp.fillRect( img.rect(), QColor(color) )
    qp.end()
    return QIcon(img)

def translate_sampler_index(index):
    if index == 0:
        return "euler"

def translate_sampler(sampler):
    if sampler == "K Euler":
        return "euler"
def get_inbetweens(key_frames, max_frames, integer=False, interp_method='Linear'):
    import numexpr
    key_frame_series = pd.Series([np.nan for a in range(max_frames)])

    for i in range(0, max_frames):
        if i in key_frames:
            value = key_frames[i]
            value_is_number = check_is_number(value)

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
def addalpha(im, mask):
    imr, img, imb, ima = im.split()
    mmr, mmg, mmb, mma = mask.split()
    im = Image.merge(
        "RGBA", [imr, img, imb, mma]
    )  # we want the RGB from the original, but the transparency from the mask
    return im


# Alternative method composites a grid of images at the positions provided
def grid_merge(source, slices):
    source.convert("RGBA")
    for slice, posx, posy in slices:  # go in reverse to get proper stacking
        source.alpha_composite(slice, (posx, posy))
    return source
