import base64
import json
import os
from io import BytesIO

import backend.settings as settings
from backend.singleton import singleton

gs = singleton
settings.load_settings_json()
if gs.system.custom_cache_dir == True:
    os.makedirs(gs.system.cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = gs.system.cache_dir
import time
import random
from datetime import datetime
from uuid import uuid4
import numpy as np
import pandas as pd
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QEasingCurve, Slot, QThreadPool, QDir, Signal, QObject
from PySide6.QtWidgets import QMainWindow, QToolBar, QListWidgetItem, QFileDialog, \
    QLabel
from PySide6.QtGui import QAction, QIcon, QColor, QPixmap, QPainter, Qt, QShortcut, QKeySequence
from PySide6 import QtCore
from backend.deforum.six.animation import check_is_number
from einops import rearrange
import copy
import torchvision.transforms as T

from backend.worker import Worker
from frontend import plugin_loader
# from frontend.ui_model_chooser import ModelChooser_UI

from backend.prompt_ai.prompt_gen import AiPrompt
from frontend.ui_paint import PaintUI
from frontend.ui_classes import SystemSetup, ThumbsUI, AnimKeyEditor
from frontend.unicontrol import UniControl

from frontend.ui_krea import Krea
from frontend.ui_lexica import LexicArt
from frontend.ui_model_download import ModelDownload
from backend.shared import model_killer

from frontend.ui_timeline import Timeline, KeyFrame

# we had to load settings first before we can do this import
from frontend.ui_prompt_fetcher import PromptFetcher_UI, FetchPrompts
from frontend.ui_image_lab import ImageLab
from frontend.ui_deforum import Deforum_UI, draw_grid_annotations
from frontend.session_params import SessionParams
from backend.shared import save_last_prompt
from backend.maintain_models import check_models_exist
from backend.sqlite import db_base
from backend.sqlite import model_db_civitai
from backend.web_requests.web_images import WebImages
from frontend.ui_outpaint import Outpainting

# please don't remove it totally, just remove what we know is not used
class Callbacks(QObject):
    txt2img_step = Signal()
    reenable_runbutton = Signal()
    txt2img_image_cb = Signal()
    deforum_step = Signal()
    deforum_image_cb = Signal()
    compviscallback = Signal()
    add_image_to_thumbnail_signal = Signal(str)
    status_update = Signal(str)
    image_ready = Signal()
    image_ready_op = Signal()
    vid2vid_one_percent = Signal(int)
    set_prompt = Signal(str)
    image_loaded = Signal(str, str, tuple, object)






class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.signals = Callbacks()

        self.thumbs = ThumbsUI()
        self.canvas = PaintUI(self)
        self.setCentralWidget(self.canvas)

        self.setWindowTitle("aiNodes - Still Mode")
        #self.timeline = Timeline(self)
        self.animKeyEditor = AnimKeyEditor()

        self.resize(1280, 800)

        self.widgets = {}
        self.current_widget = 'unicontrol'
        self.widgets[self.current_widget] = UniControl(self)
        self.outpaint = Outpainting(self)



        self.load_last_prompt()

        self.sessionparams = SessionParams(self)
        self.sessionparams.create_diffusion_params()
        self.sessionparams.create_system_params()


        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.thumbs.w.dockWidget)

        self.create_main_toolbar()
        self.create_secondary_toolbar()
        self.system_setup = SystemSetup()
        self.sessionparams.add_state_to_history()
        self.update_ui_from_params()
        self.update_ui_from_system_params()
        self.currentFrames = []
        self.renderedFrames = 0

        self.threadpool = QThreadPool()
        self.deforum_ui = Deforum_UI(self)
        self.w = 512
        self.cheight = 512
        self.setAcceptDrops(True)
        self.y = 0
        self.lastheight = None
        self.cheight = gs.diffusion.H

        self.lexicart = LexicArt(self)
        self.krea = Krea(self)
        self.prompt_fetcher = FetchPrompts()
        self.prompt_fetcher_ui = PromptFetcher_UI(self)

        self.image_lab = ImageLab(self)
        self.image_lab_ui = self.image_lab.imageLab
        self.model_download = ModelDownload(self)
        self.model_download_ui = self.model_download.model_download
        self.model_download.maintain_custom_models()
        self.widgets[self.current_widget].w.dockWidget.setWindowTitle("Parameters")
        self.system_setup.w.dockWidget.setWindowTitle("System Settings")
        self.image_lab_ui.w.dockWidget.setWindowTitle("Image Lab")
        self.lexicart.w.dockWidget.setWindowTitle("Lexica Art")
        self.krea.w.dockWidget.setWindowTitle("Krea")
        self.prompt_fetcher.w.dockWidget.setWindowTitle("Prompt Fetcher")
        self.model_download_ui.w.dockWidget.setWindowTitle("Model Download")
        self.thumbs.w.dockWidget.setWindowTitle("Outpaint rectangle history")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.model_download_ui.w.dockWidget)
        self.model_download_ui.w.dockWidget.setMaximumHeight(self.height())
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.image_lab_ui.w.dockWidget)
        self.image_lab_ui.w.dockWidget.setMaximumHeight(self.height())
        self.tabifyDockWidget(self.model_download_ui.w.dockWidget, self.image_lab_ui.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.system_setup.w.dockWidget)
        self.system_setup.w.dockWidget.setMaximumHeight(self.height())
        self.tabifyDockWidget(self.image_lab_ui.w.dockWidget, self.system_setup.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.lexicart.w.dockWidget)
        self.lexicart.w.dockWidget.setMaximumHeight(self.height())
        self.tabifyDockWidget(self.system_setup.w.dockWidget, self.lexicart.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.krea.w.dockWidget)
        self.krea.w.dockWidget.setMaximumHeight(self.height())
        self.tabifyDockWidget(self.lexicart.w.dockWidget, self.krea.w.dockWidget)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.prompt_fetcher.w.dockWidget)
        self.tabifyDockWidget(self.krea.w.dockWidget, self.prompt_fetcher.w.dockWidget)
        self.tabifyDockWidget(self.prompt_fetcher.w.dockWidget, self.widgets[self.current_widget].w.dockWidget)
        self.civitai_api = model_db_civitai.civit_ai_api()
        self.web_images = WebImages()

        self.mode = 'txt2img'
        self.stopwidth = False

        self.init_plugin_loader()
        self.connections()
        self.resize(1280, 800)
        self.create_sys_folders()

        self.widgets[self.current_widget].update_model_list()
        self.widgets[self.current_widget].update_vae_list()
        self.widgets[self.current_widget].update_hypernetworks_list()
        self.widgets[self.current_widget].update_aesthetics_list()

        check_models_exist()
        self.latent_rgb_factors = torch.tensor([
            #   R        G        B
            [0.298, 0.207, 0.208],  # L1
            [0.187, 0.286, 0.173],  # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ], dtype=torch.float, device='cuda')

        db_base.check_db_status()
        self.params = self.sessionparams.update_params(update_db=False)

        self.sessionparams.update_system_params()
        self.hide_default()
        if gs.system.show_settings is True:
            self.show_default()
        self.check_karras_enabled()
        self.make_grid = False
        self.all_images = []
        self.advanced_temp = False
        self.gpu_info()
        self.shortcut = QShortcut(QKeySequence("Ctrl+Enter"), self)
        self.shortcut.activated.connect(self.task_switcher)

    def signal_handler(signal, frame):
        print("Signal received:", signal)
    def selftest(self):  # TODO Lets extend this function with everything we have and has to work

        self.canvas.canvas.reset()
        self.params = self.sessionparams.update_params()
        gs.stop_all = False
        self.task_switcher()
        self.params.max_frames = 5
        self.task_switcher()
        self.params.max_frames = 1
        self.add_next_rect()
        self.params.advanced = True
        self.task_switcher()
        self.params.advanced = False
        self.task_switcher()
        self.widgets[self.current_widget].w.with_inpaint.setCheckState(Qt.Checked)
        self.canvas.canvas.addrect_atpos(prompt="", x=1750, y=0, w=512, h=512, params=copy.deepcopy(self.params))
        self.task_switcher()

    def create_sys_folders(self):
        os.makedirs(gs.system.out_dir, exist_ok=True)
        os.makedirs(gs.system.txt2img_out_dir, exist_ok=True)
        os.makedirs(gs.system.img2img_tmp_dir, exist_ok=True)
        os.makedirs(gs.system.img2img_out_dir, exist_ok=True)
        os.makedirs(gs.system.txt2vid_single_frame_dir, exist_ok=True)
        os.makedirs(gs.system.txt2vid_out_dir, exist_ok=True)
        os.makedirs(gs.system.vid2vid_tmp_dir, exist_ok=True)
        os.makedirs(gs.system.vid2vid_single_frame_dir, exist_ok=True)
        os.makedirs(gs.system.vid2vid_out_dir, exist_ok=True)
        os.makedirs(gs.system.custom_models_dir, exist_ok=True)
        os.makedirs(gs.system.default_config_yaml_dir, exist_ok=True)
        os.makedirs(gs.system.vae_dir, exist_ok=True)
        os.makedirs(gs.system.textual_inversion_dir, exist_ok=True)
        os.makedirs(gs.system.hypernetwork_dir, exist_ok=True)
        os.makedirs(gs.system.aesthetic_gradients_dir, exist_ok=True)

    def connections(self):

        self.signals.set_prompt.connect(self.widgets[self.current_widget].set_prompt)

        self.deforum_ui.signals.txt2img_image_cb.connect(self.image_preview_func_str)
        self.deforum_ui.signals.deforum_step.connect(self.tensor_preview_schedule)
        self.deforum_ui.signals.plot_ready.connect(self.deforum_ui.plot_ready)
        self.widgets[self.current_widget].w.dream.clicked.connect(self.task_switcher)
        self.widgets[self.current_widget].w.lucky.clicked.connect(self.show_default)

        #canvas connections
        self.canvas.W.valueChanged.connect(self.canvas.canvas.change_resolution)
        self.canvas.H.valueChanged.connect(self.canvas.canvas.change_resolution)
        self.canvas.canvas.signals.outpaint_signal.connect(self.deforum_ui.deforum_outpaint_thread)
        self.canvas.canvas.signals.txt2img_signal.connect(self.deforum_ui.deforum_six_txt2img_thread)
        self.widgets[self.current_widget].w.H.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)
        self.widgets[self.current_widget].w.W.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)

        #outpaint connections
        self.outpaint.signals.rect_ready_in_ui.connect(self.outpaint.rect_ready_in_ui)
        self.deforum_ui.signals.image_ready_in_ui.connect(self.outpaint.image_ready_in_ui)
        self.widgets[self.current_widget].w.redo.clicked.connect(self.outpaint.redo_current_outpaint)
        self.widgets[self.current_widget].w.delete_2.clicked.connect(self.outpaint.delete_outpaint_frame)
        self.widgets[self.current_widget].w.preview_batch.clicked.connect(self.outpaint.preview_batch_outpaint_thread)
        self.widgets[self.current_widget].w.resize_canvas.clicked.connect(self.outpaint.resize_canvas)
        self.widgets[self.current_widget].w.prepare_batch.clicked.connect(self.outpaint.prepare_batch_outpaint_thread)
        self.widgets[self.current_widget].w.run_batch.clicked.connect(self.outpaint.run_prepared_outpaint_batch_thread)
        self.widgets[self.current_widget].w.run_hires.clicked.connect(self.outpaint.run_hires_batch_thread)
        self.widgets[self.current_widget].w.prep_hires.clicked.connect(self.outpaint.run_create_outpaint_img2img_batch)
        self.widgets[self.current_widget].w.update_params.clicked.connect(self.outpaint.update_params)
        self.widgets[self.current_widget].w.W.valueChanged.connect(self.outpaint.update_outpaint_parameters)
        self.widgets[self.current_widget].w.H.valueChanged.connect(self.outpaint.update_outpaint_parameters)
        self.widgets[self.current_widget].w.mask_offset.valueChanged.connect(self.outpaint.outpaint_offset_signal)
        # self.widgets[self.current_widget].w.mask_offset.valueChanged.connect(self.canvas.canvas.set_offset(int(self.widgets[self.current_widget].w.mask_offset.value())))  # todo does this work?
        self.widgets[self.current_widget].w.rect_overlap.valueChanged.connect(self.outpaint.outpaint_rect_overlap)
        self.thumbs.w.thumbnails.itemClicked.connect(self.outpaint.select_outpaint_image)
        self.outpaint.signals.add_rect.connect(self.outpaint.add_rect)
        self.outpaint.signals.canvas_update.connect(self.outpaint.canvas_update)
        self.outpaint.signals.txt2img_image_op.connect(self.image_preview_func_str_op)
        self.canvas.canvas.signals.update_selected.connect(self.outpaint.show_outpaint_details)
        self.canvas.canvas.signals.update_params.connect(self.outpaint.create_params)

        self.widgets[self.current_widget].w.load_model.clicked.connect(
            self.deforum_ui.deforum_six.load_model_from_config)
        self.widgets[self.current_widget].w.load_inpaint_model.clicked.connect(
            self.deforum_ui.deforum_six.load_inpaint_model)
        self.widgets[self.current_widget].w.cleanup_memory.clicked.connect(model_killer)


        self.widgets[self.current_widget].w.selected_model.currentIndexChanged.connect(self.select_new_model)

        #self.timeline.timeline.keyFramesUpdated.connect(self.updateKeyFramesFromTemp)
        #self.animKeyEditor.w.comboBox.currentTextChanged.connect(self.showTypeKeyframes)
        #self.animKeyEditor.w.keyButton.clicked.connect(self.addCurrentFrame)

        #image labs connections
        self.image_lab.signals.upscale_start.connect(self.image_lab.upscale_start)
        self.image_lab.signals.upscale_stop.connect(self.image_lab.upscale_stop)
        self.image_lab.signals.upscale_counter.connect(self.image_lab.upscale_count)
        self.image_lab.signals.img_to_txt_start.connect(self.image_lab.img_to_text_start)
        self.image_lab.signals.image_text_ready.connect(self.image_lab.set_image_text)
        self.image_lab.signals.watermark_start.connect(self.image_lab.watermark_start)
        self.image_lab.signals.model_merge_start.connect(self.image_lab.model_merge_start_thread)
        self.image_lab.signals.ebl_model_merge_start.connect(self.image_lab.ebl_model_merge_start_thread)
        self.image_lab.signals.run_aestetic_prediction.connect(self.image_lab.run_aestetic_prediction_thread)
        self.image_lab.signals.run_interrogation.connect(self.image_lab.run_interrogation_thread)
        self.image_lab.signals.run_volta_accel.connect(self.image_lab.run_volta_accel_thread)
        self.image_lab.signals.run_upscale_20.connect(self.image_lab.run_upscale_20_thread)
        self.image_lab.signals.crop_image.connect(self.image_lab.crop_image)
        self.image_lab.signals.show_crop_image.connect(self.image_lab.show_crop_image)
        self.image_lab.signals.set_crop_image_scale.connect(self.image_lab.set_crop_image_scale)


        self.prompt_fetcher_ui.signals.run_ai_prompt.connect(self.ai_prompt_thread)
        self.prompt_fetcher_ui.signals.run_img_to_prompt.connect(self.prompt_fetcher_ui.image_to_prompt_thread)
        self.prompt_fetcher_ui.signals.get_lexica_prompts.connect(self.prompt_fetcher_ui.get_lexica_prompts_thread)
        self.prompt_fetcher_ui.signals.got_image_to_prompt.connect(self.prompt_fetcher_ui.set_img_to_prompt_text)
        self.prompt_fetcher_ui.signals.got_lexica_prompts.connect(self.prompt_fetcher_ui.set_lexica_prompts)
        self.prompt_fetcher_ui.signals.get_krea_prompts.connect(self.prompt_fetcher_ui.get_krea_prompts_thread)
        self.prompt_fetcher_ui.signals.got_krea_prompts.connect(self.prompt_fetcher_ui.set_krea_prompts)

        self.model_download.signals.startDownload.connect(self.model_download.download_model_thread)
        self.model_download.signals.set_download_percent.connect(self.model_download.model_download_progress_callback_signal)

        self.model_download.civit_ai_api.signals.civitai_no_more_models.connect(self.all_civitai_model_data_loaded_thread)
        self.model_download.civit_ai_api.signals.civitai_start_model_update.connect(self.civitai_start_model_update_thread)
        self.model_download.signals.show_model_preview_images.connect(self.show_model_preview_images)
        self.model_download.signals.add_model_search_item.connect(self.model_download.add_model_search_item)
        self.model_download.signals.set_download_info_text.connect(self.model_download.set_download_info_text)

        self.system_setup.w.ok.clicked.connect(self.sessionparams.update_system_params)
        self.system_setup.w.cancel.clicked.connect(self.update_ui_from_system_params)
        self.web_images.signals.web_image_retrived.connect(self.show_web_image_on_canvas)

        self.widgets[self.current_widget].w.sampler.currentIndexChanged.connect(self.check_karras_enabled)
        self.widgets[self.current_widget].w.hires.toggled.connect(self.set_hires_strength_visablity)

        self.canvas.canvas.signals.run_redraw.connect(self.run_redraw)
        self.canvas.canvas.signals.draw_tempRects.connect(self.draw_tempRects_signal)
        self.signals.status_update.connect(self.set_status_bar)
        self.signals.image_loaded.connect(self.render_index_image_preview_func_str)
        self.system_setup.w.update_gpu_stats.clicked.connect(self.gpu_info)

    def gpu_info(self):
        cuda_info = ''
        mem_info = ''
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                cuda_info = f"""{cuda_info}GPU number: {i}
{props.name}
total_memory: {props.total_memory/(1024*1024)}MB                
multiprocessor count: {props.multi_processor_count}
major: {props.major}
minor: {props.minor}
arch list: {torch.cuda.get_arch_list()}
allocated_memory: {torch.cuda.memory_allocated()}
max_allocated_memory: {torch.cuda.max_memory_allocated()}
"""
                mem_info = f"""{mem_info}GPU number: {i}\n"""
                memory_stats = torch.cuda.memory_stats()
                for key in gs.models:
                    mem_info = mem_info + f"model {key}: {int(torch.cuda.memory_allocated(gs.models[key])/(1024*1024))}MB\n"
                for key, value in memory_stats.items():
                    mem_info = mem_info + f"{key}: {value}\n"
            mem_info = mem_info + '\n'
        else:
            cuda_info = 'CUDA is not available.'
        self.system_setup.w.gpu_info.setPlainText(cuda_info)
        self.system_setup.w.gpu_stats.setPlainText(mem_info)


    def run_redraw(self):
        self.canvas.canvas.redraw_signal()

    def set_hires_strength_visablity(self):
        visable = self.widgets[self.current_widget].w.hires.isChecked()
        self.widgets[self.current_widget].w.hires_strength_label.setVisible(visable)
        self.widgets[self.current_widget].w.hires_strength.setVisible(visable)

    def check_karras_enabled(self):
        if self.widgets[self.current_widget].w.sampler.currentText() != 'ddim' and self.widgets[self.current_widget].w.sampler.currentText() != 'plms':
            self.widgets[self.current_widget].w.karras_switches.setVisible(True)
            self.widgets[self.current_widget].w.ddim_eta_label.setVisible(False)
            self.widgets[self.current_widget].w.ddim_eta.setVisible(False)
        else:
            self.widgets[self.current_widget].w.karras_switches.setVisible(False)
            if self.widgets[self.current_widget].w.sampler.currentText() == 'ddim':
                self.widgets[self.current_widget].w.ddim_eta_label.setVisible(True)
                self.widgets[self.current_widget].w.ddim_eta.setVisible(True)


    def select_new_model(self):
        if self.widgets[self.current_widget].w.preview_on_canvas.isChecked():
            if 'custom/' in self.widgets[self.current_widget].w.selected_model.currentText():
                custom_model_info = self.civitai_api.civitai_get_model_data(self.widgets[self.current_widget].w.selected_model.currentText().replace('custom/',''))
                if len(custom_model_info) > 0:
                    custom_model_info = custom_model_info[0]
                    images = custom_model_info['images']
                    images =  json.loads(images)
                    for image in images:
                        self.show_image_from_url(image['url'])

        if 'custom/' in self.widgets[self.current_widget].w.selected_model.currentText():
            custom_model_info = self.civitai_api.civitai_get_model_data(self.widgets[self.current_widget].w.selected_model.currentText().replace('custom/',''))
            if len(custom_model_info) > 0:
                custom_model_info = custom_model_info[0]
                self.widgets[self.current_widget].w.selected_model.setToolTip(custom_model_info['description'])
                self.widgets[self.current_widget].w.prompts.setPlaceholderText(custom_model_info['trained_words'])
            else:
                self.widgets[self.current_widget].w.selected_model.setToolTip('')
                self.widgets[self.current_widget].w.prompts.setPlaceholderText('Enter your prompt here')
        else:
            self.widgets[self.current_widget].w.selected_model.setToolTip('')
            self.widgets[self.current_widget].w.prompts.setPlaceholderText('Enter your prompt here')

    @Slot()
    def civitai_start_model_update_thread(self):
        worker = Worker(self.model_download.civit_ai_api.civitai_start_model_update)
        self.threadpool.start(worker)

    @Slot()
    def all_civitai_model_data_loaded_thread(self):
        worker = Worker(self.model_download.civit_ai_api.all_civitai_model_data_loaded)
        self.threadpool.start(worker)

    def task_switcher(self):
        gs.stop_all = False
        if not self.widgets[self.current_widget].w.toggle_animations.isChecked():
            self.widgets[self.current_widget].w.max_frames.setValue(0)

        save_last_prompt(self.widgets[self.current_widget].w.prompts.toHtml(),
                         self.widgets[self.current_widget].w.prompts.toPlainText())
        if self.widgets[self.current_widget].w.with_inpaint.isChecked() == True:
            self.params = self.sessionparams.update_params()
            self.params.advanced = True
            self.canvas.canvas.reusable_outpaint(self.canvas.canvas.selected_item)
            self.deforum_ui.deforum_outpaint_thread()
        else:
            if self.widgets[self.current_widget].w.toggle_outpaint.isChecked():
                pass
                self.deforum_ui.deforum_six_txt2img_outpaint_thread()
            else:
                self.deforum_ui.deforum_six_txt2img_thread()

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
    def set_status_bar(self, txt):
        self.statusBar().showMessage(txt)

    @Slot()
    def ai_prompt_thread(self):
        self.aiPrompt = AiPrompt()
        self.aiPrompt.signals.ai_prompt_ready.connect(self.prompt_fetcher_ui.set_ai_prompt)
        self.aiPrompt.signals.status_update.connect(self.set_status_bar)
        worker = Worker(self.aiPrompt.get_prompts, self.prompt_fetcher.w.input.toPlainText())
        self.threadpool.start(worker)

    def run_as_thread(self, fn):
        worker = Worker(fn)
        self.threadpool.start(worker)

    def translate_settings_values(self, key, value):
        if key == 'sampler':
            value = self.sessionparams.reverse_translate_sampler(value)
        if key == 'axis':
            if value == {'x'}:
                value = 'X'
            elif value == {'y'}:
                value = 'Y'
            elif value == {'x', 'y'}:
                value = 'Both'
        return value

    def update_ui_from_params(self):
        current_widget = self.widgets[self.current_widget].w
        for key, value in self.sessionparams.params.items():
            try:
                #print(f'key {key} value {value} type {type(value)}' )
                if type(value) == set:
                    value = self.translate_settings_values(key, value)
                # We have to add check for Animation Mode as thats a radio checkbox with values 'anim2d', 'anim3d', 'animVid'
                # add colormatch_image (it will be with a fancy preview)
                obj_type = str(getattr(self.widgets[self.current_widget].w, key))

                if 'QSpinBox' in obj_type or 'QDoubleSpinBox' in obj_type:
                    getattr(self.widgets[self.current_widget].w, key).setValue(value)
                elif 'QTextEdit' in obj_type or 'QLineEdit' in obj_type:
                    getattr(self.widgets[self.current_widget].w, key).setText(str(value))
                elif 'QCheckBox' in obj_type or 'QRadioButton' in obj_type:
                    if value == True:
                        getattr(self.widgets[self.current_widget].w, key).setCheckState(QtCore.Qt.Checked)
                elif 'QComboBox' in obj_type:
                    if type(value) == str:
                        value = self.translate_settings_values(key, value)
                        item_count = getattr(current_widget, key).count()
                        items = []
                        for i in range(0, item_count):
                            items.append(getattr(current_widget, key).itemText(i))
                        if item_count > 0:
                            getattr(current_widget, key).setCurrentIndex(items.index(value))
                        else:
                            getattr(current_widget, key).setCurrentIndex(0)
                    elif type(value) == int:
                        getattr(current_widget, key).setCurrentIndex(value)
                    else:
                        print(f'unknown type for combobox {key} {type(value)}: {value}')
            except Exception as e:
                #print(f'setting still to be fixed {key} {value}', e)
                continue

    def update_ui_from_system_params(self):
        for key, value in self.sessionparams.system_params.items():
            try:
                current_widget = self.system_setup.w
                obj_type = str(getattr(current_widget, key))

                if 'QSpinBox' in obj_type or 'QDoubleSpinBox' in obj_type:
                    getattr(current_widget, key).setValue(value)
                elif 'QTextEdit' in obj_type or 'QLineEdit' in obj_type:
                    getattr(current_widget, key).setText(str(value))
                elif 'QCheckBox' in obj_type or 'QRadioButton' in obj_type:
                    if value == True:
                        getattr(current_widget, key).setCheckState(QtCore.Qt.Checked)
                elif 'QComboBox' in obj_type:
                    if type(value) == str:
                        value = self.translate_settings_values(key, value)
                        item_count = getattr(current_widget, key).count()
                        items = []
                        for i in range(0, item_count):
                            items.append(getattr(current_widget, key).itemText(i))
                        if item_count > 0:
                            getattr(current_widget, key).setCurrentIndex(items.index(value))
                        else:
                            getattr(current_widget, key).setCurrentIndex(0)
                    elif type(value) == int:
                        getattr(current_widget, key).setCurrentIndex(value)
                    else:
                        print(f'unknown type for combobox {type(value)}: {value}')

            except Exception as e:
                continue

    def init_plugin_loader(self):
        self.widgets[self.current_widget].w.loadbutton.clicked.connect(self.load_plugin)
        self.widgets[self.current_widget].w.unloadbutton.clicked.connect(self.unload_plugin)
        self.plugins = plugin_loader.PluginLoader(self)
        list = self.plugins.list_plugins()
        for i in list:
            self.widgets[self.current_widget].w.plugins.addItem(i)

    def load_plugin(self):
        plugin_name = self.widgets[self.current_widget].w.plugins.currentText()
        self.plugins.load_plugin(f"plugins.{plugin_name}.{plugin_name}")

    def unload_plugin(self):
        plugin_name = self.widgets[self.current_widget].w.plugins.currentText()
        self.plugins.unload_plugin(plugin_name)

    def create_main_toolbar(self):
        self.toolbar = QToolBar('Outpaint Tools')
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.toolbar)
        skip_back = QAction(QIcon_from_svg('frontend/icons/skip-back.svg'), 'back', self)
        play = QAction(QIcon_from_svg('frontend/icons/play.svg'), 'Enable Playback / Play All', self)
        stop = QAction(QIcon_from_svg('frontend/icons/square.svg'), 'Stop All', self)
        skip_forward = QAction(QIcon_from_svg('frontend/icons/skip-forward.svg'), 'forward', self)
        clear_canvas = QAction(QIcon_from_svg('frontend/icons/frown.svg'), 'Clear Canvas', self)
        #test_mode = QAction(QIcon_from_svg('frontend/icons/alert-octagon.svg'), 'Run Self Test - It will take a while', self)
        #help_mode = QAction(QIcon_from_svg('frontend/icons/help-circle.svg'), 'Help', self)
        #still_mode = QAction(QIcon_from_svg('frontend/icons/instagram.svg'), 'Still', self)

        self.toolbar.addAction(skip_back)
        self.toolbar.addAction(skip_forward)
        self.toolbar.addAction(play)
        self.toolbar.addAction(stop)
        self.toolbar.addAction(clear_canvas)
        #self.toolbar.addAction(still_mode)
        #self.toolbar.addAction(test_mode)

        skip_back.triggered.connect(self.canvas.canvas.skip_back)
        skip_forward.triggered.connect(self.canvas.canvas.skip_forward)
        play.triggered.connect(self.canvas.canvas.start_main_clock)
        stop.triggered.connect(self.canvas.canvas.stop_main_clock)
        clear_canvas.triggered.connect(self.canvas.canvas.reset)
        #test_mode.triggered.connect(self.selftest)

    def create_secondary_toolbar(self):
        self.secondary_toolbar = QToolBar('Outpaint Tools')
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.secondary_toolbar)
        select_mode = QAction(QIcon_from_svg('frontend/icons/mouse-pointer.svg'), 'Select Rect', self)
        drag_mode = QAction(QIcon_from_svg('frontend/icons/wind.svg'), 'Drag Canvas', self)
        add_mode = QAction(QIcon_from_svg('frontend/icons/plus.svg'), 'Outpaint', self)
        inpaint_mode = QAction(QIcon_from_svg('frontend/icons/edit.svg'), 'Inpaint', self)
        inpaint_current = QAction(QIcon_from_svg('frontend/icons/edit.svg'), 'Inpaint Current Frame', self)
        move_mode = QAction(QIcon_from_svg('frontend/icons/move.svg'), 'Move', self)
        save_canvas = QAction(QIcon_from_svg('frontend/icons/file-text.svg'), 'Save as Json', self)
        save_canvas_png = QAction(QIcon_from_svg('frontend/icons/save.svg'), 'Save as PNG', self)

        load_canvas = QAction(QIcon_from_svg('frontend/icons/folder.svg'), 'Load from Json', self)
        load_image = QAction(QIcon_from_svg('frontend/icons/folder.svg'), 'Load Image', self)


        self.secondary_toolbar.addAction(select_mode)
        self.secondary_toolbar.addAction(drag_mode)
        self.secondary_toolbar.addAction(add_mode)
        self.secondary_toolbar.addAction(inpaint_mode)
        self.secondary_toolbar.addAction(inpaint_current)
        self.secondary_toolbar.addAction(move_mode)
        self.secondary_toolbar.addSeparator()
        self.secondary_toolbar.addAction(save_canvas)
        self.secondary_toolbar.addAction(save_canvas_png)
        self.secondary_toolbar.addAction(load_canvas)
        self.secondary_toolbar.addAction(load_image)

        self.secondary_toolbar.addSeparator()


        select_mode.triggered.connect(self.canvas.canvas.select_mode)
        drag_mode.triggered.connect(self.canvas.canvas.drag_mode)
        add_mode.triggered.connect(self.canvas.canvas.add_mode)
        inpaint_mode.triggered.connect(self.canvas.canvas.inpaint_mode)
        inpaint_current.triggered.connect(self.canvas.canvas.inpaint_current_frame)
        move_mode.triggered.connect(self.canvas.canvas.move_mode)
        save_canvas.triggered.connect(self.canvas.canvas.save_rects_as_json)
        load_canvas.triggered.connect(self.canvas.canvas.load_rects_from_json)
        load_image.triggered.connect(self.canvas.canvas.load_img_into_rect)

        save_canvas_png.triggered.connect(self.canvas.canvas.save_canvas)

    def hide_default(self):

        self.toolbar.setVisible(False)
        self.secondary_toolbar.setVisible(False)

        self.widgets[self.current_widget].w.base_setup.setVisible(False)
        self.widgets[self.current_widget].w.advanced_toppics.setVisible(False)
        self.widgets[self.current_widget].w.negative_prompts.setVisible(False)
        self.widgets[self.current_widget].w.keyframes.setVisible(False)
        self.widgets[self.current_widget].w.axis.setVisible(False)
        self.system_setup.w.dockWidget.setVisible(False)
        self.animKeyEditor.w.dockWidget.setVisible(False)
        self.image_lab_ui.w.dockWidget.setVisible(False)
        self.lexicart.w.dockWidget.setVisible(False)
        self.krea.w.dockWidget.setVisible(False)
        self.prompt_fetcher.w.dockWidget.setVisible(False)
        self.model_download_ui.w.dockWidget.setVisible(False)
        self.widgets[self.current_widget].w.cleanup_memory.setVisible(False)


        self.widgets[self.current_widget].w.preview_mode_label.setVisible(False)
        self.widgets[self.current_widget].w.preview_mode.setVisible(False)
        self.widgets[self.current_widget].w.stop_dream.setVisible(False)
        self.widgets[self.current_widget].w.hires.setVisible(False)
        self.widgets[self.current_widget].w.hires_strength_label.setVisible(False)
        self.widgets[self.current_widget].w.hires_strength.setVisible(False)
        self.widgets[self.current_widget].w.seamless.setVisible(False)



        if self.widgets[self.current_widget].samHidden == False:
            self.widgets[self.current_widget].hideSampler_anim()
        if self.widgets[self.current_widget].aesHidden == False:
            self.widgets[self.current_widget].hideAesthetic_anim()
        if self.widgets[self.current_widget].aniHidden == False:
            self.widgets[self.current_widget].hideAnimation_anim()

        if self.widgets[self.current_widget].ploHidden == False:
            self.widgets[self.current_widget].hidePlotting_anim()

        self.thumbs.w.dockWidget.setVisible(False)

        self.default_hidden = True

    def show_default(self):
        if self.default_hidden == True:
            if gs.system.show_settings != True:
                gs.system.show_settings = True
                self.sessionparams.update_system_params()
            self.toolbar.setVisible(True)

            self.widgets[self.current_widget].w.base_setup.setVisible(True)
            self.widgets[self.current_widget].w.advanced_toppics.setVisible(True)
            self.widgets[self.current_widget].w.negative_prompts.setVisible(True)
            self.system_setup.w.dockWidget.setVisible(True)
            self.image_lab_ui.w.dockWidget.setVisible(True)
            self.lexicart.w.dockWidget.setVisible(True)
            self.krea.w.dockWidget.setVisible(True)
            self.prompt_fetcher.w.dockWidget.setVisible(True)

            self.animKeyEditor.w.dockWidget.setVisible(True)
            self.model_download_ui.w.dockWidget.setVisible(True)
            self.widgets[self.current_widget].w.cleanup_memory.setVisible(True)

            self.widgets[self.current_widget].w.preview_mode_label.setVisible(True)
            self.widgets[self.current_widget].w.stop_dream.setVisible(True)
            self.widgets[self.current_widget].w.preview_mode.setVisible(True)
            self.widgets[self.current_widget].w.hires.setVisible(True)
            self.widgets[self.current_widget].w.seamless.setVisible(True)

            self.set_hires_strength_visablity()
            self.default_hidden = False
        else:
            if gs.system.show_settings != False:
                gs.system.show_settings = False
                self.sessionparams.update_system_params()
            self.hide_default()

    def thumbnails_Animation(self):
        self.thumbsShow = QtCore.QPropertyAnimation(self.thumbnails, b"maximumHeight")
        self.thumbsShow.setDuration(2000)
        self.thumbsShow.setStartValue(0)
        self.thumbsShow.setEndValue(self.cheight() / 4)
        self.thumbsShow.setEasingCurve(QEasingCurve.Linear)
        self.thumbsShow.start()

    def show_system_settings(self):
        self.system_setup.w.show()

    def load_last_prompt(self):
        data = ''
        try:
            with open('configs/ainodes/last_prompt.txt', 'r') as file:
                data = file.read().replace('\n', '')
        except:
            pass
        gs.diffusion.prompt = data
        self.widgets[self.current_widget].w.prompts.setHtml(data)

    def show_model_preview_images(self, model):
        for image in model['images']:
            self.show_image_from_url(image['url'])

    def show_image_from_url(self, url):
        self.web_images.get_image(url)

    def show_web_image_on_canvas(self, image_string):
        try:
            gs.temppath = ''
            self.params.max_frames = 0
            image = Image.open(BytesIO(image_string))
            image = image.convert("RGB")
            mode = image.mode
            size = image.size
            enc_image = base64.b64encode(image.tobytes()).decode()
            self.deforum_ui.signals.txt2img_image_cb.emit(enc_image, mode, size)

        except Exception as e:
            print('Error while fetching the images from web: ', e)


    def image_preview_signal(self, image, *args, **kwargs):
        try:
            mode = image.mode
            size = image.size
            enc_image = base64.b64encode(image.tobytes()).decode()
            self.deforum_ui.signals.txt2img_image_cb.emit(enc_image, mode, size)
            self.signals.image_ready.emit()
        except Exception as e:
            print('image_preview_signal', e)
    @Slot()
    def image_preview_func(self, image=None):
        try:
            img = image #self.image

            if self.params.advanced == True and (self.canvas.canvas.rectlist == [] or self.canvas.canvas.rectlist is None):
                self.params.advanced = False
                self.advanced_temp = True

            if self.params.advanced == True:

                if self.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.canvas.canvas.rectlist[self.render_index].images is not None:
                            templist = self.canvas.canvas.rectlist[self.render_index].images
                        else:
                            templist = []
                        self.canvas.canvas.rectlist[self.render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.canvas.canvas.rectlist[self.render_index].render_index)
                        self.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.canvas.canvas.rectlist[self.render_index].render_index}"))

                        if self.canvas.canvas.anim_inpaint == True:
                            templist[self.canvas.canvas.rectlist[self.render_index].render_index] = qimage
                            self.canvas.canvas.anim_inpaint = False
                        elif self.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.canvas.canvas.rectlist[self.render_index].render_index == None:
                                self.canvas.canvas.rectlist[self.render_index].render_index = 0
                            else:
                                self.canvas.canvas.rectlist[self.render_index].render_index += 1
                        self.canvas.canvas.rectlist[self.render_index].images = templist
                        self.canvas.canvas.rectlist[self.render_index].image = self.canvas.canvas.rectlist[self.render_index].images[self.canvas.canvas.rectlist[self.render_index].render_index]
                        #self.canvas.canvas.rectlist[self.render_index].image = qimage
                        self.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                        self.canvas.canvas.rectlist[self.render_index].img_path = gs.temppath
                    self.canvas.canvas.newimage = True
                    self.canvas.canvas.update()
                    self.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.params.advanced == False:

                if self.advanced_temp == True:
                    self.advanced_temp = False
                    self.params.advanced = True

                if img is not None:
                    image = img
                    w, h = image.size
                    self.add_next_rect(h, w)
                    self.render_index = len(self.canvas.canvas.rectlist) - 1

                    # for items in self.canvas.canvas.rectlist:
                    #    if items.id == self.canvas.canvas.render_item:
                    if self.canvas.canvas.rectlist[self.render_index].images is not None:
                        templist = self.canvas.canvas.rectlist[self.render_index].images
                    else:
                        templist = []
                    self.canvas.canvas.rectlist[self.render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.canvas.canvas.rectlist[self.render_index].images = templist
                    if self.canvas.canvas.rectlist[self.render_index].render_index == None:
                        self.canvas.canvas.rectlist[self.render_index].render_index = 0
                    else:
                        self.canvas.canvas.rectlist[self.render_index].render_index += 1
                    self.canvas.canvas.rectlist[self.render_index].image = \
                    self.canvas.canvas.rectlist[self.render_index].images[
                        self.canvas.canvas.rectlist[self.render_index].render_index]
                    self.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                    self.canvas.canvas.rectlist[self.render_index].params = self.params

                    self.canvas.canvas.newimage = True
                    self.canvas.canvas.redraw()
                    self.canvas.canvas.update()

            if self.params.advanced == False and self.params.max_frames > 1:
                self.params.advanced = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))
        except Exception as e:
            print('image_preview_func', e)

    @Slot()
    def image_preview_func_str(self, image, mode, size):
        try:
            decoded_image = base64.b64decode(image.encode())
            image = Image.frombytes(mode, size, decoded_image)
            image = image.convert('RGB')
            self.image_preview_func(image)
        except Exception as e:
            print('image_preview_func_str', e)
    @Slot()
    def render_index_image_preview_func_str(self, image, mode, size, render_index):
        decoded_image = base64.b64decode(image.encode())
        self.render_index_image_preview_func(Image.frombytes(mode, size, decoded_image), render_index)
    @Slot()
    def render_index_image_preview_func(self, image=None, render_index=None):
        try:
            img = image
            if self.outpaint.batch_process == 'run_hires_batch':
                self.outpaint.last_batch_image = img
            if self.params.advanced == True:
                if self.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.canvas.canvas.rectlist[render_index].images is not None:
                            templist = self.canvas.canvas.rectlist[render_index].images
                        else:
                            templist = []
                        self.canvas.canvas.rectlist[render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.canvas.canvas.rectlist[render_index].render_index)
                        self.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.canvas.canvas.rectlist[render_index].render_index}"))

                        if self.canvas.canvas.anim_inpaint == True:
                            templist[self.canvas.canvas.rectlist[render_index].render_index] = qimage
                            self.canvas.canvas.anim_inpaint = False
                        elif self.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.canvas.canvas.rectlist[render_index].render_index == None:
                                self.canvas.canvas.rectlist[render_index].render_index = 0
                            else:
                                self.canvas.canvas.rectlist[render_index].render_index += 1
                        self.canvas.canvas.rectlist[render_index].images = templist
                        self.canvas.canvas.rectlist[render_index].image = \
                            self.canvas.canvas.rectlist[render_index].images[
                                self.canvas.canvas.rectlist[render_index].render_index]
                        self.canvas.canvas.rectlist[render_index].timestring = time.time()
                        self.canvas.canvas.rectlist[render_index].img_path = gs.temppath
                    self.canvas.canvas.newimage = True
                    self.canvas.canvas.update()
                    self.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.params.advanced == False:

                if img is not None:
                    image = img
                    h, w = image.size
                    self.add_next_rect(h, w)
                    render_index = len(self.canvas.canvas.rectlist) - 1

                    # for items in self.canvas.canvas.rectlist:
                    #    if items.id == self.canvas.canvas.render_item:
                    if self.canvas.canvas.rectlist[render_index].images is not None:
                        templist = self.canvas.canvas.rectlist[render_index].images
                    else:
                        templist = []
                    self.canvas.canvas.rectlist[render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.canvas.canvas.rectlist[render_index].images = templist
                    if self.canvas.canvas.rectlist[render_index].render_index == None:
                        self.canvas.canvas.rectlist[render_index].render_index = 0
                    else:
                        self.canvas.canvas.rectlist[render_index].render_index += 1
                    self.canvas.canvas.rectlist[render_index].image = self.canvas.canvas.rectlist[render_index].images[self.canvas.canvas.rectlist[render_index].render_index]
                    self.canvas.canvas.rectlist[render_index].timestring = time.time()
                    self.canvas.canvas.rectlist[render_index].params = self.params
            self.canvas.canvas.newimage = True
            self.canvas.canvas.redraw()
            self.canvas.canvas.update()

            if self.params.advanced == False and self.params.max_frames > 1:
                self.params.advanced = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))

        except Exception as e:
            print('render_index_image_preview_func', e)
    def image_preview_signal_op(self, image, *args, **kwargs):
        mode = image.mode
        size = image.size
        enc_image = base64.b64encode(image.tobytes()).decode()
        self.outpaint.signals.txt2img_image_op.emit(enc_image, mode, size)
        self.signals.image_ready.emit()
    @Slot()
    def image_preview_func_str_op(self, image, mode, size):
        decoded_image = base64.b64decode(image.encode())
        self.image_preview_func_op(Image.frombytes(mode, size, decoded_image))

    @Slot()
    def image_preview_func_op(self, image=None):
        try:
            img = image #self.image
            # store the last image for a part of the batch hires process
            if self.outpaint.batch_process == 'run_hires_batch':
                index = self.render_index
                if index < 0:
                    index = 0
                self.outpaint.betterslices.append((img.convert('RGBA'),
                                          self.canvas.canvas.rectlist[index].x,
                                          self.canvas.canvas.rectlist[index].y))
            if self.params.advanced == True and (self.canvas.canvas.rectlist == [] or self.canvas.canvas.rectlist is None):
                self.params.advanced = False
                self.advanced_temp = True

            if self.params.advanced == True:

                if self.canvas.canvas.rectlist != []:
                    if img is not None:
                        if self.canvas.canvas.rectlist[self.render_index].images is not None:
                            templist = self.canvas.canvas.rectlist[self.render_index].images
                        else:
                            templist = []
                        self.canvas.canvas.rectlist[self.render_index].PILImage = img
                        qimage = ImageQt(img.convert("RGBA"))
                        pixmap = QPixmap.fromImage(qimage)
                        print(self.canvas.canvas.rectlist[self.render_index].render_index)
                        self.thumbs.w.thumbnails.addItem(QListWidgetItem(QIcon(pixmap),
                                                                         f"{self.canvas.canvas.rectlist[self.render_index].render_index}"))

                        if self.canvas.canvas.anim_inpaint == True:
                            templist[self.canvas.canvas.rectlist[self.render_index].render_index] = qimage
                            self.canvas.canvas.anim_inpaint = False
                        elif self.canvas.canvas.anim_inpaint == False:
                            templist.append(qimage)
                            if self.canvas.canvas.rectlist[self.render_index].render_index == None:
                                self.canvas.canvas.rectlist[self.render_index].render_index = 0
                            else:
                                self.canvas.canvas.rectlist[self.render_index].render_index += 1
                        self.canvas.canvas.rectlist[self.render_index].images = templist
                        self.canvas.canvas.rectlist[self.render_index].image = self.canvas.canvas.rectlist[self.render_index].images[self.canvas.canvas.rectlist[self.render_index].render_index]
                        #self.canvas.canvas.rectlist[self.render_index].image = qimage
                        self.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                        self.canvas.canvas.rectlist[self.render_index].img_path = gs.temppath
                    self.canvas.canvas.newimage = True
                    self.canvas.canvas.update()
                    self.canvas.canvas.redraw()
                    del qimage
                    del pixmap
            elif self.params.advanced == False:

                if self.advanced_temp == True:
                    self.advanced_temp = False
                    self.params.advanced = True

                if img is not None:
                    image = img
                    h, w = image.size
                    self.add_next_rect(h, w)
                    self.render_index = len(self.canvas.canvas.rectlist) - 1

                    # for items in self.canvas.canvas.rectlist:
                    #    if items.id == self.canvas.canvas.render_item:
                    if self.canvas.canvas.rectlist[self.render_index].images is not None:
                        templist = self.canvas.canvas.rectlist[self.render_index].images
                    else:
                        templist = []
                    self.canvas.canvas.rectlist[self.render_index].PILImage = image
                    qimage = ImageQt(image.convert("RGBA"))
                    templist.append(qimage)
                    self.canvas.canvas.rectlist[self.render_index].images = templist
                    if self.canvas.canvas.rectlist[self.render_index].render_index == None:
                        self.canvas.canvas.rectlist[self.render_index].render_index = 0
                    else:
                        self.canvas.canvas.rectlist[self.render_index].render_index += 1
                    self.canvas.canvas.rectlist[self.render_index].image = \
                    self.canvas.canvas.rectlist[self.render_index].images[
                        self.canvas.canvas.rectlist[self.render_index].render_index]
                    self.canvas.canvas.rectlist[self.render_index].timestring = time.time()
                    self.canvas.canvas.rectlist[self.render_index].params = self.params

                    self.canvas.canvas.newimage = True
                    self.canvas.canvas.redraw()
                    self.canvas.canvas.update()

            if self.params.advanced == False and self.params.max_frames > 1:
                self.params.advanced = True

            if self.make_grid:
                self.all_images.append(T.functional.pil_to_tensor(image))
        except Exception as e:
            print('image_preview_func_op', e)

    def add_next_rect(self, h, w):
        #w = self.widgets[self.current_widget].w.W.value()
        #h = self.widgets[self.current_widget].w.H.value()
        resize = False
        try:
            params = copy.deepcopy(self.params)
            if self.canvas.canvas.rectlist == []:
                self.canvas.canvas.w = w
                self.canvas.canvas.h = h
                self.canvas.canvas.addrect_atpos(x=0, y=0, params=params)
                self.cheight = h
                self.w = w
                self.canvas.canvas.render_item = self.canvas.canvas.selected_item
                # print(f"this should only haappen once {self.cheight}")
                # self.canvas.canvas.resize_canvas(w=self.w, h=self.cheight)
            elif self.canvas.canvas.rectlist != []:
                for i in self.canvas.canvas.rectlist:
                    if i.id == self.canvas.canvas.render_item:
                        if i.id == self.canvas.canvas.render_item:
                            w = self.canvas.canvas.rectlist[self.canvas.canvas.rectlist.index(i)].w
                            x = self.canvas.canvas.rectlist[self.canvas.canvas.rectlist.index(i)].x + w + 20
                            y = i.y
                            if x > 3000:
                                x = 0
                                y = self.cheight + 25
                                if self.stopwidth == False:
                                    self.stopwidth = True
                            if self.stopwidth == False:
                                self.w = x + w
                                resize = True
                            if self.cheight < y + i.h:
                                self.cheight = y + i.h
                                resize = True
                            # self.canvas.canvas.selected_item = None
                self.canvas.canvas.addrect_atpos(x=x, y=y, params=params)
                self.canvas.canvas.render_item = self.canvas.canvas.selected_item
            # if resize == True:
            # pass
            # print(self.w, self.cheight)
            self.canvas.canvas.resize_canvas(w=self.w, h=self.cheight)
            # self.canvas.canvas.update()
            # self.canvas.canvas.redraw()
        except Exception as e:
            print('add_next_rect', e)

    def tensor_preview_signal(self, data, data2):
        self.data = data

        if data2 is not None:
            self.data2 = data2
        else:
            self.data2 = None
        self.deforum_ui.signals.deforum_step.emit()

    def tensor_preview_schedule(self):  # TODO: Rename this function to tensor_draw_function
        try:
            if len(self.data) != 1:
                print(
                    f'we got {len(self.data)} Tensors but Tensor Preview will show only one')

            # Applying RGB fix on incoming tensor found at: https://github.com/keturn/sd-progress-demo/
            self.data = torch.einsum('...lhw,lr -> ...rhw', self.data[0], self.latent_rgb_factors)
            self.data = (((self.data + 1) / 2)
                         .clamp(0, 1)  # change scale from -1..1 to 0..1
                         .mul(0xFF)  # to 0..255
                         .byte())
            # Copying to cpu as numpy array
            self.data = rearrange(self.data, 'c h w -> h w c').cpu().numpy()
            dPILimg = Image.fromarray(self.data)
            dqimg = ImageQt(dPILimg)
            # Setting Canvas's Tensor Preview item, then calling function to draw it.
            self.canvas.canvas.tensor_preview_item = dqimg
            self.canvas.canvas.tensor_preview()
            del dPILimg
            del dqimg
        except Exception as e:
            print('tensor_preview_schedule', e)


    @Slot(object)
    def draw_tempRects_signal(self, values):
        self.canvas.canvas.draw_tempRects_signal(values)


    @Slot()
    def stop_processing(self):
        self.stopprocessing = True

    def sort_rects(self, e):
        return e.order

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
        # t.selectFilter(filter or 'All Files (*.*);;')
        if text:
            (next(x for x in t.findChildren(QLabel) if x.text() == 'File &name:')).setText(text)
        if button_caption:
            t.setLabelText(QFileDialog.Accept, button_caption)
        if title:
            t.setWindowTitle(title)
            t.exec_()
        if len(t.selectedFiles()) > 0:
            return t.selectedFiles()[0]
        else:
            return

    # Timeline functions
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
                self.widgets[self.current_widget].w.contrast_schedule.setText(tempString)
            if "Noise" in selection:
                self.widgets[self.current_widget].w.noise_schedule.setText(tempString)
            if "Strength" in selection:
                self.widgets[self.current_widget].w.strength_schedule.setText(tempString)
            if "Rotation X" in selection:
                self.widgets[self.current_widget].w.rotation_3d_x.setText(tempString)
            if "Rotation Y" in selection:
                self.widgets[self.current_widget].w.rotation_3d_y.setText(tempString)
            if "Rotation Z" in selection:
                self.widgets[self.current_widget].w.rotation_3d_z.setText(tempString)
            if "Translation X" in selection:
                self.widgets[self.current_widget].w.translation_x.setText(tempString)
            if "Translation Y" in selection:
                self.widgets[self.current_widget].w.translation_y.setText(tempString)
            if "Translation Z" in selection:
                self.widgets[self.current_widget].w.translation_z.setText(tempString)
            if "Angle" in selection:
                self.widgets[self.current_widget].w.angle.setText(tempString)
            if "Zoom" in selection:
                self.widgets[self.current_widget].w.zoom.setText(tempString)

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


def QIcon_from_svg(svg_filepath, color='black'):
    img = QPixmap(svg_filepath)
    qp = QPainter(img)
    qp.setCompositionMode(QPainter.CompositionMode_SourceIn)
    qp.fillRect(img.rect(), QColor(color))
    qp.end()
    return QIcon(img)

