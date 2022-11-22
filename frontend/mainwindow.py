import time
import random

import numpy as np
import torch
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import QFile, QIODevice, QEasingCurve, Slot, QRect, QThreadPool
from PySide6.QtWidgets import QMainWindow, QToolBar, QPushButton, QGraphicsColorizeEffect
from PySide6.QtGui import QAction, QIcon, QColor, QPixmap, QPainter, Qt
from PySide6 import QtCore, QtWidgets
from einops import rearrange

from backend.worker import Worker
from frontend import plugin_loader
from frontend.ui_model_chooser import ModelChooser_UI
from frontend.ui_paint import PaintUI, spiralOrder, random_path
from frontend.ui_classes import Thumbnails, PathSetup
from frontend.unicontrol import UniControl
import backend.settings as settings
from backend.singleton import singleton
from backend.devices import choose_torch_device
from frontend.ui_timeline import Timeline, KeyFrame

gs = singleton
settings.load_settings_json()

from frontend.ui_deforum import Deforum_UI
from frontend.session_params import SessionParams

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.load_last_prompt()
        self.canvas = PaintUI(self)
        self.setCentralWidget(self.canvas)
        self.setWindowTitle("aiNodes - Still Mode")

        self.resize(1280, 800)
        self.unicontrol = UniControl(self)
        self.sessionparams = SessionParams(self)
        self.params = self.sessionparams.create_params()
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.unicontrol.w.dockWidget)

        self.create_main_toolbar()
        self.create_secondary_toolbar()
        self.history = []
        self.history_index = 0
        self.max_history = 100
        self.add_state_to_history()
        self.update_ui_from_params()

        self.currentFrames = []
        self.renderedFrames = 0

        self.threadpool = QThreadPool()
        self.deforum_ui = Deforum_UI(self)
        self.deforum_ui.signals.txt2img_image_cb.connect(self.image_preview_func)
        self.deforum_ui.signals.deforum_step.connect(self.tensor_preview_schedule)

        self.unicontrol.w.dream.clicked.connect(self.taskswitcher)
        self.canvas.W.valueChanged.connect(self.canvas.canvas.change_resolution)
        self.canvas.H.valueChanged.connect(self.canvas.canvas.change_resolution)
        #self.canvas.canvas.signals.update_selected.connect(self.show_outpaint_details)
        #self.canvas.canvas.signals.update_params.connect(self.create_params)
        #self.canvas.canvas.signals.outpaint_signal.connect(self.deforum_ui.deforum_outpaint_thread)
        self.canvas.canvas.signals.txt2img_signal.connect(self.deforum_six_txt2img_thread)
        self.unicontrol.w.H.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)
        self.unicontrol.w.W.valueChanged.connect(self.canvas.canvas.change_rect_resolutions)
        self.unicontrol.w.lucky.clicked.connect(self.show_default)
        self.unicontrol.w.negative_prompts.setVisible(False)
        self.y = 0
        self.lastheight = None
        self.height = gs.diffusion.H

        self.path_setup = PathSetup()
        self.model_chooser = ModelChooser_UI(self)
        self.unicontrol.w.dockWidget.setWindowTitle("Parameters")
        self.path_setup.w.dockWidget.setWindowTitle("Model / Paths")
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.path_setup.w.dockWidget)
        self.tabifyDockWidget(self.path_setup.w.dockWidget, self.unicontrol.w.dockWidget)
        self.hide_default()
        self.mode = 'txt2img'
        self.init_plugin_loader()
    def taskswitcher(self):
        print(self.unicontrol.w.use_inpaint.isChecked())
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



    def add_state_to_history(self):
        if len(self.history) == self.max_history:
            self.history.pop(0)
        self.history.append(self.params)
        self.history_index = len(self.history)

    def undo(self):
        if self.history_index > 0:
            self.params = self.history[self.history_index - 1]
            self.history_index -= 1
            self.update_ui_from_params()
    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.params = self.history[self.history_index + 1]
            self.history_index += 1
            self.update_ui_from_params()
    def update_ui_from_params(self):
        for key, value in self.params.items():
            try:
                #We have to add check for Animation Mode as thats a radio checkbox with values 'anim2d', 'anim3d', 'animVid'
                #add colormatch_image (it will be with a fancy preview)
                type = str(getattr(self.unicontrol.w, key))
                #print(type, value)
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
    #Main Toolbar, Secondary toolbar to be added


    def init_plugin_loader(self):
        self.unicontrol.w.loadbutton.clicked.connect(self.load_plugin)
        self.unicontrol.w.unloadbutton.clicked.connect(self.unload_plugin)
        self.plugins = plugin_loader.PluginLoader(MainWindow)
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

        self.secondary_toolbar.addAction(select_mode)
        self.secondary_toolbar.addAction(drag_mode)
        self.secondary_toolbar.addAction(add_mode)

        select_mode.triggered.connect(self.canvas.canvas.select_mode)
        drag_mode.triggered.connect(self.canvas.canvas.drag_mode)
        add_mode.triggered.connect(self.canvas.canvas.add_mode)
    def hide_default(self):
        self.toolbar.setVisible(False)
        self.secondary_toolbar.setVisible(False)

        self.unicontrol.w.hidePlottingButton.setVisible(False)
        self.unicontrol.w.hideAdvButton.setVisible(False)
        self.unicontrol.w.hideAesButton.setVisible(False)
        self.unicontrol.w.hideAnimButton.setVisible(False)
        self.unicontrol.w.showHideAll.setVisible(False)
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
        if self.unicontrol.advHidden == False:
            self.unicontrol.hideAdvanced_anim()
        if self.unicontrol.aesHidden == False:
            self.unicontrol.hideAesthetic_anim()
        if self.unicontrol.aniHidden == False:
            self.unicontrol.hideAnimation_anim()

        if self.unicontrol.ploHidden == False:
            self.unicontrol.hidePlotting_anim()




        self.default_hidden = True
    def show_default(self):
        if self.default_hidden == True:
            self.toolbar.setVisible(True)
            self.secondary_toolbar.setVisible(True)

            self.unicontrol.w.hidePlottingButton.setVisible(True)
            self.unicontrol.w.hideAdvButton.setVisible(True)
            self.unicontrol.w.hideAesButton.setVisible(True)
            self.unicontrol.w.hideAnimButton.setVisible(True)
            self.unicontrol.w.showHideAll.setVisible(True)
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


    def run_with_params(self):
        pass

    def deforum_six_txt2img_thread(self):
        self.update = 0
        height = self.height
        #for debug
        #self.deforum_ui.run_deforum_txt2img()
        self.params = self.sessionparams.update_params()
        self.add_state_to_history()
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

        if self.canvas.canvas.rectlist != []:
            for i in self.canvas.canvas.rectlist:
                if i.id == self.canvas.canvas.selected_item:
                    x = i.x + i.w + 20
                    if i.h > self.height:
                        self.height = i.h
                    if self.canvas.canvas.pixmap.width() < 3000:
                        w = self.canvas.canvas.pixmap.width() + self.unicontrol.w.W.value() + 25
                    else:
                        w = self.canvas.canvas.pixmap.width()
                    if x > 3000:
                        self.y = self.y + i.h + 20
                        x = 0
                        self.lastheight = self.lastheight + i.h + 20
                        self.height = self.lastheight
                        w = w
                    if self.lastheight is not None:
                        if self.lastheight < self.height + i.h + 20:
                            self.lastheight = self.height + i.h + 20
                            self.canvas.canvas.resize_canvas(w=w, h=self.lastheight + self.unicontrol.w.H.value())
                    y = self.y


            if x != 0 or y > 0:
                if self.params['advanced'] == False:
                    self.canvas.canvas.w = self.unicontrol.w.W.value()
                    self.canvas.canvas.h = self.unicontrol.w.H.value()
                    self.canvas.canvas.addrect_atpos(x=x, y=self.y, params=self.params)
                    print(f"resizing canvas to {self.height}")
                    self.canvas.canvas.resize_canvas(w=w, h=self.height)
        elif self.params['advanced'] == False or self.canvas.canvas.selected_item == None:
            w = self.unicontrol.w.W.value()
            h = self.unicontrol.w.H.value()
            self.canvas.canvas.w = w
            self.canvas.canvas.h = h
            self.canvas.canvas.addrect_atpos(x=0, y=0)
            self.height = self.unicontrol.w.H.value()
            print(f"this should only haappen once {self.height}")
            self.canvas.canvas.resize_canvas(w=w, h=self.height)

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
                #if self.deforum_ui.deforum.temppath is not None:
                #    items.img_path = self.deforum_ui.deforum.temppath
                self.canvas.canvas.update()


    def tensor_preview_signal(self, data, data2):
        self.data = data
        #print(data)
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
