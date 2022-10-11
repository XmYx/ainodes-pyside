
from PySide6 import QtUiTools

from PySide6.QtWidgets import QApplication, QGraphicsView
from PySide6.QtWidgets import *
from PySide6.QtGui import *


import sys, os
app = QApplication(sys.argv)
pixmap = QPixmap('frontend/main/splash_2.png')
splash = QSplashScreen(pixmap)
splash.show()

icon = QIcon('frontend/main/splash_2.png')


if (os.name == 'nt'):
    #This is needed to display the app icon on the taskbar on Windows 7
    import ctypes
    myappid = 'aiNodes' # arbitrary string
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)


# Create the tray
tray = QSystemTrayIcon()
tray.setIcon(icon)
tray.setVisible(True)

# Create the menu
menu = QMenu()
action = QAction("A menu item")
menu.addAction(action)

# Add a Quit option to the menu.
quit = QAction("Quit")
quit.triggered.connect(app.quit)
menu.addAction(quit)

# Add the menu to the tray
tray.setContextMenu(menu)

#from PyQt6 import QtCore as qtc
#from PyQt6 import QtWidgets as qtw
#from PyQt6 import uic
#from PyQt6.Qt import *
from PySide6.QtCore import *
from PySide6 import QtCore
from PySide6.QtGui import QIcon, QPixmap
import transformers
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import BertTokenizerFast
import warnings, random, traceback, time


from ldm.generate import Generate
from ui_classes import *
from backend.ui_func import getLatestGeneratedImagesFromPath

import torch
import torchvision
import torchvision.transforms as T
from PIL.ImageQt import ImageQt
from PIL import Image
from einops import rearrange
import numpy as np
import cv2
import time

from frontend.main_window_class import GenerateWindow



#Node Editor Functions - We have to make it a QWidget because its now a MainWindow object, which can only be created in a QApplication, which we already have.
#from nodeeditor.utils import loadStylesheet
#from nodeeditor.node_editor_window import NodeEditorWindow
#from frontend.example_calculator.calc_window import CalculatorWindow
#from qtpy.QtWidgets import QApplication as qapp


from PySide6.QtGui import QIcon, QKeySequence, QAction
from PySide6.QtWidgets import QMdiArea, QWidget, QDockWidget, QMessageBox, QFileDialog
from PySide6.QtCore import Qt, QSignalMapper

from nodeeditor.node_editor_window import NodeEditorWindow

from nodeeditor.node_editor_window import NodeEditorWindow
from frontend.example_calculator.calc_sub_window import CalculatorSubWindow
from frontend.example_calculator.calc_drag_listbox import QDMDragListbox
from nodeeditor.utils import dumpException, pp
from frontend.example_calculator.calc_conf import CALC_NODES

# Enabling edge validators
from nodeeditor.node_edge import Edge
from nodeeditor.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node
)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)


#from nodeeditor.utils import loadStylesheets
#from nodeeditor.node_editor_window import NodeEditorWindow


from backend.singleton import singleton

import backend.settings as settings
settings.load_settings_json()

gr = Generate(  weights     = 'models/sd-v1-4.ckpt',
                config     = 'configs/stable-diffusion/v1-inference.yaml',
                )

gs = singleton

gs.result = ""
gs.callbackBusy = False

gs.album = getLatestGeneratedImagesFromPath()

def prepare_loading():
    transformers.logging.set_verbosity_error()

    # this will preload the Bert tokenizer fles
    print('preloading bert tokenizer...')

    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print('...success')

    # this will download requirements for Kornia
    print('preloading Kornia requirements (ignore the deprecation warnings)...')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        import kornia
    print('...success')

    version = 'openai/clip-vit-large-patch14'

    print('preloading CLIP model (Ignore the deprecation warnings)...')
    sys.stdout.flush()
    load_upscalers()
    tokenizer = CLIPTokenizer.from_pretrained(version)
    transformer = CLIPTextModel.from_pretrained(version)
    print('\n\n...success')

    # In the event that the user has installed GFPGAN and also elected to use
    # RealESRGAN, this will attempt to download the model needed by RealESRGANer


def load_upscalers():
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


import platform

if "macOS" in platform.platform():
    gs.platform = "macOS"
    #prepare_loading()

load_upscalers()



"""class CalculatorWin(CalculatorWindow):
    def __init__(self, *args, **kwargs):




        app2 = qapp(sys.argv)
        nodes = CalculatorWindow()
        nodes.show()
        app2.exec()"""
"""def show_nodes():
    #in main thread:
    CalculatorWin()

    #in a separate thread
    #worker = Worker(CalculatorWin) # Any other args, kwargs are passed to the run function
    # Execute
    #threadpool.start(worker)"""

if __name__ == "__main__":

    sshFile="frontend/style/QTDark.stylesheet"


    threadpool = QThreadPool()
    mainWindow = GenerateWindow(threadpool)

    mainWindow.w.setWindowTitle("aiNodes")
    mainWindow.w.setWindowIcon(QIcon('frontend/main/splash_2.png'))
    with open(sshFile,"r") as fh:
        mainWindow.w.setStyleSheet(fh.read())
    #app.setIcon(QIcon('frontend/main/splash.png'))
    mainWindow.w.show()
    #mainWindow.nodeWindow.show()
    mainWindow.w.resize(1280, 720)
    splash.finish(mainWindow)
    #mainWindow.progress_thread()

    #mainWindow.thumbnails.setGeometry(680,0,800,600)

    mainWindow.w.prompt.w.runButton.clicked.connect(mainWindow.txt2img_thread)
    #mainWindow.runner.runButton.clicked.connect(mainWindow.progress_thread)

    mainWindow.w.actionNodes.triggered.connect(mainWindow.nodeWindow.show)
    mainWindow.w.sizer_count.w.scaleSlider.valueChanged.connect(mainWindow.update_scaleNumber)
    mainWindow.w.sizer_count.w.gfpganSlider.valueChanged.connect(mainWindow.update_gfpganNumber)

    mainWindow.w.preview.w.zoomInButton.clicked.connect(mainWindow.zoom_IN)
    mainWindow.w.preview.w.zoomOutButton.clicked.connect(mainWindow.zoom_OUT)

    sys.exit(app.exec())
