

from PySide6 import QtUiTools

from PySide6.QtWidgets import QApplication, QGraphicsView
from PySide6.QtWidgets import *
from PySide6.QtGui import *


import sys, os


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


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)




class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done
