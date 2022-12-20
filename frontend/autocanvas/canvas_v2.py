import copy
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Signal, QPoint, QSize, QRect, QPointF, QObject, QFile, Slot, QDir, Qt
from PySide6.QtGui import Qt, QColor, QFont, QPainter, QPen, QPixmap, QTransform, QImage
from PySide6.QtWidgets import QGraphicsScene, \
    QGraphicsView, QLabel, QGraphicsPixmapItem, QGraphicsRectItem, \
    QFileDialog, QRubberBand, QGraphicsOpacityEffect, QGraphicsProxyWidget
from frontend.autocanvas.canvas_image_search import ImageSearchWidget
from frontend.autocanvas.canvas_frame_object import Rectangle
from frontend.autocanvas.canvas_proxy_widget import MyProxyWidget
from frontend.ui_classes import SimplePrompt, SimpleKreaPrompts
from backend.singleton import singleton
gs = singleton

import time
from uuid import uuid4
import random

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)
__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)

class Callbacks(QObject):
    outpaint_signal = Signal()
    txt2img_signal = Signal()
    update_selected = Signal()
    update_params = Signal(str)

class Canvas(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)







        self.proxy = None
        self.acceptDrops()
        self.setAcceptDrops(True)
        self.setUpdatesEnabled(True)
        self.parent = parent
        self.last_pos = None
        self.signals = Callbacks()
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        #self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.reset()
        self.soft_reset()
        self.sub_hover_item = None
        self.hover_item = None
        self.selected_item = None
        self.tempbatch = None
        self.undoitems = []
        self.maintimer = QtCore.QTimer()
        self.maintimer.timeout.connect(self.set_new)
        self.running = False
        self.setAcceptDrops(True)
        self.anim_inpaint = False
        self.origin = QPoint()
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.proxies = {}
        self.uids = []
        self.textshown = None
        self.hover_item = None
        self.last_hover_item = None
        self.imagesearch = ImageSearchWidget()
        self.imagesearch.show()
        #self.animkeyeditor = AnimKeyEditor()
        #self.proxy = MyProxyWidget(self.animkeyeditor.w)
        #self.proxy.setWidget(self.animkeyeditor.w)
        #self.scene.addItem(self.proxy)
        #self.mouseMoveEvent = self.onMouse
        #self.start_main_clock()

    def soft_reset(self, w=512, h=512):
        w = self.parent.W.value()
        h = self.parent.H.value()
        self.pixmap = QPixmap(w, h)
        self.currentWidth = w
        self.currentHeight = h
        self.temprects = None
        self.pixmap.fill(Qt.transparent)
        self.bgitem = QGraphicsPixmapItem()
        self.rectItem = QGraphicsRectItem(0, 0, 512, 512)
        self.parent.parent.w = w
        self.parent.parent.cheight = h
        if w < 3000:
            self.parent.parent.stopwidth = False
        #self.debugtext = QGraphicsTextItem("0, 0\n")
        #self.helpText = QGraphicsTextItem("C - Hand Drag\nV - Place Rectangles")
        self.bgitem.setPixmap(self.pixmap)

        #self.setPixmap(self.pixmap)
        self.scene.addItem(self.bgitem)
        #self.parent.parent.widgets[self.parent.parent.current_widget] = UniControl(self.parent.parent)
        #self.parent.parent.proxy = MyProxyWidget(self.parent.parent.widgets[self.parent.parent.current_widget].w)
        #self.scene.addItem(self.parent.parent.proxy)
        #self.scene.addItem(self.rectItem)
        self.tensor_preview_item = None
        self.rectlist.clear()
        self.rectlist = []
        self.selected_item = None
        self.render_item = None
        #self.signals.update_selected.emit()
        self.parent.parent.render_index = 0
        self.parent.parent.thumbs.w.thumbnails.clear()

        self.krea_proxy = QGraphicsProxyWidget()
        self.krea = SimpleKreaPrompts()
        self.krea_proxy.setWidget(self.krea.w)
        #self.scene.addItem(self.krea_proxy)
        self.krea.w.setWindowOpacity(0)
        self.krea.w.krea_prompt_1.clicked.connect(self.set_to_krea_1)
        self.krea.w.krea_prompt_2.clicked.connect(self.set_to_krea_2)
        self.krea.w.krea_prompt_3.clicked.connect(self.set_to_krea_3)
        self.krea.w.krea_prompt_4.clicked.connect(self.set_to_krea_4)
        self.krea.w.krea_prompt_5.clicked.connect(self.set_to_krea_5)
        gs.donthover = None



    def reset(self):
        self.zoom = 1
        self.rotate = 0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.w = 4096
        self.h = 4096

        self.rectlist = []
        self.rectlist.clear()
        self.scene = Scene()
        self.parent.w = 4096
        self.parent.cheight = 4096
        self.parent.stopwidth = False

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')
        #self.mode = 'drag'
        self.setMouseTracking(True)
        self.painter = QPainter()

        self.update()
        self.setScene(self.scene)

        self.tempbatch = []
        self.hover_item = None
        self.selected_item = None
        self.render_item = None
        self.outpaintitem = None
        self.outpaintsource = None
        self.soft_reset()

        self.rendermode = 1
        ###print(self.rendermode)
        self.painter.begin(self.pixmap)
        self.painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform | QPainter.LosslessImageRendering)
        self.painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        self.painter.end()
        self.fitInView(self.bgitem, Qt.AspectRatioMode.KeepAspectRatio)
        self.updateView()
        self.offset = 0
        self.maskoffset = 0
        self.counter = 0
        self.newimage = False
        self.txt2img = False
        self.lastpos = False
        self.rectsdrawn = False
        self.ctrlmodifier = None
        self.shiftmodifier = None
        self.hover_item = None
        self.last_hover_item = None
