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




    @Slot()
    def start_main_clock(self):
        if self.running == False:
            self.maintimer.start(80)
            self.running = True
        elif self.running == True:
            for i in self.rectlist:
                i.play()

    def stop_main_clock(self):
        if self.running == True:
            for i in self.rectlist:
                i.stop()
            self.redraw()
            self.maintimer.stop()
            self.running = False
    def set_new(self):
        ##print("triggered")
        self.newimage = True
        self.update()

    def play_selected(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    if i.running == False:
                        if i.images != []:
                            i.play()
    def stop_selected(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    if i.running == True:
                        i.stop()
    def skip_forward(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    if i.running == True:
                        i.stop()
                    i.iterate()
    def skip_back(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    if i.running == True:
                        i.stop()
                    i.iterate_back()
    def resize_canvas(self, w, h):
        self.pixmap = QPixmap(w, h)
        self.bgitem.setPixmap(self.pixmap)
        self.newimage = True
        self.currentWidth = w
        self.currentHeight = h
        ##print(f"resized to {w, h}")


    def change_resolution(self):
        w = self.parent.W.value()
        h = self.parent.H.value()
        self.resize_canvas(w, h)
        #self.pixmap = QPixmap(w, h)
        #self.bgitem.setPixmap(self.pixmap)
        #self.updateScene([self.bgitem])

        #if self.selected_item is not None:
        #    for i in self.rectlist:
        #        if i.id == self.selected_item:
        #            i.w = w
        #            i.h = h
        #self.newimage = True
    def change_rect_resolutions(self):
        self.w = self.parent.parent.widgets[self.parent.parent.current_widget].w.W.value()
        self.h = self.parent.parent.widgets[self.parent.parent.current_widget].w.H.value()
        #if self.selected_item is not None:
        #    for i in self.rectlist:
        #        if i.id == self.selected_item:
        #            i.w = w
        #            i.h = h
        #self.newimage = True
    def set_to_krea_1(self):
        self.proxies[self.uid].widget.prompts.setText(self.proxies[self.uid].promptlist[0])
    def set_to_krea_2(self):
        self.proxies[self.uid].widget.prompts.setText(self.proxies[self.uid].promptlist[1])
    def set_to_krea_3(self):
        self.proxies[self.uid].widget.prompts.setText(self.proxies[self.uid].promptlist[2])
    def set_to_krea_4(self):
        self.proxies[self.uid].widget.prompts.setText(self.proxies[self.uid].promptlist[3])
    def set_to_krea_5(self):
        self.proxies[self.uid].widget.prompts.setText(self.proxies[self.uid].promptlist[4])
    def reset(self):
        self.zoom = 1
        self.rotate = 0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.rectlist = []
        self.rectlist.clear()
        self.scene = Scene()
        self.w = 512
        self.h = 512
        self.parent.w = 4096
        self.parent.cheight = 4096
        self.parent.stopwidth = False
        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')
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
    def getXScale(self):
        return float(1024)/float(self.width())
    def getYScale(self):
        return float(1024)/float(self.height())

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def addrect(self, dummy=False):
        rect = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        prompt = ""
        if dummy == False:
            rect[uid] = Rectangle(self, prompt, self.scene.scenePos.x() - self.w / 2, self.scene.scenePos.y() - self.h / 2, self.w, self.h, uid, params=self.parent.parent.sessionparams.update_params())
        else:
            rect[uid] = Rectangle(self, prompt, 0, 0, 1, 1, 1)
        #rect[uid].signals.set_new_signal.connect(self.set_new)
        self.selected_item = uid
        self.rectlist.append(rect[uid])
        if dummy == True:
            self.rectlist.remove(rect[uid])

    def tensor_preview(self):
        ###print(self.tensor_preview_item)
        if self.tensor_preview_item is not None:
            w = self.tensor_preview_item.size().width() * 8
            h = self.tensor_preview_item.size().height() * 8
            try:
                x = self.rectlist[self.parent.parent.render_index].x
                y = self.rectlist[self.parent.parent.render_index].y
            except:
                x = 0
                y = 0

            self.painter.begin(self.pixmap)
            pixmap = QPixmap(w, h).fromImage(self.tensor_preview_item)

            self.painter.drawPixmap(x, y, w, h, pixmap)
            self.painter.end()
            self.bgitem.setPixmap(self.pixmap)
            self.update()


    def drawRect(self, x=None, y=None, width=256, height=256):
        ###print(f"selected:{self.selected_item}")
        ###print(f"we are putting that thing to:{x}, {y}, and our width is {self.width()}")
        if x != None:
            Xscale = self.getXScale()
            Yscale = self.getYScale()
            scaledWidth = (self.w / 2) / Xscale
            scaledHeight = (self.h / 2) / Yscale
            x = (x - scaledWidth) * Xscale
            y = (y - scaledHeight) * Yscale
            if x < 0: x = 0
            if y < 0: y = 0
        if self.mode == "generic" or self.mode == "outpaint" or self.mode == "inpaint" or self.mode == "move":
            pen = QPen(Qt.GlobalColor.blue, 3, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
            self.rectItem.setPen(pen)
            self.rectItem.setRect(x, y, self.w, self.h)
        elif self.mode == "select_":
            if self.selected_item is not None:
                for i in self.rectlist:
                    if i.id == self.selected_item:
                        ###print(i)
                        self.rectItem.show()
                        pen = QPen(Qt.green, 3, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
                        self.rectItem.setPen(pen)
                        self.rectItem.setRect(i.x, i.y, self.w, self.h)
        if self.selected_item is None:
            self.rectItem.hide()
        self.bgitem.setPixmap(self.pixmap)
        self.update()

    def hoverCheck(self):
        matchFound = False
        #gs.donthover = None
        #if gs.donthover == None:

        self.hover_item = None

        #prev_index = self.last_index
        for i in self.rectlist:
            if i.x <= self.scene.scenePos.x() <= i.x + i.w and i.y <= self.scene.scenePos.y() <= i.y + i.h:
                self.hover_item = i.id
                index = self.rectlist.index(i)

            else:
                i.color = __idleColor__
        if self.hover_item is not None:
            if self.rectlist[index].prompt_visible is None:
                self.rectlist[index].show_prompt()
        if self.hover_item != self.last_hover_item and self.last_hover_item != None:
            for i in self.rectlist:
                if i.id == self.last_hover_item:
                    if i.prompt_visible == True:
                        i.hide_prompt()
        self.last_hover_item = self.hover_item
    def save_canvas(self):
        self.redraw(transparent=True)
        timestring = time.strftime('%Y-%m-%d-%H-%S')
        filename = f"output/canvas/canvas_{timestring}.png"
        os.makedirs('output/canvas', exist_ok=True)
        file = QFile(filename)
        self.pixmap.save(file, "PNG")
        self.redraw()
    def first_rectangle(self):
        self.hoverCheck()
        if self.hover_item is None:
            self.addrect()
            self.signals.txt2img_signal.emit()
            self.mode = "outpaint"
    @Slot(int)
    def set_offset(self, offset):
        self.maskoffset = offset
        ###print(f"offset is now: {self.maskoffset}")
    def toJSON(self, item):
        return json.dumps(item, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)
    def save_rects_as_json(self, filename=None):
        ###print(filename)
        # Save json to file (data.json)
        templist = []
        for items in self.rectlist:
            ###print(items.x)
            item = {}
            print(items.params.__dict__)
            params = items.params.__dict__
            item = {
                "x": items.x,
                "y": items.y,
                "w": items.w,
                "h": items.h,
                "id": items.id,
                "img_path": items.img_path,
                "timestring": items.timestring,
                "order": items.order,
            }
            for key, value in params.items():
                pass
                #item[key] = value
                #print(key, value)
                #print(item[key], value)
                #item[key] = value
                #print(item[key])
                #print(value)

            #print(item)
            templist.append(item)
        if filename != False:
            data = filename
        else:
            data = self.getfile(save=True)
        with open(data, "w") as output:
            json.dump(templist, output, sort_keys=True, indent=4)
        print("File Saved")
    def getfile(self, file_ext='', text='', button_caption='', button_type=0, title='', save=False):
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
    def load_img_into_rect(self):
        data = self.getfile()
        if data is not None:
            gs.temppath = data
            if self.selected_item is not None:
                for i in self.rectlist:
                    if i.id == self.selected_item:
                        self.parent.parent.image = Image.open(data)
                        i.w = self.parent.parent.image.size[0]
                        i.h = self.parent.parent.image.size[1]
                        self.parent.parent.render_index = self.rectlist.index(i)

                        self.parent.parent.sessionparams.params.advanced = True
                        self.parent.parent.image_preview_func()

    def load_rects_from_json(self):

        #self.reset()
        #self.soft_reset()

        data = self.getfile()

        if data is not None:
            self.rectlist.clear()
            with open(data, 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
            x = 0
            for key in json_object:
                #print(key)
                rect = {}
                for w, x in key.items():
                    #print(w, x)
                    rect[w] = x
                    ###print(x['x'])
                    #rect = {}
                    #uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
                    ###print(f"adding rectangles at:{x} {y}")
                    #try:
                    #    prompt = x['prompt']
                    #except:
                    #    prompt = ''
                    #try:
                    #    #params = SimpleNamespace(**x['params'])
                    #    params = x['params']
                    #except Exception as e:
                    #    print(e)
                    #    params = {}
                    #print(x)
                rect[rect['id']] = Rectangle(self, rect['x'], rect['x'], rect['y'], rect['w'], rect['h'], rect['id'], rect['order'], rect['img_path'], params=None)
                self.rectlist.append(rect[rect['id']])
            for items in self.rectlist:
                if items.img_path is not None:
                    image = Image.open(items.img_path) #.convert("RGBA")
                    qimage = ImageQt(image.convert("RGBA"))
                    items.images = []
                    items.image = qimage
                    items.images.append(qimage)
                    items.index = 0
                    items.active = True
            self.newimage = True
            self.pixmap.fill(__backgroudColor__)
            self.redraw()

    @Slot()
    def create_tempBatch(self, prompts, keyframes, startOffsetX=0, startOffsetY=0, randomize = False):

        self.rows = self.rows + 1
        self.cols = self.cols + 1
        self.first_image = True
        row = 1
        x = startOffsetX
        y = startOffsetY
        self.counter = 0
        self.tempbatch = []
        self.tempbatch.clear()
        #self.rectlist.clear()

        prompt_series = pd.Series([np.nan for a in range(int(self.rows) * int(self.cols))])

        prom = prompts
        if keyframes == '':
            keyframes = "0"

        new_prom = list(prom.split("\n"))
        new_key = list(keyframes.split("\n"))

        prompts = dict(zip(new_key, new_prom))

        for i, prompt in prompts.items():
            n = int(i)
            prompt_series[n] = prompt
        prompt_series = prompt_series.ffill().bfill()

        while row < self.rows:
            self.add_cols(prompt_series, row, self.cols, self.offset, x, y, randomize)
            row = row + 1
            y = y + self.h - self.offset + (random.randint(-50, 50)) if randomize else y + self.h - self.offset #
            if y >= self.pixmap.height() - 4 * self.offset:
                row = self.rows
        self.pen = QPen(Qt.green, 3, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        #self.draw_tempBatch(self.tempbatch)
        return self.tempbatch

    def add_cols(self, prompt_series, row, cols, offset, x, y, randomize = False):
        col = 1
        thiscol = []
        while col < cols:
            if row == 1:
                prompt_index = col - 1
            else:
                prompt_index = ((self.cols - 1) * (row - 1)) + col - 1
            randomized = y + (random.randint(-50, 50))
            randomizedX = x + (random.randint(-50, 50))
            batch = {
                "x" : randomizedX if randomizedX > 0 and randomize else x,
                "y" : randomized if randomized > 0 and randomize else y,
                "width" : self.w,
                "height" : self.h,
                "order" : int(self.counter),
                "prompt" : prompt_series[prompt_index],
            }
            x = x + self.w - offset + (random.randint(-50, 50)) if randomize else x + self.w - offset
            thiscol.append(batch)
            self.counter = self.counter + 1
            col = col + 1
            if x >= self.pixmap.width():
                col = cols
        self.tempbatch.append(thiscol)
    def draw_tempBatch(self, tempbatch, run = True):
        #self.pixmap.fill(__backgroudColor__)
        self.pen = QPen(Qt.green, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        x = 0
        for rows in tempbatch:
            if type(rows) == dict:
                self.draw_tempRects(rows["x"], rows["y"], self.w, self.h, rows["order"], x)
                x += 1
            else:
                for items in rows:
                    self.draw_tempRects(items["x"], items["y"], self.w, self.h, items["order"], x)
                    x += 1

    def draw_rects(self):
        painter = QPainter(self.pixmap)
        rect = self.sceneRect()
        mgridsize = 128
        self.draw_grid_(painter, rect, mgridsize)


        # Set pen color to black and line width to 1

    def draw_grid_(self, painter, rect, mgridsize):
        #rows = self.h / mgridsize
        #cols = self.w / mgridsize


        penHLines = QPen(QColor(75, 75, 75), 4, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin)
        painter.setPen(penHLines)
        for i in range(0, self.pixmap.width(), 64):
            painter.drawLine(i, 0, i, self.pixmap.height())
        for i in range(0, self.pixmap.height(), 64):
            painter.drawLine(0, i, self.pixmap.width(), i)

        painter.setPen(QPen(QColor(100, 100, 100), 10, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
        for i in range(0, self.pixmap.width(), 512):
            painter.drawLine(i, 0, i, self.pixmap.height())
        for i in range(0, self.pixmap.height(), 512):
            painter.drawLine(0, i, self.pixmap.width(), i)
        #self.pixmap.fill(__backgroudColor__)
        self.pen = QPen(Qt.red, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(self.pen)
        x = 0
        for i in self.rectlist:
            ###print(self.rectlist[x].order)
            ###print(i.order, x)
            painter.drawRect(i.x, i.y, i.w, i.h)
            x += 1
        """mgridsize = int(mgridsize)
        left = int(rect.left())
        top = int(rect.top())

        lines = []
        #print(left, int(rect.right()) + 1, mgridsize)
        for x in range(left, int(rect.right()), mgridsize):
            print(x, rect.top(), x, rect.bottom())
            lines.append(QLineF(x, rect.top(), x, rect.bottom()))
        for y in range(top, int(rect.bottom()) + 1, mgridsize):
            lines.append(QLineF(rect.left(), y, rect.right(), y))

        thickLines = []

        for x in range(left, int(rect.right()), mgridsize * 5):
            thickLines.append(QLineF(x, int(rect.top()), x, rect.bottom()))
        for y in range(top, int(rect.bottom()), mgridsize * 5):
            thickLines.append(QLineF(rect.left(), y, rect.right(), y))

        #myPen = QPen(Qt.NoPen)
        #painter.setBrush(QBrush(QColor(55, 55, 55, 255)))
        #painter.setPen(myPen)
        #painter.drawRect(rect)

        penHLines = QPen(QColor(75, 75, 75), 2, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin)
        painter.setPen(penHLines)
        painter.drawLines(lines)

        painter.setPen(QPen(QColor(100, 100, 100), 4, Qt.SolidLine, Qt.FlatCap, Qt.RoundJoin))
        painter.drawLines(thickLines)

        painter.setPen(Qt.blue)

        points = []
        for x in range(left, int(rect.right()), mgridsize):
            for y in range(top, int(rect.bottom()), mgridsize):
                points.append(QPointF(x, y))
        painter.drawPoints(points)

        w = 0
        # Draw time
        scale = self.getXScale()
        while w <= self.w:
            print(w, 0, 100, 100, 15, str(w))
            painter.drawText(w - 128, 0, 100, 100, 15, str(w))
            w += 128"""
    def draw_tempRects(self, x, y, width, height, order, value):
        #self.painter.begin(self.pixmap)
        self.painter.setPen(self.pen)
        rect = QRect(x, y, width, height)
        font = QFont("Segoe UI Black")
        font.setPointSize(52)
        self.painter.drawRect(rect)
        self.painter.setFont(font)
        #self.painter.drawText(x - 25 + width / 2, y  + 25 + width / 2, f"{order} / {value}")
        #self.painter.end()
        #self.bgitem.setPixmap(self.pixmap)

    def visualize_rects(self, overlays=False):
        self.painter.begin(self.pixmap)
        self.painter.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Overlay)
        self.pen = QPen(Qt.green, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        font = QFont("Segoe UI Black")
        font.setPointSize(52)
        self.painter.setFont(font)
        x = 0
        self.painter.setPen(self.pen)
        for items in self.rectlist:

            r = random.randint(25, 255)
            g = random.randint(25, 255)
            b = random.randint(25, 255)

            color = QColor(r, g, b)
            rect = QRect(items.x, items.y, items.w, items.h)
            self.painter.fillRect(rect, color)
            self.painter.drawText(items.x - 25 + items.w / 2, items.y  + 25 + items.w / 2, f"{x}")
            x = x + 1
        self.painter.end()
        self.bgitem.setPixmap(self.pixmap)
    #def get_next_color(self, x):


    def addrect_atpos(self, prompt='', x=0, y=0, image=None, render_index=None, order=None, params=None):
        #rect = {}
        #uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        #rect[uid] = Rectangle(self, prompt, x, y, self.w, self.h, uid, order=order, image=image, render_index=render_index,
        #                      params=params)
        #self.rectlist.append(rect[uid])
        rect = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        ###print(f"adding rectangles at:{x} {y}")
        matchfound = False
        for i in self.rectlist:
            if i.x == x and i.y == y:
                if image is not None:
                    i.images.append(image)
                    i.image = image
                    i.order = order
                    i.params = params
                    if i.render_index is not None:
                        i.render_index += 1
                    else:
                        i.render_index = 0
                    uid = i.id
                    self.selected_item = i.id
                    self.parent.parent.render_index = self.rectlist.index(i)
                matchfound = True

        if matchfound == False:
            if params == None:
                params = self.parent.parent.sessionparams.update_params()
            rect[uid] = Rectangle(self, prompt, x, y, self.w, self.h, uid, order = order, image=image, render_index=None, params=copy.deepcopy(params))
            print(f"adding rect with seed {params.seed}")
            self.selected_item = uid
            if self.rectlist == []:
                self.txt2img = True
            self.rectlist.append(rect[uid])
            self.parent.parent.render_index = len(self.rectlist) - 1
            self.counter += 1

        self.newimage = True
        return uid
    def inpaint_current_frame(self):
        self.mode = "inpaint"
        self.anim_inpaint = True
        if self.selected_item is not None:
            self.reusable_inpaint(self.render_item)
    def reusable_inpaint(self, id):
        self.busy = True
        #self.redraw(transparent=True)
        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpaintmaskimage = QPixmap(self.w, self.h)
        outpaintmaskimage.fill(Qt.transparent)
        maskpainter = QPainter()
        #outpainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        #outpainter.setRenderHint(QPainter.LosslessImageRendering)
        self.selected_item = id
        for items in self.rectlist:
            if items.id == id:
                self.parent.parent.render_index = self.rectlist.index(items)
                #print(items.render_index)
                rect = QRect(items.x, items.y, self.w, self.h)
                image = self.pixmap.toImage()
                newimage = image.copy(rect)
                outpainter.begin(outpaintimage)
                outpainter.drawImage(0,0,newimage)
                outpainter.end()
        outpaintimage.save("outpaint.png")
        outpaintimage.save("outpaint_mask.png")
        self.render_item = self.selected_item
        self.outpaintsource = "outpaint.png"
        self.busy = False
        self.parent.parent.widgets[self.parent.parent.current_widget].w.recons_blur.setValue(0)
        self.signals.outpaint_signal.emit()



    def reusable_outpaint(self, id):
        self.busy = True
        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpaintmaskimage = QPixmap(self.w, self.h)
        outpaintmaskimage.fill(Qt.transparent)
        maskpainter = QPainter()
        outpainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        outpainter.setRenderHint(QPainter.LosslessImageRendering)
        self.selected_item = id
        for x in self.rectlist:
            if x.id == id:
                #x.image = None
                #self.update()
                for i in self.rectlist:
                    if x.y >= i.y - i.h and x.x >= i.x - i.w:
                        if x.y <= i.y + i.h and x.x <= i.x + i.w:
                            if i.id != x.id:
                                if x.y > i.y:
                                    Ymaskoffset = self.maskoffset
                                else:
                                    Ymaskoffset = -self.maskoffset
                                if x.x > i.x:
                                    Xmaskoffset = self.maskoffset
                                else:
                                    Xmaskoffset = -self.maskoffset
                                #i.color = __selColor__
                                #self.update()
                                if i.image is not None:
                                    ###print("Found an image to outpaint")
                                    overlap = True
                                    rect = QRect(x.x - i.x, x.y - i.y, self.w, self.h)
                                    maskrect = QRect(x.x - i.x + Xmaskoffset, x.y - i.y + Ymaskoffset, self.w, self.h)
                                    newimage = i.image.copy(rect)
                                    maskimage = i.image.copy(maskrect)
                                    maskpainter.begin(outpaintmaskimage)
                                    maskpainter.drawImage(0,0,maskimage)
                                    maskpainter.end()
                                    outpainter.begin(outpaintimage)
                                    outpainter.drawImage(0,0,newimage)
                                    outpainter.end()
                                    #self.addrect()
                    if i.id == x.id:
                        print(f"setting render index to:{self.rectlist.index(i)}")
                        #self.parent.parent.sessionparams.params = x.params
                        #self.parent.parent.sessionparams.params.advanced = True
                        self.parent.parent.render_index = self.rectlist.index(i)
        outpaintimage.save("outpaint.png")
        #outpainter.end()
        outpaintmaskimage.save("outpaint_mask.png")
        self.render_item = self.selected_item
        self.outpaintsource = "outpaint.png"
        self.busy = False

    def redo_outpaint(self, id):
        print('redo')
        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpaintmaskimage = QPixmap(self.w, self.h)
        outpaintmaskimage.fill(Qt.transparent)
        maskpainter = QPainter()
        outpainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        outpainter.setRenderHint(QPainter.LosslessImageRendering)
        self.selected_item = id

        for x in self.rectlist:
            if x.id == id:
                x.image = None
                self.update()
                for i in self.rectlist:
                    if x.y >= i.y - self.h and x.x >= i.x - self.w:
                        if x.y <= i.y + self.h and x.x <= i.x + i.w:
                            if i.id != x.id:
                                if x.y > i.y:
                                    Ymaskoffset = self.maskoffset
                                else:
                                    Ymaskoffset = -self.maskoffset
                                if x.x > i.x:
                                    Xmaskoffset = self.maskoffset
                                else:
                                    Xmaskoffset = -self.maskoffset
                                #i.color = __selColor__
                                #self.update()
                                if i.image is not None:
                                    ###print("Found an image to outpaint")
                                    rect = QRect(x.x - i.x, x.y - i.y, self.w, self.h)
                                    maskrect = QRect(x.x - i.x + Xmaskoffset, x.y - i.y + Ymaskoffset, self.w, self.h)
                                    newimage = i.image.copy(rect)
                                    maskimage = i.image.copy(maskrect)
                                    maskpainter.begin(outpaintmaskimage)
                                    maskpainter.drawImage(0,0,maskimage)
                                    maskpainter.end()
                                    outpainter.begin(outpaintimage)
                                    outpainter.drawImage(0,0,newimage)
                                    outpainter.end()
                    if i.id == x.id:
                        print("Found an image to outpaint")
                        self.parent.parent.render_index = self.rectlist.index(i)
        outpaintimage.save("outpaint.png")
        #outpainter.end()
        outpaintmaskimage.save("outpaint_mask.png")
        self.outpaintsource = "outpaint.png"
        self.redo = True
        self.render_item = self.selected_item
        self.signals.update_params.emit(id)
        # = render_index
        self.signals.outpaint_signal.emit()

    def region_to_outpaint(self, event):


        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpaintmaskimage = QPixmap(self.w, self.h)
        outpaintmaskimage.fill(Qt.transparent)


        outpainter = QPainter()
        maskpainter = QPainter()
        outpainter.setRenderHint(QPainter.LosslessImageRendering)

        outpainter.setCompositionMode(QPainter.CompositionMode_Screen)
        #outpainter.setRenderHint(QPainter.LosslessImageRendering)
        for i in self.rectlist:
            if (self.scene.scenePos.y() - self.h / 2) >= i.y - self.h and (self.scene.scenePos.x() - self.w / 2) >= i.x - self.w:
                if (self.scene.scenePos.y() - self.h / 2) <= i.y + self.h and (self.scene.scenePos.x() - self.w / 2) <= i.x + i.w:
                    if (self.scene.scenePos.y() - self.h / 2) > i.y:
                        Ymaskoffset = self.maskoffset
                    else:
                        Ymaskoffset = -self.maskoffset
                    if (self.scene.scenePos.x() - self.w / 2) > i.x:
                        Xmaskoffset = self.maskoffset
                    else:
                        Xmaskoffset = -self.maskoffset
                    if i.image is not None:
                        maskrect = QRect(int((self.scene.scenePos.x() - self.w / 2) - i.x) + Xmaskoffset, int((self.scene.scenePos.y() - self.h / 2) - i.y) + Ymaskoffset, self.w, self.h)
                        rect = QRect(int((self.scene.scenePos.x() - self.w / 2) - i.x), int((self.scene.scenePos.y() - self.h / 2) - i.y), self.w, self.h)
                        newimage = i.image.copy(rect)
                        maskimage = i.image.copy(maskrect)
                        maskpainter.begin(outpaintmaskimage)
                        maskpainter.drawImage(0,0,maskimage)
                        maskpainter.end()
                        outpainter.begin(outpaintimage)
                        outpainter.drawImage(0,0,newimage)
                        outpainter.end()


        self.addrect()
        self.parent.render_index = len(self.rectlist) - 1
        self.render_item = self.selected_item
        self.draw_rects()
        self.newimage = True
        self.redraw()
        #self.signals.update_params.emit(self.selected_item)
        outpaintimage.save("outpaint.png")
        outpaintmaskimage.save("outpaint_mask.png")


        #This creates a resized image, did not like results, might be useful for inpaint still
        """newimage = outpaintimage.scaled(self.w - self.offset, self.h - self.offset, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        rect = QRect(int(-self.offset/2), int(-self.offset/2), self.w, self.h)
        newimage2 = newimage.copy(rect)
        outpainter.end()
        newoutpaintimage = QPixmap(self.w, self.h)
        newoutpaintimage.fill(Qt.transparent)
        outpainter.begin(newoutpaintimage)
        outpainter.drawImage(int(self.offset/2),int(self.offset/2),newimage2.toImage())
        outpainter.end()
        newoutpaintimage.save("outpaint_mask.png")"""

        self.outpaintsource = "outpaint.png"
        self.signals.outpaint_signal.emit()


    def mousePressEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.drag_mode()
        fn = getattr(self, "%s_mousePressEvent" % self.mode, None)
        if fn:
            fn(e)
            super(Canvas, self).mousePressEvent(e)


    def mouseMoveEvent(self, e):
        fn = getattr(self, "%s_mouseMoveEvent" % self.mode, None)
        if fn:
            fn(e)
            super(Canvas, self).mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MiddleButton:
            self.select_mode()

        fn = getattr(self, "%s_mouseReleaseEvent" % self.mode, None)
        if fn:
            fn(e)
            super(Canvas, self).mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e):
        fn = getattr(self, "%s_mouseDoubleClickEvent" % self.mode, None)
        if fn:
            return fn(e)
    def sortRects(self, e):
        if e.order is not None:
            key = e.timestring
        else:
            key = 0
        return key
    def redraw(self, transparent=None):
        if transparent:
            self.pixmap.fill(Qt.transparent)
        else:
            self.pixmap.fill(__backgroudColor__)
            self.draw_rects()
        self.painter.begin(self.pixmap)
        self.painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        #text_render_index = None
        if self.rectlist is not [] and self.rectlist is not None:
            #self.rectlist.sort(reverse=False, key=self.sortRects)
            for i in self.rectlist:
                #if self.hover_item is not None:
                    #if i.id == self.hover_item:
                        #text_render_index = self.rectlist.index(i)
                if i.image is not None and i.active == True:
                    #rect = QRect(i.x, i.y, i.w, i.h)
                    pic = i.image.copy(0, 0, i.image.width(), i.image.height())
                    pixmap = QPixmap.fromImage(pic)
                    self.painter.drawPixmap(int(i.x), int(i.y), i.w, i.h, pixmap, 0, 0, i.w, i.h)
                    if self.running == True:
                        if len(i.images) > 1:
                            if i.running is True:
                                pixmap = QPixmap('frontend/icons/square.svg')
                            elif i.running is False:
                                pixmap = QPixmap('frontend/icons/play.svg')
                            self.painter.drawPixmap(int(i.x), int(i.y), i.w, i.h, pixmap, 0, 0, i.w, i.h)
        '''if text_render_index is not None:
            #print(text_render_index)
            x = self.rectlist[text_render_index].x
            y = self.rectlist[text_render_index].y
            w = self.rectlist[text_render_index].w
            h = self.rectlist[text_render_index].h
            self.painter.setPen(QColor(Qt.GlobalColor.darkGreen))
            self.painter.setBrush(QColor(Qt.GlobalColor.darkGray))
            font = QFont("Segoe UI Black")
            font.setPointSize(52)
            self.painter.setFont(font)
            self.painter.setCompositionMode(QPainter.CompositionMode_Overlay)
            self.painter.drawRect(x, y, w, h)
            self.painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
            self.painter.setPen(Qt.black)
            self.painter.drawText(x + 10, y + 10, w, h, Qt.TextWordWrap, self.rectlist[text_render_index].prompt)
            self.painter.setPen(Qt.white)
            self.painter.drawText(x, y, w, h, Qt.TextWordWrap, self.rectlist[text_render_index].prompt)'''



        self.painter.end()
        #self.bgitem.setX(0)
        self.bgitem.setPixmap(self.pixmap)
        self.newimage = False

        #self.anim = QtCore.QPropertyAnimation(self, b"geometry")
        #self.anim.setDuration(500)
        #self.anim.setStartValue(QRect(512, 0, self.width(), self.height()))
        #self.anim.setEndValue(QRect(0, 0, self.width(), self.height()))
        #self.anim.setEasingCurve(QEasingCurve.Linear)

        #self.anim.start()


        #self.update()
    def old_paintEvent(self):

        #self.pixmap.fill(Qt.black)


        #self.pixmap.fill(Qt.black)
        #painter.drawText(0, 50, 256, 256, Qt.AlignHCenter, "C - Hand Drag\nV - Place Rectangles")
        #Show Outpaint Preview rectangle
        #rect = QRect(0, self.height() - 256, 256, 256)
        #painter.drawRect(rect)
        #self.newimage = True
        #self.pixmap.fill(Qt.transparent)

        help2 = "None"
        if self.hover_item is not None:
            for items in self.rectlist:
                if items.id == self.hover_item:
                    help2 = f"hover item details:{items.x}, {items.y}"


    def paintEvent(self, e):
        super(Canvas, self).paintEvent(e)
        if self.newimage == True:

            self.redraw()


    def generic_mouseMoveEvent(self, e):
        #self.redraw()
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
    def undoEvent(self):
        if self.undoitems is not []:
            x = len(self.undoitems) - 1
            self.rectlist.append(self.undoitems[x])
            self.undoitems.pop(x)
            self.newimage = True
        #self.update()

    def select_mode(self):
        self.mode = "select"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        #try:
        #    self.scene.addItem(self.rectItem)
        #except:
        #    pass
        #self.scene.addItem(self.rectItem)
        ###print("Button pressed")
        self.redraw()
        #self.update()
    def move_mode(self):
        self.mode = "move_frame"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        try:
            self.scene.addItem(self.rectItem)
        except:
            pass
        self.redraw()

    def drag_mode(self):
        self.mode = "drag"
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        try:
            self.scene.removeItem(self.rectItem)
        except:
            pass
        self.redraw()
        self.setUpdatesEnabled(True)
    def rubberband_mode(self):
        self.mode = "rubberband"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        try:
            self.scene.removeItem(self.rectItem)
        except:
            pass
        self.redraw()
        self.setUpdatesEnabled(True)
    def enterEvent(self, event):
        ##print("Enter Event")
        is_in = True
        self.bgitem.update()
        self.rectItem.update()
        #self.redraw()

    def add_mode(self):
        self.addrect(dummy=True)
        self.mode = "outpaint"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.scene.addItem(self.rectItem)
        self.setUpdatesEnabled(True)
        self.redraw()
    def inpaint_mode(self):
        self.addrect(dummy=True)
        self.mode = "inpaint"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.scene.addItem(self.rectItem)
        self.setUpdatesEnabled(True)
        self.redraw(transparent=True)
    def keyPressEvent(self, e):
        super(Canvas, self).keyPressEvent(e)
        if e.key() == 16777248:
            if e.isAutoRepeat():
                pass
            else:
                self.shiftmodifier = True
        elif e.key() == 16777249:
            if e.isAutoRepeat():
                pass
            else:
                self.ctrlmodifier = True
        elif e.key() == 32:
            if e.isAutoRepeat():
                print("Key is being held down")
                print(self.mode)
                if self.mode != "drag":
                    self.drag_mode()
            else:
                if self.mode != "drag":
                    self.drag_mode()
                print("Key was pressed")
        elif e.key() == 16777223:
            self.parent.parent.delete_outpaint_frame()

        print(f"key pressed: {e.key(), self.shiftmodifier, self.ctrlmodifier}")
        """if e.key() == 67:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.mode = "drag"
            self.drawRect(self.width(), self.height())
        elif e.key() == 86:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.mode = "generic"
        elif e.key() == 66:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.mode = "select"
            self.drawRect(self.width(), self.height())
        elif e.key() == 78:
            self.mode = "outpaint"
        elif e.key() == 77:
            #self.mode = "move"
            #print("move mode")
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif e.key() == 90 and self.ctrlmodifier == True:
            self.undoEvent()
        if e.key() == 16777249:
            self.ctrlmodifier = True
        else:
            self.ctrlmodifier = None"""
    def keyReleaseEvent(self, e):
        self.ctrlmodifier = None
        self.shiftmodifier = None
        if e.key() == 32:
            if e.isAutoRepeat():
                print("release autorepear")
            else:
                print("release key")
                self.select_mode()
        super(Canvas, self).keyReleaseEvent(e)

    def select_mousePressEvent(self, e):
        self.parent.setAcceptDrops(True)
        super(Canvas, self).mousePressEvent(e)
        ##print("select mode active")
        if self.hover_item == self.selected_item:
            #if self.sub_hover_item is not None:

            #for i in self.rectlist:
            #    if i.id == self.selected_item:
            #        i.stop()
            #self.selected_item = None
            #self.drawRect()
            return
        if self.hover_item is not None:
            self.selected_item = self.hover_item
            for i in self.rectlist:
                if i.id == self.selected_item:
                    self.parent.parent.render_index = self.rectlist.index(i)
            #self.signals.update_selected.emit()
            #if self.rectlist[self.render_index].running == True:
            #    self.rectlist[self.render_index].stop()
            #else:
            #    self.rectlist[self.render_index].play()
            self.drawRect()
            self.newimage = True
            #for i in self.rectlist:
            #    if i.id == self.selected_item:
            #        i.play()
            #self.rectlist[self.render_index].play()
        else:
            pass
            #self.selected_item = None


    def select_mouseMoveEvent(self, e):
        self.hoverCheck()
    def select_mouseReleaseEvent(self, event):
        return
    def generic_mousePressEvent(self, e):
        self._start = e.pos()
        self.posx = e.pos().x()
        self.posy = e.pos().y()
        self.first_rectangle()
        return

    def generic_mouseReleaseEvent(self, event):
        #self.redraw()
        #start = QPointF(self.mapToScene(self._start))
        #end = QPointF(self.mapToScene(event.pos()))
        #self.scene.addItem(
        #    QGraphicsLineItem(QLineF(start, end)))
        return
    def generic_mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
    def drag_mouseMoveEvent(self, e):
        super(Canvas, self).mouseMoveEvent(e)
        return
    def drag_mouseReleaseEvent(self, event):
        return
    def drag_mousePressEvent(self, event):
        return
    def move_frame_mouseMoveEvent(self, e):
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
        self.update()
        return
    def move_frame_mouseReleaseEvent(self, event):
        return
    def move_frame_mousePressEvent(self, event):
        self.move_frame_action()
        return

    def move_frame_action(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    i.x = self.scene.scenePos.x() - i.w / 2
                    i.y = self.scene.scenePos.y() - i.h / 2
                    if self.proxies[i.id] is not None:
                        self.proxies[i.id].setPos(i.x, i.y)
                    self.newimage = True
                    self.update()
    def inpaint_mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.eraser_color = QColor(QColor(Qt.white))
            self.eraser_color.setAlpha(255)
            self.pen = QPen(self.eraser_color, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            self.last_pos = self.scene.scenePos
            eraser_color = QColor(Qt.white)
            eraser_color.setAlpha(255)
            self.pen = QPen(eraser_color, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        elif e.button() == Qt.RightButton:
            self.mode = "inpaint"
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            id = self.addrect_atpos(x=self.scene.scenePos.x() - self.w / 2, y=self.scene.scenePos.y() - self.h / 2, params=copy.deepcopy(self.parent.parent.sessionparams.params))
            self.reusable_inpaint(id)

    def inpaint_mouseMoveEvent(self, e):
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
        self.update()


        if self.last_pos:
            self.painter.begin(self.pixmap)
            self.painter.setCompositionMode(QPainter.CompositionMode_Clear)
            self.painter.setPen(self.pen)
            self.painter.drawLine(self.last_pos, self.scene.scenePos)
            self.last_pos = self.scene.scenePos
            self.bgitem.setPixmap(self.pixmap)
            self.painter.end()
            self.bgitem.update()
        return

    def inpaint_mouseReleaseEvent(self, e):
        self.last_pos = None

    def outpaint_mouseMoveEvent(self, e):
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
        self.update()
        return
    def outpaint_mouseReleaseEvent(self, event):
        return
    def outpaint_mousePressEvent(self, event):
        if self.scene.pos is not None:
            self.addrect()
            #self.signals.update_selected.emit()
            self.redraw()
            if self.ctrlmodifier == True:
                self.reusable_outpaint(self.selected_item)
                self.signals.outpaint_signal.emit()
                return
            elif self.ctrlmodifier == False:
                self.drag_mode()

            #else:
                #self.addrect()
                #self.draw_rects()
                #self.newimage = True
                #self.redraw()
                #self.signals.update_params.emit(self.selected_item)
                #return
        return

    def rubberband_mousePressEvent(self, event):
        self.parent.parent.widgets[self.parent.parent.current_widget].w.hires.setCheckState(Qt.CheckState.Unchecked)
        self.parent.parent.widgets['unicontrol'].w.with_inpaint.setCheckState(Qt.CheckState.Unchecked)
        self.parent.parent.sessionparams.params.with_inpaint = False

        #self.hoverCheck()
        #if self.hover_item is not None:
        #    self.parent.parent.widgets['unicontrol'].w.with_inpaint.setCheckState(Qt.CheckState.Checked)
        self.startpoint = self.scene.scenePos
        if event.button() == Qt.LeftButton:
            self.origin = QPoint(event.pos())
            self.rubberBand.setGeometry(
                QRect(self.origin, QSize())
            )
            self.rubberBand.show()

    def rubberband_mouseMoveEvent(self, event):
        if not self.origin.isNull():
            self.rubberBand.setGeometry(
                QRect(self.origin, event.position().toPoint()).normalized()
            )

    def rubberband_mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.parent.parent.widgets[self.parent.parent.current_widget].w.mode.setCurrentText('advanced')
            #self.hoverCheck()
            #if self.hover_item is not None:
            #    self.parent.parent.widgets['unicontrol'].w.with_inpaint.setCheckState(Qt.CheckState.Checked)
            self.rubberBand.hide()
            if self.startpoint is not None:
                #if event.pos().x() > self.startpoint.x() or event.pos().y() > self.startpoint.y():
                #    print("normal case")
                x = self.startpoint.x()
                y = self.startpoint.y()
                #else:
                #    pass
                    #x = event.pos().x()
                    #y = event.pos().y()
            absw = abs(self.scene.scenePos.x() - x)
            absh = abs(self.scene.scenePos.y() - y)
            w = abs(self.scene.scenePos.x() - x)
            h = abs(self.scene.scenePos.y() - y)
            w, h = map(lambda x: x - x % 64, (w, h))
            if w > 704:
                self.parent.parent.widgets[self.parent.parent.current_widget].w.hires.setCheckState(Qt.CheckState.Checked)
            if h > 704:
                self.parent.parent.widgets[self.parent.parent.current_widget].w.hires.setCheckState(Qt.CheckState.Checked)

            print(self.scene.scenePos.x(), self.startpoint.x())
            if self.scene.scenePos.x() < self.startpoint.x() and self.scene.scenePos.y() < self.startpoint.y():
                y = self.scene.scenePos.y()
                x = self.scene.scenePos.x()
            elif self.scene.scenePos.x() < self.startpoint.x() and self.scene.scenePos.y() > self.startpoint.y():
                y = self.scene.scenePos.y() - h
                x = self.scene.scenePos.x()
            elif self.scene.scenePos.x() > self.startpoint.x() and self.scene.scenePos.y() > self.startpoint.y():
                x = self.startpoint.x()
                y = self.startpoint.y()
            elif self.scene.scenePos.x() > self.startpoint.x() and self.scene.scenePos.y() < self.startpoint.y():
                y = self.scene.scenePos.y()
                x = self.scene.scenePos.x() - w
                print("1")
            #if event.pos().y() < self.startpoint.y():
            #    y = self.scene.scenePos.pos().y()
            uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4)
            self.uid = uid
            rect = {}
            rect[uid] = Rectangle(self, "", x, y, w, h, uid)
            print(f"Rubberband added a rect at: {x, y, w, h}")
            self.selected_item = uid
            self.render_item = uid
            self.rectlist.append(rect[uid])

            self.draw_rects()
            self.parent.parent.sessionparams.params.W = w
            self.parent.parent.sessionparams.params.H = h

            prompt = SimplePrompt()

            self.proxies[uid] = MyProxyWidget(prompt, self)
            #rect[uid].subwidgets['prompt'].w.setGeometry(x, y, w, h)
            #rect[uid].subwidgets['prompt'].w.setStyleSheet("background-color: transparent")

            #self.proxies[uid].setWidget(self.prompt.w)
            self.proxies[uid].fx = QGraphicsOpacityEffect()
            self.proxies[uid].setGraphicsEffect(self.proxies[uid].fx)
            self.proxies[uid].setGeometry(x, y, w, h)
            #self.proxies[uid].resize(w, h)
            self.scene.addItem(self.proxies[uid])
            self.proxies[uid].fx.setOpacity(0)
            self.proxies[uid].fx.setEnabled(True)
            self.animation = QtCore.QPropertyAnimation(self.proxies[uid].fx, b"opacity")
            self.animation.setDuration(200)
            self.animation.setStartValue(0)
            self.animation.setEndValue(1)
            self.animation.start()
            self.proxies[uid].widget.prompts.setViewportMargins(10, 10, 10, 10)
            #self.startpoint.setX(self.startpoint.x() + self.w)
            #p = QPointF(x, y)
            #self.proxies[uid].setPos(p)

            self.proxies[uid].widget.delbutton.clicked.connect(self.proxies[uid].prompt_destroy_with_frame)
            try:
                self.parent.parent.widgets[self.parent.parent.current_widget].w.dreambutton.disconnect()
            except:
                pass
            self.proxies[uid].widget.dreambutton.clicked.connect(self.proxies[uid].proxy_task)
            self.drag_mode()
            self.draw_rects()
            if self.parent.parent.sessionparams.params.with_inpaint == True:
                w, h = map(lambda x: x - x % 64, (w, h))
                self.rectlist[len(self.rectlist) - 1].w = w
                self.rectlist[len(self.rectlist) - 1].h = h
                self.parent.parent.sessionparams.params.W = w
                self.parent.parent.sessionparams.params.H = h
                #self.parent.parent.update_ui_from_params()
                self.change_resolution()
            self.parent.parent.widgets[self.parent.parent.current_widget].w.W.setValue(w)
            self.parent.parent.widgets[self.parent.parent.current_widget].w.H.setValue(h)
            #self.flipwidget = FlipWidget()
            #self.scene.addWidget(self.flipwidget)

    def removeitem_by_uid_from_scene(self):
        self.scene.removeItem(self.proxies[self.uid])
    def check_for_frame_overlap(self):
        for x in self.rectlist:
            if x.id == self.selected_item:
                # x.image = None
                # self.update()
                for i in self.rectlist:
                    if x.y >= i.y - i.h and x.x >= i.x - i.w:
                        if x.y <= i.y + i.h and x.x <= i.x + i.w:
                            if i.id != x.id:
                                self.parent.parent.sessionparams.params.with_inpaint = True

        if self.parent.parent.sessionparams.params.with_inpaint == True:
            return True
        else:
            return False

    def proxy_task(self):
        #print(self.parent.parent.widgets['unicontrol'].w.prompts.setText('11123'))
        text = self.prompt.w.prompts.toPlainText()
        self.parent.parent.widgets['unicontrol'].w.prompts.setText(str(text))
        self.prompt_destroy()
        self.parent.parent.task_switcher()

    def prompt_destroy_with_frame(self):
        self.prompt_destroy_action()
        self.parent.parent.delete_outpaint_frame()
    def prompt_destroy(self):
        self.prompt_destroy_action()
    def prompt_destroy_action(self):
        try:
            self.prompt.w.delbutton.disconnect()
        except:
            pass
        #self.parent.parent.widgets[self.parent.parent.current_widget].w.prompts = self.backup
        self.animation = QtCore.QPropertyAnimation(self.fx, b"opacity")
        self.animation.setDuration(250)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.start()
        QtCore.QTimer.singleShot(500, self.finish_prompt_destroy)
    def finish_prompt_destroy(self):
        self.prompt.w.destroy()
        self.scene.removeItem(self.proxy)
        self.proxy = None
        self.prompt = None
        #self.parent.parent.delete_outpaint_frame()


    def wheelEvent(self, event):
        x = event.angleDelta().y() / 120
        if self.ctrlmodifier is None and self.shiftmodifier is None:
            if x > 0:
                self.zoom *= 1.05
                self.updateView()
            elif x < 0:
                self.zoom /= 1.05
                self.updateView()
        elif self.ctrlmodifier is not None:
            W = self.parent.parent.widgets['unicontrol'].w.W.value()
            if x > 0:
                self.parent.parent.widgets['unicontrol'].w.W.setValue(W + 64)
            elif x < 0:
                self.parent.parent.widgets['unicontrol'].w.W.setValue(W - 64)
        elif self.shiftmodifier is not None:
            H = self.parent.parent.widgets['unicontrol'].w.H.value()
            if x > 0:
                self.parent.parent.widgets['unicontrol'].w.H.setValue(H + 64)
            elif x < 0:
                self.parent.parent.widgets['unicontrol'].w.H.setValue(H - 64)

        #if self.gridenabled == True:


    def updateView(self):
        self.setTransform(QTransform().scale(self.zoom, self.zoom).rotate(self.rotate))
        #if self.proxy is not None:
        #    self.proxy.setScale((self.getXScale() / self.zoom))
            #print(self.getXScale() * self.zoom)
        #if self.tempbatch is not None and self.gridenabled == True:
        #    self.draw_tempBatch(self.tempbatch)


    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist') or 1==1:
            # Accept the event if it contains URLs

            event.accept()
        else:
            # Reject the event if it does not contain URLs
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist') or 1==1:
            # Accept the event if it contains URLs

            event.accept()
        else:
            # Reject the event if it does not contain URLs
            event.ignore()

    def dropEvent(self, event):
        print(event)
        #print(event.mimeData().formats())
        if event.mimeData().hasImage() == True:

            print("drop")
            #print(event.mimeData().hasImage())
            #byte_array = event.mimeData().data('application/x-qabstractitemmodeldatalist')
            # Get the list of URLs from the event
            #print(byte_array)
            image = event.mimeData().imageData().toImage()
            print(image.height())
            #print(image)
            x = self.scene.scenePos.x()
            y = self.scene.scenePos.y()

            uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            params = self.parent.parent.sessionparams.update_params()
            rect = {}
            rect[uid] = Rectangle(self, "", x, y, self.w, self.h, uid, order =  0, image=image, render_index=None, params=copy.deepcopy(params))
            print(f"adding rect with seed {params.seed}")
            self.selected_item = uid
            if self.rectlist == []:
                self.txt2img = True
            self.rectlist.append(rect[uid])
            self.newimage = True
            event.accept()

class Scene(QGraphicsScene):
    def __init__ (self, parent=None):
        QGraphicsScene.__init__ (self, parent)
        self.pos = None
        self.scenePos = None
        self.parent = parent

    def mouseMoveEvent(self, event):
        super(Scene, self).mouseMoveEvent(event)
        self.pos = QPointF(event.screenPos())
        self.scenePos = event.scenePos()
        self.gridenabled = False

