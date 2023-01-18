import base64
import copy
import json
import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Signal, QLine, QPoint, QRectF, QSize, QRect, QLineF, QPointF, QObject, QFile, Slot, QDir, Qt, \
    QPropertyAnimation, QEasingCurve, QEvent
from PySide6.QtGui import Qt, QColor, QFont, QPalette, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor, \
    QPixmap, QTransform, QDragEnterEvent, QDragMoveEvent, QImage, QMouseEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu, QGraphicsScene, \
    QGraphicsView, QGraphicsItem, QGraphicsWidget, QLabel, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsRectItem, \
    QGraphicsTextItem, QScrollArea, QHBoxLayout, QLayout, QAbstractScrollArea, QFileDialog, QSpinBox, \
    QGraphicsProxyWidget

from PySide6 import QtCore, QtGui
from backend.singleton import singleton
from frontend.ui_classes import AnimKeyEditor

gs = singleton
from time import gmtime, strftime
import time
from uuid import uuid4
import random

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)

__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)

class Rectangle(object):
    def __init__(self, prompt, x, y, w, h, id, order = None, img_path = None, image = None, render_index=None, params=None):

        #self.parent = parent
        self.prompt = prompt
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = image
        self.render_index = render_index
        self.images = []
        self.params = params
        #print(f"Hello, I'm a rectangle with seed {params.seed}")
        if self.image is not None:
            self.images.append(self.image)
            self.render_index = 0
        self.order = order
        self.color = __idleColor__
        self.timestring = time.time()
        self.active = True
        self.PILImage = None
        self.running = False
        self.img_path = img_path
        #self.signals = RectangleCallbacks()
        #self.timer = QtCore.QTimer()

    def play(self):
        if self.parent.running == True:
            if self.images != []:
                self.timer = QtCore.QTimer()
                self.timer.timeout.connect(self.iterate)
                self.timer.start(80)
                #self.signals.start_main.emit()
                self.running = True

    def iterate(self):
        self.render_index = (self.render_index + 1) % len(self.images)
        if self.render_index == len(self.images):
            self.render_index = 0
        self.image = self.images[self.render_index]
        if self.running == False:
            self.parent.newimage = True
            self.parent.update()
        print(self.render_index)
        print(len(self.images))

    def iterate_back(self):
        self.render_index = (self.render_index - 1) % len(self.images)
        if self.render_index == -1:
            self.render_index = len(self.images)
        self.image = self.images[self.render_index]
        if self.running == False:
            self.parent.newimage = True
            self.parent.update()
        print(self.render_index)

    def stop(self):
        self.timer.stop()
        self.running = False

class Callbacks(QObject):
    outpaint_signal = Signal()
    txt2img_signal = Signal()
    update_selected = Signal()
    update_params = Signal(str)
    run_redraw = Signal()
    draw_tempRects = Signal(object)

class RectangleCallbacks(QObject):
    start_main = Signal()


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

class MyProxyWidget(QGraphicsProxyWidget):
    def __init__(self, widget):
        super(MyProxyWidget, self).__init__()
        self.setWidget(widget)

    def mousePressEvent(self, event):
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        dx = event.pos().x() - self.last_pos.x()
        dy = event.pos().y() - self.last_pos.y()
        self.setPos(self.x() + dx, self.y() + dy)
        self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        self.setCursor(QtCore.Qt.ArrowCursor)

class Canvas(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.setUpdatesEnabled(True)
        self.parent = parent
        self.last_pos = None
        self.signals = Callbacks()
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
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
        self.busy = False
        self.tile_size = 512

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

    def change_resolution(self):
        w = self.parent.W.value()
        h = self.parent.H.value()
        self.resize_canvas(w, h)
        self.redraw()

    def change_rect_resolutions(self):
        self.w = self.parent.parent.widgets[self.parent.parent.current_widget].w.W.value()
        self.h = self.parent.parent.widgets[self.parent.parent.current_widget].w.H.value()


    def soft_reset(self, w=512, h=512):
        w = self.parent.W.value()
        h = self.parent.H.value()
        self.pixmap = QPixmap(w, h)
        self.currentWidth = w
        self.currentHeight = h
        self.temprects = None
        self.pixmap.fill(__backgroudColor__)
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
        #self.scene.addItem(self.rectItem)
        self.tensor_preview_item = None
        self.rectlist.clear()
        self.rectlist = []
        self.selected_item = None
        self.render_item = None
        self.signals.update_selected.emit()
        self.parent.parent.render_index = 0
        self.parent.parent.thumbs.w.thumbnails.clear()

    def reset(self):
        self.zoom = 1
        self.rotate = 0
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.w = 512
        self.h = 512

        self.rectlist = []
        self.rectlist.clear()
        self.scene = Scene()
        self.parent.w = 512
        self.parent.cheight = 512
        self.parent.stopwidth = False

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')
        self.mode = 'drag'
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
        self.ctrlmodifier = False

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
            rect[uid] = Rectangle(prompt, self.scene.scenePos.x() - self.w / 2, self.scene.scenePos.y() - self.h / 2, self.w, self.h, uid, params=self.parent.parent.sessionparams.update_params())
        else:
            rect[uid] = Rectangle(prompt, 0, 0, 1, 1, 1)
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
            #self.update()


    def drawRect(self, x=None, y=None, width=256, height=256):

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
        elif self.mode == "select":
            if self.selected_item is not None:
                for i in self.rectlist:
                    if i.id == self.selected_item:
                        ###print(i)
                        pen = QPen(Qt.green, 3, Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
                        self.rectItem.setPen(pen)
                        self.rectItem.setRect(i.x, i.y, self.w, self.h)
        if self.selected_item is None:
            self.rectItem.setRect(0, 0, 5, 5)
        self.bgitem.setPixmap(self.pixmap)
        #self.update()

    def hoverCheck(self):

        self.hover_item = None
        self.sub_hover_item = None
        matchFound = False

        for i in self.rectlist:
            ###print(i.id)
            if i.x <= self.scene.scenePos.x() <= i.x + i.w and i.y <= self.scene.scenePos.y() <= i.y + i.h:

                if self.hover_item is not None:
                    self.sub_hover_item = i.id
                if self.sub_hover_item is None:
                    self.hover_item = i.id
                    self.render_index = self.rectlist.index(i)
                matchFound = True
            else:
                i.color = __idleColor__
                #self.update()
            if not matchFound:
                self.hover_item = None
        #self.update()

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

            print(item)
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
        if len(t.selectedFiles()) > 0:
            return t.selectedFiles()[0]
        else:
            return

    def load_img_into_rect(self):
        data = self.getfile()
        if data is not None:
            gs.temppath = data
            if self.selected_item is not None:
                for i in self.rectlist:
                    if i.id == self.selected_item:
                        image = Image.open(data)
                        i.w = image.size[0]
                        i.h = image.size[1]
                        render_index = self.rectlist.index(i)
                        self.parent.parent.params.advanced = True

                        mode = image.mode
                        size = image.size
                        enc_image = base64.b64encode(image.tobytes()).decode()
                        self.parent.parent.signals.image_loaded.emit(enc_image, mode, size, render_index)


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
                print(key)
                rect = {}
                for w, x in key.items():
                    print(w, x)
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
                rect[rect['id']] = Rectangle(rect['x'], rect['x'], rect['y'], rect['w'], rect['h'], rect['id'], rect['order'], rect['img_path'], params=None)
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
        #print('self.cols', self.cols)
        #print('self.rows', self.rows)
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

        while row <= self.rows:
            self.add_cols(prompt_series, row, self.cols, self.offset, x, y, randomize)
            row = row + 1
            y = (y + self.h - self.offset + (random.randint(-50, 50))) if randomize else (y + self.h - self.offset) #

            #print('y', y)
            #print('compare to', self.pixmap.height() - ((self.rows -1) * self.offset))
            #print('self.offset', self.offset)
            #print('self.pixmap.height()', self.pixmap.height())
            if y >= self.pixmap.height():
                print('y to big', y)
                row = self.rows+1
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
            #print('x', x)
            #print('self.pixmap.width()', self.pixmap.width())
            if x >= self.pixmap.width()+self.tile_size:
                print('x to big', x)
                col = cols
        self.tempbatch.append(thiscol)

    def draw_tempBatch(self, tempbatch, run = True):
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
        x = 0
        for rect in self.rectlist:
            self.draw_permRects(rect.x, rect.y, rect.w, rect.h, rect.order, x)
            x += 1

    def draw_permRects(self, x, y, width, height, order, value):
        self.pen = QPen(Qt.red, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        permRect = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'order': order,
            'value': value
        }
        self.signals.draw_tempRects.emit(permRect)

    def draw_tempRects(self, x, y, width, height, order, value):
        self.pen = QPen(Qt.green, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
        tempRect = {
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'order': order,
            'value': value
        }
        self.signals.draw_tempRects.emit(tempRect)

    def draw_tempRects_signal(self, tempRect):
        self.painter.begin(self.pixmap)
        self.painter.setPen(self.pen)
        rect = QRect(tempRect['x'], tempRect['y'], tempRect['width'], tempRect['height'])
        font = QFont("Segoe UI Black")
        font.setPointSize(52)
        self.painter.drawRect(rect)
        self.painter.setFont(font)
        #self.painter.drawText(x - 25 + width / 2, y  + 25 + width / 2, f"{order} / {value}")
        self.painter.end()
        self.bgitem.setPixmap(self.pixmap)

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


    def addrect_atpos(self, prompt='', x=0, y=0, image=None, render_index=None, order=None, params=None, color=None):
        if color is not None:
            self.pen = QPen(color, int(3 / self.zoom), Qt.DashDotLine, Qt.RoundCap, Qt.RoundJoin)
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
            rect[uid] = Rectangle(prompt, x, y, self.w, self.h, uid, order = order, image=image, render_index=None, params=copy.deepcopy(params))
            #print(f"adding rect with seed {params.seed}")
            self.selected_item = uid
            if self.rectlist == []:
                self.txt2img = True
            self.rectlist.append(rect[uid])
            self.parent.parent.render_index = len(self.rectlist) - 1
            self.counter += 1

        self.newimage = True
        return uid

    def addrect_atpos_object(self, data_object):

        prompt=data_object['prompt']
        x=data_object['x']
        y=data_object['y']
        image=data_object['image']
        render_index=data_object['render_index']
        order=data_object['order']
        params=data_object['params']


        rect = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())

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
            rect[uid] = Rectangle(prompt, x, y, self.w, self.h, uid, order = order, image=image, render_index=None, params=copy.deepcopy(params))

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

        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpaintmaskimage = QPixmap(self.w, self.h)
        outpaintmaskimage.fill(Qt.transparent)

        self.selected_item = id
        for items in self.rectlist:
            if items.id == id:
                self.parent.parent.render_index = self.rectlist.index(items)
                print(items.render_index)
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

                    if i.id == x.id:
                        print(f"setting render index to:{self.rectlist.index(i)}")

                        self.parent.parent.render_index = self.rectlist.index(i)
        filename = "outpaint.png"
        outpaintimage.save(filename)
        while not os.path.exists(filename):
            time.sleep(0.25)
        filename = "outpaint_mask.png"
        outpaintmaskimage.save(filename)
        while not os.path.exists(filename):
            time.sleep(0.25)
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
                #self.update()
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

        outpaintmaskimage.save("outpaint_mask.png")
        self.outpaintsource = "outpaint.png"
        self.redo = True
        self.render_item = self.selected_item
        self.signals.update_params.emit(id)

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

        outpaintimage.save("outpaint.png")
        outpaintmaskimage.save("outpaint_mask.png")
        self.outpaintsource = "outpaint.png"
        self.signals.outpaint_signal.emit()

    def mousePressEvent(self, e):
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
        self.redraw_transparent = transparent
        self.signals.run_redraw.emit()

    def redraw_signal(self):
        transparent = self.redraw_transparent
        if transparent:
            self.pixmap.fill(Qt.transparent)
        else:
            self.pixmap.fill(__backgroudColor__)
            self.draw_rects()
        self.painter.begin(self.pixmap)
        self.painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        if self.rectlist is not [] and self.rectlist is not None:
            for i in self.rectlist:
                if i.image is not None and i.active == True:
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

        self.painter.end()

        self.bgitem.setPixmap(self.pixmap)
        self.newimage = False
        self.scene.update()

    def old_paintEvent(self):

        help2 = "None"
        if self.hover_item is not None:
            for items in self.rectlist:
                if items.id == self.hover_item:
                    help2 = f"hover item details:{items.x}, {items.y}"


    def generic_mouseMoveEvent(self, e):

        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)

    def undoEvent(self):
        if self.undoitems is not []:
            x = len(self.undoitems) - 1
            self.rectlist.append(self.undoitems[x])
            self.undoitems.pop(x)
            self.newimage = True


    def select_mode(self):
        self.mode = "select"
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        try:
            self.scene.addItem(self.rectItem)
        except:
            pass
        self.redraw()


    def move_mode(self):
        self.mode = "move"
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

    def enterEvent(self, event):
        self.bgitem.update()
        self.rectItem.update()


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
        if e.key() == 67:
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
            self.mode = "move"
            print("move mode")
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        elif e.key() == 90 and self.ctrlmodifier == True:
            self.undoEvent()
        elif e.key() == 16777249:
            self.ctrlmodifier = True
        else:
            self.ctrlmodifier = None

    def keyReleaseEvent(self, e):
        self.ctrlmodifier = False
        super(Canvas, self).keyReleaseEvent(e)

    def select_mousePressEvent(self, e):
        if self.hover_item == self.selected_item:

            return
        if self.hover_item is not None:
            self.selected_item = self.hover_item
            for i in self.rectlist:
                if i.id == self.selected_item:
                    self.parent.parent.render_index = self.rectlist.index(i)
            self.signals.update_selected.emit()

            self.drawRect()
            self.newimage = True

        else:
            pass


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

    def generic_mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

    def drag_mouseMoveEvent(self, e):
        return

    def drag_mouseReleaseEvent(self, event):
        return

    def drag_mousePressEvent(self, event):
        return

    def move_mouseMoveEvent(self, e):
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
        self.update()
        return

    def move_mouseReleaseEvent(self, event):
        return

    def move_mousePressEvent(self, event):
        self.move_action()
        return

    def move_action(self):
        if self.selected_item is not None:
            for i in self.rectlist:
                if i.id == self.selected_item:
                    i.x = self.scene.scenePos.x() - i.w / 2
                    i.y = self.scene.scenePos.y() - i.h / 2
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
            id = self.addrect_atpos(x=self.scene.scenePos.x() - self.w / 2, y=self.scene.scenePos.y() - self.h / 2, params=copy.deepcopy(self.parent.parent.params))
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
            self.signals.update_selected.emit()
            self.redraw()
            if self.ctrlmodifier == True:
                self.reusable_outpaint(self.selected_item)
                self.signals.outpaint_signal.emit()
                return
            elif self.ctrlmodifier == False:
                self.drag_mode()

        return

    def wheelEvent(self, event):

        x = event.angleDelta().y() / 120
        if x > 0:
            self.zoom *= 1.05
            self.updateView()
        elif x < 0:
            self.zoom /= 1.05
            self.updateView()

    def updateView(self):
        self.setTransform(QTransform().scale(self.zoom, self.zoom).rotate(self.rotate))

    def dragEnterEvent(self, event):
        event.acceptProposedAction()
        print(event.mimeData())

class PaintUI(QDockWidget):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"Outpaint")
        self.setAccessibleName(u'outpaintCanvas')
        self.parent = parent
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setBaseSize(QSize(800, 680))
        self.setLayout(QVBoxLayout())

        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        self.W_spinbox = QSpinBox()
        self.W_spinbox.setMinimum(256)
        self.W_spinbox.setMaximum(16000)
        self.W_spinbox.setValue(512)
        self.W_spinbox.setSingleStep(64)
        self.H_spinbox = QSpinBox()
        self.H_spinbox.setMinimum(256)
        self.H_spinbox.setMaximum(16000)
        self.H_spinbox.setValue(512)
        self.H_spinbox.setSingleStep(64)

        self.W = QSlider()
        self.W.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.W.setMinimumSize(QSize(100, 15))
        self.W.setMaximumSize(QSize(1000, 15))
        self.W.setMinimum(512)
        self.W.setMaximum(16000)
        self.W.setValue(512)
        self.W.setPageStep(64)
        self.W.setSingleStep(64)
        self.W.setOrientation(Qt.Horizontal)

        self.H = QSlider()
        self.H.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.H.setMinimumSize(QSize(100, 15))
        self.H.setMaximumSize(QSize(1000, 15))
        self.H.setMinimum(512)
        self.H.setMaximum(16000)
        self.H.setValue(512)
        self.H.setPageStep(64)
        self.H.setSingleStep(64)
        self.H.setOrientation(Qt.Horizontal)
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 0, 5, 0)
        self.canvas = Canvas(self)
        self.widget_2 = QWidget()
        self.horizontalLayout = QHBoxLayout(self.widget_2)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontal")
        self.horizontalLayout.setContentsMargins(5, 0, 5, 0)


        self.horizontalLayout.addWidget(self.W)
        self.horizontalLayout.addWidget(self.W_spinbox)

        self.horizontalLayout.addWidget(self.H)
        self.horizontalLayout.addWidget(self.H_spinbox)

        self.verticalLayout_2.addWidget(self.canvas)
        self.verticalLayout_2.addWidget(self.widget_2)

        self.setWidget(self.dockWidgetContents)


        self.H.valueChanged.connect(self.update_spinners)
        self.W.valueChanged.connect(self.update_spinners)
        self.H_spinbox.valueChanged.connect(self.update_sliders)
        self.W_spinbox.valueChanged.connect(self.update_sliders)
        self.setAcceptDrops(True)
        self.canvas.setAcceptDrops(True)

    def dropEvent(self, event):
        print(event.mimeData())

    def update_spinners(self):
        self.H_spinbox.setValue(self.H.value())
        self.W_spinbox.setValue(self.W.value())

    def update_sliders(self):
        self.H.setValue(int(self.H_spinbox.value()))
        self.W.setValue(int(self.W_spinbox.value()))

    def rectangleDraw(self):
        self.canvas.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)

def spiralOrder(matrix):
    ans = []

    if (len(matrix) == 0):
        return ans

    m = len(matrix)
    n = len(matrix[0])
    seen = [[0 for i in range(n)] for j in range(m)]
    dr = [0, 1, 0, -1]
    dc = [1, 0, -1, 0]
    x = 0
    y = 0
    di = 0
    c = 0
    # Iterate from 0 to R * C - 1

    for i in range(m * n):
        matrix[x][y]['order'] = c
        ans.append(matrix[x][y])
        c += 1
        seen[x][y] = True
        cr = x + dr[di]
        cc = y + dc[di]

        if (0 <= cr and cr < m and 0 <= cc and cc < n and not(seen[cr][cc])):
            x = cr
            y = cc
        else:
            di = (di + 1) % 4
            x += dr[di]
            y += dc[di]
    return ans

def random_path(order, columns):
    templist = []
    newlist = []
    for i in order:
        for x in i:
            templist.append(x["order"])
    x = 0
    c = 0
    steps = len(templist) - 1
    newlist.append(templist[x])
    while templist != []:

        match = False
        while match == False:

            newpair = random.choice(templist)

            if newpair - 1 == templist[x] or newpair + 1 == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1


            elif newpair - columns == templist[x] or newpair + columns == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1

            elif newpair - columns - 1 == templist[x] or newpair + columns + 1 == templist[x]:
                match = True
                newlist.append(newpair)
                templist.pop(x)
                x = newpair
                c += 1

    for i in order:
        for x in i:
            x["order"] = newlist[i]
    return order
