import os
from datetime import datetime

from PySide6.QtCore import Signal, QLine, QPoint, QRectF, QSize, QRect, QLineF, QPointF, QObject, QFile
from PySide6.QtGui import Qt, QColor, QFont, QPalette, QPainter, QPen, QPolygon, QBrush, QPainterPath, QAction, QCursor, \
    QPixmap, QTransform, QDragEnterEvent, QDragMoveEvent, QImage
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget, QSlider, QDockWidget, QMenu, QGraphicsScene, \
    QGraphicsView, QGraphicsItem, QGraphicsWidget, QLabel, QGraphicsPixmapItem, QGraphicsLineItem, QGraphicsRectItem, \
    QGraphicsTextItem, QScrollArea, QHBoxLayout, QLayout, QAbstractScrollArea
from time import gmtime, strftime
import time
from uuid import uuid4

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)


__idleColor__ = QColor(91, 48, 232)
__selColor__ = QColor(255, 102, 102)


class Rectangle(object):
    def __init__(self, x, y, w, h, id):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.image = None
        self.index = None
        self.images = []
        self.order = None
        self.color = __idleColor__
        self.timestring = time.time()
        self.active = True

class Callbacks(QObject):
    outpaint_signal = Signal()
    txt2img_signal = Signal()
    update_selected = Signal()



class Scene(QGraphicsScene):
    def __init__ (self, parent=None):
        QGraphicsScene.__init__ (self, parent)
        self.pos = None
        self.scenePos = None
    def mouseMoveEvent(self, event):
        super(Scene, self).mouseMoveEvent(event)
        self.pos = QPointF(event.screenPos())
        self.scenePos = event.scenePos()

class Canvas(QGraphicsView):

    def __init__(self, parent=None):
        QGraphicsView.__init__(self, parent)
        self.signals = Callbacks()
        self.reset()

    def soft_reset(self):
        self.pixmap = QPixmap(4096, 4096)
        #self.pixmap.fill(__backgroudColor__)
        self.pixmap.fill(Qt.transparent)
        self.bgitem = QGraphicsPixmapItem()
        self.rectItem = QGraphicsRectItem(256, 256, 512, 512)
        self.debugtext = QGraphicsTextItem("0, 0\n")

        #self.helpText = QGraphicsTextItem("C - Hand Drag\nV - Place Rectangles")
        self.bgitem.setPixmap(self.pixmap)
        #self.setPixmap(self.pixmap)


        self.scene.addItem(self.bgitem)
        self.scene.addItem(self.rectItem)
        self.scene.addItem(self.debugtext)


    def reset(self):
        self.zoom = 1
        self.rotate = 0
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.w = 512
        self.h = 512

        self.rectlist = []
        #rect = QRect(QPoint(256, 256), QSize(512, 512))
        #ainter = QPainter()
        #painter.begin(self.pixmap)
        #painter.setPen(Qt.black)
        #p = painter.pen()
        #p.setWidth(4)
        #painter.drawRect(rect)
        #painter.end()

        self.scene = Scene()

        #self.scene.addItem(self.helpText)

        self.last_x, self.last_y = None, None
        self.pen_color = QColor('#000000')
        self.mode = 'drag'
        self.setMouseTracking(True)
        self.painter = QPainter()

        self.update()
        self.setScene(self.scene)



        self.hover_item = None
        self.selected_item = None
        self.outpaintitem = None
        self.outpaintsource = None

        self.rendermode = 1
        #print(self.rendermode)

        self.painter.setRenderHint(QPainter.SmoothPixmapTransform)
        self.painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        self.soft_reset()
        self.fitInView(self.bgitem, Qt.AspectRatioMode.IgnoreAspectRatio)

    def getXScale(self):
        return float(1024)/float(self.width())
    def getYScale(self):
        return float(1024)/float(self.height())

    def set_pen_color(self, c):
        self.pen_color = QColor(c)

    def addrect(self):
        rect = {}
        uid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
        rect[uid] = Rectangle(self.scene.scenePos.x() - self.w / 2, self.scene.scenePos.y() - self.h / 2, self.w, self.h, uid)
        self.selected_item = uid

        self.rectlist.append(rect[uid])

    def drawRect(self, x, y, width=256, height=256):

        #print(f"we are putting that thing to:{x}, {y}, and our width is {self.width()}")
        Xscale = self.getXScale()
        Yscale = self.getYScale()
        scaledWidth = (self.w / 2) / Xscale
        scaledHeight = (self.h / 2) / Yscale
        x = (x - scaledWidth) * Xscale
        y = (y - scaledHeight) * Yscale
        if x < 0: x = 0
        if y < 0: y = 0
        if self.mode == "generic" or "outpaint":
            self.rectItem.setRect(x, y, self.w, self.h)
        else:
            self.rectItem.setRect(self.width(), self.height(), 5, 5)
        self.bgitem.setPixmap(self.pixmap)
        self.update()

    def hoverCheck(self):
        matchFound = False
        for i in self.rectlist:
            #print(i.x)
            if i.x <= self.scene.scenePos.x() <= i.x + i.w and i.y <= self.scene.scenePos.y() <= i.y + i.h:
                #print(f"found{id}")
                i.color = __selColor__
                self.update()
                self.hover_item = i.id
                matchFound = True
            else:
                i.color = __idleColor__
                self.update()
            if not matchFound:
                self.hover_item = None
        self.update()
    def save_canvas(self):
        timestring = time.strftime('%Y-%m-%d-%H-%S')
        filename = f"output/canvas/canvas_{timestring}.png"
        os.makedirs('output/canvas', exist_ok=True)
        file = QFile(filename)
        self.pixmap.save(file, "PNG")
    def first_rectangle(self):
        self.hoverCheck()
        if self.hover_item is None:
            self.addrect()
            self.signals.txt2img_signal.emit()
            self.mode = "outpaint"

    def redo_outpaint(self, id):
        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpainter.begin(outpaintimage)
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
                                #i.color = __selColor__
                                #self.update()
                                    if i.image is not None:
                                        print("Found an image to outpaint")
                                        rect = QRect(x.x - i.x, x.y - i.y, self.w, self.h)
                                        newimage = i.image.copy(rect)
                                        outpainter.drawImage(0,0,newimage)
                                        #self.addrect()
        outpaintimage.save("outpaint.png")
        outpainter.end()
        self.outpaintsource = "outpaint.png"
        self.redo = True
        self.signals.outpaint_signal.emit()

    def region_to_outpaint(self, event):
        outpaintimage = QPixmap(self.w, self.h)
        outpaintimage.fill(Qt.transparent)
        outpainter = QPainter()
        outpainter.begin(outpaintimage)
        outpainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        #outpainter.setRenderHint(QPainter.LosslessImageRendering)
        for i in self.rectlist:
            if (self.scene.scenePos.y() - self.h / 2) >= i.y - self.h and (self.scene.scenePos.x() - self.w / 2) >= i.x - self.w:
                if (self.scene.scenePos.y() - self.h / 2) <= i.y + self.h and (self.scene.scenePos.x() - self.w / 2) <= i.x + i.w:
                    print("Found an image to outpaint")
                    i.color = __selColor__
                    self.update()
                    if i.image is not None and i.active == True:
                        rect = QRect(int((self.scene.scenePos.x() - self.w / 2) - i.x), int((self.scene.scenePos.y() - self.h / 2) - i.y), self.w, self.h)
                        newimage = i.image.copy(rect)
                        outpainter.drawImage(0,0,newimage)
                        self.addrect()
        outpaintimage.save("outpaint.png")
        outpainter.end()
        self.outpaintsource = "outpaint.png"
        self.signals.outpaint_signal.emit()


    def mousePressEvent(self, e):
        #for i in self.rectlist:
        #    print(i.order)

        #self.reset()
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
    def paintEvent(self, e):


        self.painter.begin(self.pixmap)

        #self.pixmap.fill(Qt.black)
        #painter.drawText(0, 50, 256, 256, Qt.AlignHCenter, "C - Hand Drag\nV - Place Rectangles")
        #Show Outpaint Preview rectangle
        #rect = QRect(0, self.height() - 256, 256, 256)
        #painter.drawRect(rect)
        if self.rectlist is not [] and self.rectlist is not None:
            self.rectlist.sort(reverse=False, key=self.sortRects)
            for i in self.rectlist:


                rect = QRect(i.x, i.y, i.w, i.h)
                if i.image is not None and i.active == True:
                    pic = i.image.copy(0, 0, i.image.width(), i.image.height())
                    pixmap = QPixmap.fromImage(pic)
                    #painter.drawImage(QRect(QPoint(i.x, i.y), QSize(i.w, i.h)), pic)



                    self.painter.drawPixmap(int(i.x), int(i.y), i.w, i.h, pixmap, 0, 0, i.w, i.h)
                    #self.drawForeground(painter, rect)
                    #painter.drawPixmap()
                else:
                    if self.mode == "generic" or self.mode == "outpaint":
                        self.painter.setPen(i.color)
                        self.painter.drawRect(rect)
        self.painter.end()

        self.bgitem.setPixmap(self.pixmap)

        #self.setSceneRect(QRectF(self.viewport().rect()))

        self.update()
        help2 = "None"
        if self.hover_item is not None:
            for items in self.rectlist:
                if items.id == self.hover_item:
                    help2 = f"hover item details:{items.x}, {items.y}"


        self.debugtext.setPlainText(f"{self.scene.pos}, {self.scene.scenePos}\nHover Item: {self.hover_item}\nSelected Item:{self.selected_item}\nMode: {self.mode}\n{help2}\n\n\n\n"
                                    f"-"
                                    f"-"
                                    f"-"
                                    f"-"
                                    f"-"
                                    f"-"
                                    f"-\n"
                                    f"--")
        super(Canvas, self).paintEvent(e)

    def generic_mouseMoveEvent(self, e):
        if self.scene.pos is not None:

            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)

        self.update()
    def keyPressEvent(self, e):
        super(Canvas, self).keyPressEvent(e)
        print(f"key pressed: {e.key()}")
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
    def keyReleaseEvent(self, e):
        super(Canvas, self).keyReleaseEvent(e)

    def select_mousePressEvent(self, e):
        if self.hover_item is not None:
            self.selected_item = self.hover_item
            self.signals.update_selected.emit()
            #self.scene.clear()
            self.pixmap.fill(Qt.black)
            self.pixmap.fill(Qt.transparent)
        else:
            self.selected_item = None

    def select_mouseMoveEvent(self, e):
        self.hoverCheck()
        self.update()
    def select_mouseReleaseEvent(self, event):
        return
    def generic_mousePressEvent(self, e):
        self._start = e.pos()
        self.posx = e.pos().x()
        self.posy = e.pos().y()
        self.first_rectangle()
        return

    def generic_mouseReleaseEvent(self, event):
        start = QPointF(self.mapToScene(self._start))
        end = QPointF(self.mapToScene(event.pos()))
        self.scene.addItem(
            QGraphicsLineItem(QLineF(start, end)))
        return

    """def paintEvent(self, e):
        painter = QPainter()
        painter.begin(self.pixmap)

        rect = QRect(QPoint(self.posx, self.posy), QSize(512, 512))
        painter.setPen(Qt.black)
        p = painter.pen()
        p.setWidth(4)
        painter.drawRect(rect)
        painter.end()


        #self.setPixmap(self.pixmap)
        #self.update()"""

    def generic_mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None
    def drag_mouseMoveEvent(self, e):
        return
    def drag_mouseReleaseEvent(self, event):
        return
    def drag_mousePressEvent(self, event):
        return
    def outpaint_mouseMoveEvent(self, e):
        if self.scene.pos is not None:
            self.drawRect(self.scene.scenePos.x() / self.getXScale(), self.scene.scenePos.y() / self.getYScale(), self.w, self.h)
        self.update()
        return
    def outpaint_mouseReleaseEvent(self, event):
        return
    def outpaint_mousePressEvent(self, event):
        self.region_to_outpaint(event)
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

"""class Outpaint(QGraphicsWidget):

    keyFramesUpdated = Signal()
    #selectionChanged = Signal(VideoSample)

    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene()
        self.canvas = QPixmap(4096, 4096)
        self.scene.addPixmap(self.canvas)

        self.duration = 1000
        self.length = 1000

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.clicking = False  # Check if mouse left button is being pressed
        self.is_in = False  # check if user is in the widget
        #self.setMouseTracking(False)  # Mouse events
        #self.setAutoFillBackground(True)  # background
        self.pos = None

        self.posy = None
        self.rectangle = None
        self.rectList = []
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, self.length, 200)
        self.setWindowTitle("TESTE")

        # Set Background
        pal = QPalette()
        pal.setColor(QPalette.Base, self.backgroundColor)
        self.setPalette(pal)

    def rectangleDraw(self):
        self.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)

    def paintEvent(self, event):

        qp = QPainter(self.canvas)
        qp.device()
        #qp.begin()
        qp.setPen(self.textColor)
        qp.setFont(self.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0
        # Draw time
        scale = self.getScale()
        qp.setPen(QPen(Qt.darkGreen, 5, Qt.SolidLine))
        #qp.drawLine(0, 500, self.width(), 40)
        if self.rectangle is not None:
            qp.drawRect(self.rectangle)


        #qp.end()

    # Mouse movement
    def mouseMoveEvent(self, e):

        print("5135435")

        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer
        self.pos = e.pos()
        #self.rectangleDraw()

        if self.clicking:
            pass
            #self.oldPos = self.pointerPos
            #x = self.pos
            #self.pointerPos = x
            #self.pointerTimePos = self.pointerPos*self.getScale()





        self.update()

    # Mouse pressed

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()

            #print(self.keyClicked)
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.clicking = True  # Set clicking check to true

        elif e.button() == Qt.RightButton:
            self.popMenu = QMenu(self)
            menuPosition = QCursor.pos()
            self.popMenu.clear()
            #populate
            self.populateBtnContext()
            #show
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            #self.popMenu.delete_action.triggered.connect(self.delete_action)

    def populateBtnContext(self):

        # Do some if here :
        self.popMenu.delete_action = QAction('delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)

    # Mouse release


    def delete_action(self):
        pass
    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false

    # Enter
    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    # check selection



    # Get time string from seconds
    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        #return "%02d:%02d:%02d" % (h, m, s)
        return "%05d" % (seconds)


    # Get scale from length
    def getScale(self):
        #return float(self.duration)/float(self.width())
        pass


    # Set background color
    def setBackgroundColor(self, color):
        self.backgroundColor = color

    # Set text color
    def setTextColor(self, color):
        self.textColor = color

    # Set Font
    def setTextFont(self, font):
        self.font = font"""








class OutpaintUI(QDockWidget):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        #self.setWindowModality(Qt.WindowModal)
        self.setMouseTracking(True)  # Mouse events

        #self.resize(759, 544)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setBaseSize(QSize(800, 680))
        self.setLayout(QVBoxLayout())
        #self.verticalLayout = QVBoxLayout(self)
        #self.dockWidget = QDockWidget(self)
        #self.dockWidget.setObjectName(u"dockWidget")
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        #self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(5, 0, 5, 0)
        self.canvas = Canvas(self)
        #self.item = QGraphicsItem(self.outpaint.canvas)

        #self.graphicsView = QGraphicsView()


        #self.graphicsView.setScene(self.outpaint.scene)

        #self.sample_1 = VideoSample(20)
        #self.timeline.videoSamples.append(self.sample_1)
        self.tZoom = QSlider()

        self.tZoom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tZoom.setMinimumSize(QSize(100, 15))
        self.tZoom.setMaximumSize(QSize(1000, 15))
        self.tZoom.setMinimum(0.0)
        self.tZoom.setMaximum(10.0)
        self.tZoom.setValue(1.0)
        self.tZoom.setOrientation(Qt.Horizontal)
        self.verticalLayout_2.addWidget(self.canvas)
        #self.verticalLayout_2.addWidget(self.tZoom)

        self.setWidget(self.dockWidgetContents)
        #self.tZoom.valueChanged.connect(self.update_timelineZoom)
        #self.timeline.scale = 1

        #self.verticalLayout_2.addWidget(self.viewport)

    def rectangleDraw(self):
        self.outpaint.rectangle = QRect(self.pos.x(), self.pos.y(), 400, 400)

    """def mouseMoveEvent(self, e):
        print('event is still there...')
        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer
        self.pos = e.pos()
        self.rectangleDraw()

        #if self.clicking:
            #pass
    def paintEvent(self, event):

        qp = QPainter(self.outpaint.canvas)
        #qp.device()
        qp.begin(self.outpaint.canvas)
        qp.setPen(self.outpaint.textColor)
        qp.setFont(self.outpaint.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0
        # Draw time
        #scale = self.outpaint.getScale()
        qp.setPen(QPen(Qt.darkGreen, 5, Qt.SolidLine))
        #qp.drawLine(0, 500, self.width(), 40)
        if self.outpaint.rectangle is not None:
            qp.drawRect(self.outpaint.rectangle)
        qp.end()
class Outpaint(QWidget):

    #keyFramesUpdated = Signal()
    #selectionChanged = Signal(VideoSample)

    def __init__(self):
        super().__init__()

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.rectangle = None
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, 800, 600)
        self.setWindowTitle("TESTE")

        # Set Background
        pal = QPalette()
        pal.setColor(QPalette.Base, self.backgroundColor)
        self.setPalette(pal)

    def paintEvent(self, event):

        qp = QPainter()
        #qp.device()
        qp.begin(self)
        qp.setPen(self.textColor)
        qp.setFont(self.font)
        qp.setRenderHint(QPainter.Antialiasing)
        w = 0


        # Draw down line
        qp.setPen(QPen(Qt.darkCyan, 5, Qt.SolidLine))
        qp.drawLine(0, 500, self.width(), 40)
        if self.rectangle is not None:
            qp.drawRect(self.rectangle)
        qp.end()
        # Draw dash lines

    def mouseMoveEvent(self, e):
        print("outpaint move event")
        print(e.pos().y(), e.pos().x())


        self.pos = e.pos()
        self.rectangleDraw()
        self.update()

    def mousePressEvent(self, e):
        if self.is_in == True:
            print('mousePressEvent while in')
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()
            self.clicking = False
        elif e.button() == Qt.RightButton:
            pass



    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false



    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    def getScale(self):
        return float(self.width())


        #self.update()
    def setTextFont(self, font):
        self.font = font
        
        point = 0
        qp.setPen(QPen(self.textColor))
        qp.drawLine(0, 40, self.width(), 40)
        while point <= self.width():
            if point % 30 != 0:
                qp.drawLine(3 * point, 40, 3 * point, 30)
            else:
                qp.drawLine(3 * point, 40, 3 * point, 20)
            point += 10

        if self.pos is not None and self.is_in:
            qp.drawLine(self.pos, 0, self.pos, 40)

        if self.pointerPos is not None:
            self.pointerTimePos = int(self.pointerTimePos)
            line = QLine(QPoint(int(self.pointerTimePos/self.getScale()), 40),
                         QPoint(int(self.pointerTimePos/self.getScale()), self.height()))
            poly = QPolygon([QPoint(int(self.pointerTimePos/self.getScale() - 10), 20),
                             QPoint(int(self.pointerTimePos/self.getScale() + 10), 20),
                             QPoint(int(self.pointerTimePos/self.getScale()), 40)])
        else:
            line = QLine(QPoint(0, 0), QPoint(0, self.height()))
            poly = QPolygon([QPoint(-10, 20), QPoint(10, 20), QPoint(0, 40)])

        if self.selectedValueType is not None:
            for i in self.keyFrameList:
                if i is not None:
                    if i.valueType == self.selectedValueType:

                        kfbrush = QBrush(Qt.darkRed)
                        kfStartPoint = int(int(i.position) / self.getScale())
                        scaleMod = 5
                        kfPoly = QPolygon([QPoint(int(kfStartPoint - scaleMod), 50), QPoint(kfStartPoint, 45), QPoint(kfStartPoint + scaleMod, 50), QPoint(kfStartPoint, 55)])
                        qp.setPen(Qt.darkRed)
                        qp.setBrush(kfbrush)
                        qp.drawPolygon(kfPoly)

        # Draw samples
        t = 0
        for sample in self.videoSamples:
            # Clear clip path
            path = QPainterPath()

            path.addRoundedRect(QRectF((t + sample.startPos)/scale, 50, sample.duration / scale, 200), 10, 10)

            qp.setClipPath(path)

            # Draw sample
            path = QPainterPath()
            qp.setPen(sample.color)
            qp.setBrush(sample.color)



            #path.addRoundedRect(QRectF(((t + sample.startPos)/scale), 50, (sample.duration / scale), 50), 10, 10)
            path.addRect((t + sample.startPos)/scale, 50, (sample.duration / scale), 50)
            #sample.startPos = (t + sample.startPos)*scale
            sample.endPos = (t + sample.startPos)/scale + sample.duration/scale
            qp.fillPath(path, sample.color)
            qp.drawPath(path)


            # Draw preview pictures
            if sample.picture is not None:
                if sample.picture.size().width() < sample.duration/scale:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t/scale, 52.5, sample.picture.size().width(), 45), 10, 10)
                    qp.setClipPath(path)
                    qp.drawPixmap(QRect(int(t/scale), 52.5, sample.picture.size().width(), 45), sample.picture)
                else:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t / scale, 52.5, sample.duration/scale, 45), 10, 10)
                    qp.setClipPath(path)
                    pic = sample.picture.copy(0, 0, sample.duration/scale, 45)
                    qp.drawPixmap(QRect(int(t / scale), 52.5, sample.duration/scale, 45), pic)
            t += sample.duration

        # Clear clip path
        path = QPainterPath()
        path.addRect(self.rect().x(), self.rect().y(), self.rect().width(), self.rect().height())
        qp.setClipPath(path)

        # Draw pointer
        qp.setPen(Qt.darkCyan)
        qp.setBrush(QBrush(Qt.darkCyan))

        qp.drawPolygon(poly)
        qp.drawLine(line)
        qp.end()

    # Mouse movement
    def mouseMoveEvent(self, e):



        self.pos = e.pos().x()
        self.posy = e.pos().y()

        # if mouse is being pressed, update pointer

        self.checkKeyframeHover(self.pos)

        if self.clicking:

            self.oldPos = self.pointerPos
            x = self.pos
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos*self.getScale()


            if self.keyHover == True:
                for item in self.keyFrameList:
                    if self.selectedKey is item.uid:
                        item.position = int(self.pointerPos*self.scale)
            if self.edgeGrabActive == True:


                for sample in self.videoSamples:
                    sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
            elif self.middleHoverActive == True:

                self.scale = self.getScale()
                for sample in self.videoSamples:
                    change = (x - self.oldPos)
                    change = (change * self.scale)
                    #print(change)
                    sample.startPos = sample.startPos + change
                    sample.endPos = sample.endPos + change



        self.update()

    # Mouse pressed
    def checkKeyframeHover(self, x):
        for item in self.keyFrameList:
            kfStartPoint = int(int(item.position) / self.getScale())
            if kfStartPoint - 5 < x < kfStartPoint + 5 and 55 > self.posy > 45:
                self.keyHover = True
                #print(item.uid)
                self.hoverKey = item.uid

    def checkKeyClicked(self):
        for item in self.keyFrameList:
            if self.hoverKey is item.uid:
                self.selectedKey = self.hoverKey
                self.keyHover = True

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()
            self.checkKeyClicked()

            #print(self.keyClicked)
            self.pointerPos = x
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.clicking = True  # Set clicking check to true
            if self.edgeGrab == True:
                self.edgeGrabActive = True
            else:
                self.edgeGrabActive = False
            if self.middleHover == True:
                self.middleHoverActive = True
            else:
                self.middleHoverActive = False
        elif e.button() == Qt.RightButton:
            self.popMenu = QMenu(self)
            menuPosition = QCursor.pos()
            self.checkKeyClicked()
            self.popMenu.clear()
            #populate
            self.populateBtnContext()
            #show
            self.popMenu.move(menuPosition)
            self.popMenu.show()
            self.popMenu.delete_action.triggered.connect(self.delete_action)

    def populateBtnContext(self):

        # Do some if here :
        self.popMenu.delete_action = QAction('delete keyframe', self)
        self.popMenu.addAction(self.popMenu.delete_action)

    # Mouse release


    def delete_action(self):
        for idx, item in enumerate(self.keyFrameList):
            print(idx)
            print(item)
            if self.hoverKey is item.uid:
                self.keyFrameList.pop(idx)

                #item.remove()
                #return
    def mouseReleaseEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            self.clicking = False  # Set clicking check to false
            self.selectedKey = None
            self.keyHover = False
            self.hoverKey = None
            self.keyFramesUpdated.emit()

    # Enter
    def enterEvent(self, e):
        self.is_in = True

    # Leave
    def leaveEvent(self, e):
        self.is_in = False
        self.update()

    # check selection
    def checkSelection(self, x):
        # Check if user clicked in video sample
        for sample in self.videoSamples:
            if sample.startPos + 25 < x < sample.endPos - 25:
                sample.color = Qt.darkCyan
                self.middleHover = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.middleHover = False


    def checkEdges(self, x, y=50):

        for sample in self.videoSamples:
            if sample.startPos < x < sample.startPos + 24:
                sample.color = Qt.darkMagenta
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            elif sample.endPos - 24 < x < sample.endPos:
                sample.color = Qt.darkGreen
                self.edgeGrab = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    #self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.edgeGrab = False


    # Get time string from seconds
    def get_time_string(self, seconds):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        #return "%02d:%02d:%02d" % (h, m, s)
        return "%05d" % (seconds)


    # Get scale from length
    def getScale(self):
        return float(self.duration)/float(self.width())

    # Get duration
    def getDuration(self):
        return self.duration

    # Get selected sample
    def getSelectedSample(self):
        return self.selectedSample

    # Set background color
    def setBackgroundColor(self, color):
        self.backgroundColor = color

    # Set text color
    def setTextColor(self, color):
        self.textColor = color

    # Set Font
    def setTextFont(self, font):
        self.font = font"""