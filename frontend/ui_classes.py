#from PyQt6 import uic
from PySide6 import QtUiTools
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtWidgets import (QDockWidget, QGraphicsScene, QGraphicsPixmapItem,
                              QGraphicsView, QWidget, QSizePolicy, QSlider, QPushButton,
                              QAbstractItemView, QListView, QListWidget, QVBoxLayout,
                               QMenu)
from PySide6.QtGui import QAction, QPainter, QPainterPath, QPolygon, QPen
from PySide6.QtCore import (QMetaObject, QFile, QRectF,
                            QCoreApplication, QSize, Qt,
                            QEvent, QObject, QTimeLine, QPoint, QLine, Signal, QRect)

from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)

import concurrent.futures
class AnimKeyEditor(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/keyframe_editor.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)

class AnimKeys(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/animKeys.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)


class AnimDials(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/animDials.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)


class SizerCount(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/sizer_count.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)
class Dynaimage(QObject):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/dynaimage.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class Dynaview(QObject):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/dynaview.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class Sampler(QObject):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/sampler.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class Prompt(QObject):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/prompt.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/prompt.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class Anim(QObject):
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/anim.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/anim.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class PhotoViewer(QGraphicsView):

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        #self.setDragMode(QtWidgets.QGraphicsView.RubberBandDrag)
        #self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        #self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        #self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        #self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        #self.setFrameShape(QtWidgets.QFrame.NoFrame)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()


    """def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0"""

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def mousePressEvent(self, event):
        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)



class Preview(QObject):
    loader = QtUiTools.QUiLoader()
    file = QFile("frontend/ui_widgets/preview.ui")
    file.open(QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/preview.ui", self)

        self._zoom = 0

        #self.graphicsView = PhotoViewer(self)
        #self.scene = QGraphicsScene()
        #self.graphicsView.setScene(self.scene)
        self.zoom = 1
        self.rotate = 0

    def fitInView(self, *args, **kwargs):
        super().fitInView(*args, **kwargs)
        self.zoom = self.transform().m11()



    def updateView(self):
        self.graphicsView.scale(self.zoom, self.zoom).rotate(self.rotate)

    def wheelEvent(self, event):

        if event.angleDelta().y() > 0:
            factor = 1.25
            self._zoom += 1
        else:
            factor = 0.8
            self._zoom -= 1
        if self._zoom > 0:
            self.graphicsView.scale(factor, factor)
        elif self._zoom == 0:
            self.graphicsView.fitInView()
            self._zoom = 1
        else:
            self._zoom = 0


class Thumbnails2(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        self.setWindowModality(Qt.WindowModal)
        #self.resize(759, 544)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setBaseSize(QSize(800, 680))
        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.dockWidget = QDockWidget(self)
        self.dockWidget.setObjectName(u"dockWidget")
        sizePolicy.setHeightForWidth(self.dockWidget.sizePolicy().hasHeightForWidth())
        self.dockWidget.setSizePolicy(sizePolicy)
        self.dockWidgetContents = QWidget()
        self.dockWidgetContents.setObjectName(u"dockWidgetContents")
        sizePolicy.setHeightForWidth(self.dockWidgetContents.sizePolicy().hasHeightForWidth())
        self.dockWidgetContents.setSizePolicy(sizePolicy)
        self.verticalLayout_2 = QVBoxLayout(self.dockWidgetContents)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.thumbs = QListWidget(self.dockWidgetContents)
        self.thumbs.setObjectName(u"thumbs")
        sizePolicy.setHeightForWidth(self.thumbs.sizePolicy().hasHeightForWidth())
        self.thumbs.setSizePolicy(sizePolicy)
        self.thumbs.setBaseSize(QSize(800, 680))
        self.thumbs.setFocusPolicy(Qt.NoFocus)
        #self.thumbs.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.thumbs.setAcceptDrops(False)
        self.thumbs.setToolTip(u"")
        self.thumbs.setAccessibleName(u"")
        self.thumbs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thumbs.setProperty("showDropIndicator", False)
        self.thumbs.setDragEnabled(False)
        self.thumbs.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.thumbs.setDefaultDropAction(Qt.IgnoreAction)
        self.thumbs.setIconSize(QSize(150, 150))

        self.thumbs.setMovement(QListView.Free)
        self.thumbs.setResizeMode(QListView.Adjust)
        self.thumbs.setLayoutMode(QListView.Batched)
        self.thumbs.setGridSize(QSize(150, 200))
        self.thumbs.setViewMode(QListView.IconMode)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setWordWrap(True)
        self.thumbs.setSelectionRectVisible(False)
        self.thumbs.setSortingEnabled(False)
        self.verticalLayout_2.addWidget(self.thumbs)

        self.refresh = QPushButton(self.dockWidgetContents)
        self.refresh.setObjectName(u"refresh")

        self.verticalLayout_2.addWidget(self.refresh)

        self.thumbsZoom = QSlider(self.dockWidgetContents)
        self.thumbsZoom.setObjectName(u"thumbsZoom")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.thumbsZoom.sizePolicy().hasHeightForWidth())
        self.thumbsZoom.setSizePolicy(sizePolicy1)
        self.thumbsZoom.setMinimumSize(QSize(0, 15))
        self.thumbsZoom.setMinimum(5)
        self.thumbsZoom.setMaximum(512)
        self.thumbsZoom.setValue(150)
        self.thumbsZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.thumbsZoom)

        self.dockWidget.setWidget(self.dockWidgetContents)

        self.verticalLayout.addWidget(self.dockWidget)


        QMetaObject.connectSlotsByName(self)

class Thumbnails(QDockWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        self.setWindowModality(Qt.WindowModal)
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
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.thumbs = QListWidget(self.dockWidgetContents)
        self.thumbs.setObjectName(u"thumbs")
        sizePolicy.setHeightForWidth(self.thumbs.sizePolicy().hasHeightForWidth())
        self.thumbs.setSizePolicy(sizePolicy)
        self.thumbs.setBaseSize(QSize(800, 680))
        self.thumbs.setFocusPolicy(Qt.NoFocus)
        #self.thumbs.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.thumbs.setAcceptDrops(False)
        self.thumbs.setToolTip(u"")
        self.thumbs.setAccessibleName(u"")
        self.thumbs.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.thumbs.setProperty("showDropIndicator", False)
        self.thumbs.setDragEnabled(False)
        self.thumbs.setDragDropMode(QAbstractItemView.NoDragDrop)
        self.thumbs.setDefaultDropAction(Qt.IgnoreAction)
        self.thumbs.setIconSize(QSize(150, 150))
        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setMovement(QListView.Free)
        self.thumbs.setResizeMode(QListView.Adjust)
        self.thumbs.setLayoutMode(QListView.Batched)
        self.thumbs.setGridSize(QSize(150, 200))
        self.thumbs.setViewMode(QListView.IconMode)
        self.thumbs.setUniformItemSizes(True)
        self.thumbs.setWordWrap(True)
        self.thumbs.setSelectionRectVisible(False)
        self.thumbs.setSortingEnabled(False)
        self.verticalLayout_2.addWidget(self.thumbs)

        self.refresh = QPushButton(self.dockWidgetContents)
        self.refresh.setObjectName(u"refresh")

        self.verticalLayout_2.addWidget(self.refresh)

        self.thumbsZoom = QSlider(self.dockWidgetContents)
        self.thumbsZoom.setObjectName(u"thumbsZoom")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.thumbsZoom.sizePolicy().hasHeightForWidth())
        self.thumbsZoom.setSizePolicy(sizePolicy1)
        self.thumbsZoom.setMinimumSize(QSize(100, 15))
        self.thumbsZoom.setMaximumSize(QSize(1000, 15))

        self.thumbsZoom.setMinimum(5)
        self.thumbsZoom.setMaximum(512)
        self.thumbsZoom.setValue(150)
        self.thumbsZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.thumbsZoom)

        self.setWidget(self.dockWidgetContents)
        self.thumbs.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.thumbs.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        #self.verticalLayout.addWidget(self)


        QMetaObject.connectSlotsByName(self)



class VideoSample:

    def __init__(self, duration, color=Qt.darkYellow, picture=None, audio=None):
        self.duration = duration
        self.color = color  # Floating color
        self.defColor = color  # DefaultColor
        #self.position = 0
        if picture is not None:
            self.picture = picture.scaledToHeight(45)
        else:
            self.picture = None
        self.startPos = 0  # Inicial position
        self.endPos = self.duration  # End position

__textColor__ = QColor(187, 187, 187)
__backgroudColor__ = QColor(60, 63, 65)
__font__ = QFont('Decorative', 10)

class OurTimeline(QWidget):

    positionChanged = Signal(int)
    selectionChanged = Signal(VideoSample)

    def __init__(self, duration, length):
        super().__init__()
        self.duration = duration
        self.length = length

        # Set variables
        self.backgroundColor = __backgroudColor__
        self.textColor = __textColor__
        self.font = __font__
        self.pos = None
        self.oldPos = None
        self.pointerPos = None
        self.pointerTimePos = None
        self.selectedSample = None
        self.clicking = False  # Check if mouse left button is being pressed
        self.is_in = False  # check if user is in the widget
        self.videoSamples = []  # List of videos samples
        self.middleHover  = False
        self.setMouseTracking(True)  # Mouse events
        self.setAutoFillBackground(True)  # background
        self.edgeGrab = False
        self.scale = None
        self.middleHoverActive = False
        self.keyFrameList = {}
        self.initUI()

    def initUI(self):

        self.setGeometry(300, 300, self.length, 200)
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
        # Draw time
        scale = self.getScale()
        while w <= self.width():
            qp.drawText(w - 50, 0, 100, 100, Qt.AlignHCenter, self.get_time_string(w * scale))
            w += 100
        # Draw down line
        qp.setPen(QPen(Qt.darkCyan, 5, Qt.SolidLine))
        qp.drawLine(0, 40, self.width(), 40)
        # Draw dash lines
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
            line = QLine(QPoint(self.pointerTimePos/self.getScale(), 40),
                         QPoint(self.pointerTimePos/self.getScale(), self.height()))
            poly = QPolygon([QPoint(self.pointerTimePos/self.getScale() - 10, 20),
                             QPoint(self.pointerTimePos/self.getScale() + 10, 20),
                             QPoint(self.pointerTimePos/self.getScale(), 40)])
        else:
            line = QLine(QPoint(0, 0), QPoint(0, self.height()))
            poly = QPolygon([QPoint(-10, 20), QPoint(10, 20), QPoint(0, 40)])


        """for keys in self.keyFrameList:
            for start in self.keyFrameList[keys].items():
                if start is not None:
                    kfbrush = QBrush(Qt.darkRed)
                    kfStartPoint = int(start) / self.getScale()
                    scaleMod = 5
                    kfPoly = QPolygon([QPoint(kfStartPoint - scaleMod, 50), QPoint(kfStartPoint, 45), QPoint(kfStartPoint + scaleMod, 50), QPoint(kfStartPoint, 55)])
                    qp.setPen(Qt.darkRed)
                    qp.setBrush(kfbrush)
                    qp.drawPolygon(kfPoly)"""

        # Draw samples
        t = 0
        for sample in self.videoSamples:
            # Clear clip path
            path = QPainterPath()

            path.addRoundedRect(QRectF((t + sample.startPos)/scale, 50, sample.duration / scale, 200), 10, 10)

            path.addRoundedRect
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
                    qp.drawPixmap(QRect(t/scale, 52.5, sample.picture.size().width(), 45), sample.picture)
                else:
                    path = QPainterPath()
                    path.addRoundedRect(QRectF(t / scale, 52.5, sample.duration/scale, 45), 10, 10)
                    qp.setClipPath(path)
                    pic = sample.picture.copy(0, 0, sample.duration/scale, 45)
                    qp.drawPixmap(QRect(t / scale, 52.5, sample.duration/scale, 45), pic)
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
        self.checkEdges(self.pos)
        #print(f'mouseMove func: {self.edgeGrab}')
        # if mouse is being pressed, update pointer
        if self.clicking:
            self.oldPos = self.pointerPos
            x = self.pos


            self.pointerPos = x
            self.positionChanged.emit(x)
            self.checkSelection(x)


            self.pointerTimePos = self.pointerPos*self.getScale()
            if self.edgeGrabActive == True:


                for sample in self.videoSamples:
                        sample.duration = sample.duration + ((self.pointerPos - self.oldPos) * self.scale)
            elif self.middleHoverActive == True:

                self.scale = self.getScale()
                for sample in self.videoSamples:
                    #print(f'BEGIN OF DEBUG BLOCK')
                    #print(f'x:{x}')
                    #print(f'scale:{self.scale}')
                    #print(f'oldpos:{self.oldPos}')
                    #print(f'sample.startPos:{sample.startPos}')
                    #print(f'x:{x}')

                    #print(f'END OF DEBUG BLOCK')

                    change = (x - self.oldPos)
                    change = (change * self.scale)
                    #print(change)
                    sample.startPos = sample.startPos + change
                    sample.endPos = sample.endPos + change



        self.update()

    # Mouse pressed

    def mousePressEvent(self, e):
        self.scale = self.getScale()
        if e.button() == Qt.LeftButton:
            x = e.pos().x()

            self.pointerPos = x

            self.positionChanged.emit(x)
            self.pointerTimePos = self.pointerPos * self.getScale()

            self.checkSelection(x)
            self.checkEdges(x)

            self.update()
            self.clicking = True  # Set clicking check to true
            if self.edgeGrab == True:
                self.edgeGrabActive = True
            else:
                self.edgeGrabActive = False
            if self.middleHover == True:
                self.middleHoverActive = True
            else:
                self.middleHoverActive = False
    # Mouse release
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
    def checkSelection(self, x):
        # Check if user clicked in video sample
        for sample in self.videoSamples:
            if sample.startPos + 25 < x < sample.endPos - 25:
                sample.color = Qt.darkCyan
                self.middleHover = True
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    self.selectionChanged.emit(sample)
            else:
                sample.color = sample.defColor
                self.middleHover = False



    def checkEdges(self, x, y=50):
        # Check if user clicked in video sample
        for sample in self.videoSamples:
            if sample.startPos < x < sample.startPos + 24:
                sample.color = Qt.darkMagenta
                if self.selectedSample is not sample:
                    self.selectedSample = sample
                    self.selectionChanged.emit(sample)

            elif sample.endPos - 24 < x < sample.endPos:
                    sample.color = Qt.darkGreen
                    self.edgeGrab = True



                    if self.selectedSample is not sample:
                        self.selectedSample = sample
                        self.selectionChanged.emit(sample)


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
        self.font = font

class Test(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeline = QTimeLine(1000)
class Timeline(QDockWidget):


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        self.setWindowModality(Qt.WindowModal)
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
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.timeline = OurTimeline(1000, 1000)

        self.sample_1 = VideoSample(20)
        self.timeline.videoSamples.append(self.sample_1)
        self.tZoom = QSlider()
        self.tZoom.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tZoom.setMinimumSize(QSize(100, 15))
        self.tZoom.setMaximumSize(QSize(1000, 15))
        self.tZoom.setMinimum(0.0)
        self.tZoom.setMaximum(10.0)
        self.tZoom.setValue(1.0)
        self.tZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.timeline)
        self.verticalLayout_2.addWidget(self.tZoom)

        self.setWidget(self.dockWidgetContents)
        self.tZoom.valueChanged.connect(self.update_timelineZoom)
        #self.timeline.scale = 1

        #self.timeline.timeline.start()
    def update_timelineZoom(self):
        self.timeline.scale = self.tZoom.value()



