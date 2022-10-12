#from PyQt6 import uic
from PySide6 import QtUiTools
from PySide6 import QtWidgets
from PySide6.QtWidgets import (QDockWidget, QGraphicsScene, QGraphicsPixmapItem,
                              QGraphicsView, QWidget, QSizePolicy, QSlider, QPushButton,
                              QAbstractItemView, QListView, QListWidget, QVBoxLayout,
                               QMenu)
from PySide6.QtGui import QAction, QPainter
from PySide6.QtCore import (QMetaObject, QFile, QRectF,
                            QCoreApplication, QSize, Qt,
                            QEvent, QObject)

from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)

import concurrent.futures





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


class Thumbnails(QWidget):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.objectName():
            self.setObjectName(u"thumbnails")
        self.setWindowModality(Qt.WindowModal)
        self.resize(759, 544)
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
        self.thumbsZoom.setMinimumSize(QSize(0, 15))
        self.thumbsZoom.setMinimum(5)
        self.thumbsZoom.setMaximum(512)
        self.thumbsZoom.setValue(150)
        self.thumbsZoom.setOrientation(Qt.Horizontal)

        self.verticalLayout_2.addWidget(self.thumbsZoom)

        self.dockWidget.setWidget(self.dockWidgetContents)

        self.verticalLayout.addWidget(self.dockWidget)


        QMetaObject.connectSlotsByName(self)

