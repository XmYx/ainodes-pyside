#from PyQt6 import uic
from PySide6 import QtCore
from PySide6 import QtUiTools
from PySide6 import QtWidgets
from PySide6.QtWidgets import QDockWidget, QGraphicsScene, QGraphicsPixmapItem, QGraphicsView, QWidget


class SizerCount(QDockWidget):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/ui_widgets/sizer_count.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sizer_count.ui", self)
class Dynaimage(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/dynaimage.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
        file.close()
class Dynaview(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/dynaview.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
        file.close()
class Sampler(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/sampler.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/sampler.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
        file.close()

class Runner(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/runner.ui", self)

class Prompt(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/prompt.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/prompt.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
        file.close()
class Anim(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/anim.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/anim.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
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
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
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
            self._photo.setPixmap(QtGui.QPixmap())
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



class Preview(QWidget):
    loader = QtUiTools.QUiLoader()
    file = QtCore.QFile("frontend/ui_widgets/preview.ui")
    file.open(QtCore.QFile.ReadOnly)
    w = loader.load(file)
    file.close()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


class Thumbnails(QDockWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #uic.loadUi("frontend/ui_widgets/thumbnails.ui", self)
        loader = QtUiTools.QUiLoader()
        file = QtCore.QFile("frontend/ui_widgets/thumbnails.ui")
        file.open(QtCore.QFile.ReadOnly)
        self.w = loader.load(file, self)
        file.close()

