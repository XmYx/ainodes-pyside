import json
import shutil
from io import StringIO

from PySide6 import QtUiTools, QtNetwork
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtWidgets import (QDockWidget, QGraphicsScene, QGraphicsPixmapItem,
                               QGraphicsView, QWidget, QSizePolicy, QSlider, QPushButton,
                               QAbstractItemView, QListView, QListWidget, QVBoxLayout,
                               QMenu, QListWidgetItem)
from PySide6.QtGui import QAction, QPainter, QPainterPath, QPolygon, QPen
from PySide6.QtCore import (QMetaObject, QFile, QRectF,
                            QCoreApplication, QSize, Qt,
                            QEvent, QObject, QTimeLine, QPoint, QLine, Signal, QRect, QByteArray)

from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
                           QFont, QFontDatabase, QGradient, QIcon,
                           QImage, QKeySequence, QLinearGradient, QPainter,
                           QPalette, QPixmap, QRadialGradient, QTransform)

import requests



class LexicArt(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/lexicart.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
        self.setup()

    def setup(self):
        self.view = "icon"
        self.w.search.clicked.connect(self.doRequest)
        self.w.toggleview.clicked.connect(self.toggleView)
        self.w.zoom.valueChanged.connect(self.setZoom)

    def doRequest(self):
        query = self.w.query.text()
        url = "https://lexica.art/api/v1/search?q=" + query
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))

        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleResponse)
        self.nam.get(req)


    def handleResponse(self, reply):
        self.nam2 = QtNetwork.QNetworkAccessManager()
        self.nam2.finished.connect(self.handleImages)
        self.w.results.clear()
        er = reply.error()
        self.prompts = []
        self.counter = 0
        imageList = []
        if er == QtNetwork.QNetworkReply.NoError:
            bytes_string = reply.readAll()
            responseDict = json.loads(str(bytes_string, 'utf-8'))
            for i in responseDict["images"]:
                self.prompts.append(i['prompt'])
            for i in responseDict["images"]:
                #print(i['srcSmall'])
                req = QtNetwork.QNetworkRequest(QtCore.QUrl(i['srcSmall']))
                self.nam2.get(req)

                #self.w.results.addItem(QIcon(QImage(i['srcSmall'])), i['prompt'])

            #print(responseDict["images"])
            
            
            #print(str(bytes_string, 'utf-8'))
        else:
            print("Error occured: ", er)
            print(reply.errorString())


    def request_images(self,imagelist):


        for url in imagelist:
            r = requests.get(url)
            if r.status_code == 200:
                print('requested url:', url)
                r.raw.decode_content = True
                bytes_string = None
                bytes_string = r.content
                img = QImage()
                img.loadFromData(bytes_string)
                pixmap = QPixmap.fromImage(img)
                self.w.results.addItem(QListWidgetItem(QIcon(pixmap), self.prompts[self.counter]))


    def handleImages(self, images):
        #print(f"These will be the images: {images.error()}")
        er = images.error()
        if er == QtNetwork.QNetworkReply.NoError:
            bytes_string = images.readAll()
            img = QImage()
            img.loadFromData(bytes_string)
            pixmap = QPixmap.fromImage(img)
            self.w.results.addItem(QListWidgetItem(QIcon(pixmap), self.prompts[self.counter]))
            self.counter += 1
    def toggleView(self):
        self.w.update()
        self.w.zoom.setMinimum(25)
        self.w.zoom.setMaximum(1000)
        if self.view == "list":
            self.w.results.setViewMode(QListView.IconMode)
            print("icon")
            self.view = "icon"
        elif self.view == "icon":
            self.w.results.setViewMode(QListView.ListMode)
            self.view = "list"
            print("list")
        self.w.update()
    def setZoom(self):
        size = self.w.zoom.value()
        if self.view == "icon":
            self.w.results.setGridSize(QSize(size + 20, size + 200))
            self.w.results.setIconSize(QSize(size, size))
        elif self.view == "list":
            self.w.results.setGridSize(QSize(size, size))
            self.w.results.setIconSize(QSize(size, size))

class SimplePrompt(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/simple_prompt.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class SimplePromptDisplay(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/simple_prompt_display.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SimpleKreaPrompts(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/simple_krea_prompts.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class Hypernetwork(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/hypernetwork.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class ThumbsUI(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/thumbnails_new.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class FineTune(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/finetune.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SetTxtToVid(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/set_txt2vid.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class InputPreview(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/input_preview.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SetImgToImg(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/set_img2img.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SetTxtToImg(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/set_txt2img.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class Compass(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/compass.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class OutpaintControls(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/outpaint_controls.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class AnimKeyEditor(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/keyframe_editor.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class FetchPrompts(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/prompt_fetcher.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class AnimKeys(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/animKeys.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class AnimDials(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/animSliders.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class AnimSliders(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/animSliders.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SystemSetup(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/system_config.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class SizerCount(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/sizer_count.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class Dynaimage(QObject):
    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/dynaimage.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()
class Dynaview(QObject):
    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/dynaview.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class Sampler(QObject):
    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/sampler.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()

class Prompt(QObject):
    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/prompt.ui")
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
        self._zoom = 0
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
        self.setAccessibleName(u'thumbnails')
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

        QMetaObject.connectSlotsByName(self)
