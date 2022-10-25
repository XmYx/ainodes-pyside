import os

from PySide6 import QtUiTools, QtCore, QtWidgets
from PySide6.QtCore import QObject, QFile, Signal

from backend.modelloader import load_upscaler
from backend.singleton import singleton
from backend.upscale import Upscale
from backend.img2ascii import to_ascii

gs = singleton


class ImageLab_ui(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui_widgets/img_lab.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class DropListView(QtWidgets.QListWidget):

    fileDropped = QtCore.Signal(list)

    def __init__(self, parent=None):
        super(DropListView, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QtCore.QSize(72, 72))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))
            self.fileDropped.emit(links)
        else:
            event.ignore()

class Callbacks(QObject):
    upscale_start = Signal()
    upscale_stop = Signal()
    upscale_counter = Signal(int)
    img_to_txt_start = Signal()


class ImageLab():  # for signaling, could be a QWidget  too

    def __init__(self):
        super().__init__()
        self.signals = Callbacks()
        self.imageLab = ImageLab_ui()
        self.dropWidget = DropListView()
        self.dropWidget.setAccessibleName('fileList')
        self.dropWidget.fileDropped.connect(self.pictureDropped)
        self.imageLab.w.dropZone.addWidget(self.dropWidget)
        self.imageLab.w.startUpscale.clicked.connect(self.signal_start_upscale)
        self.imageLab.w.startImgToTxt.clicked.connect(self.signal_start_img_to_txt)

    def signal_start_upscale(self):
        self.signals.upscale_start.emit()

    def signal_start_img_to_txt(self):
        self.signals.img_to_txt_start.emit()

    def show(self):
        self.imageLab.w.show()

    def pictureDropped(self, l):
        self.fileList = []
        self.dropWidget.clear()
        for path in l:
            if os.path.isdir(path):
                result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(path) for f in filenames if os.path.splitext(f)[1] == '.png']
                for file in result:
                    self.fileList.append(file)
                    self.dropWidget.insertItem(0, file)
            elif os.path.isfile(path):
                self.dropWidget.insertItem(0, path)
                self.fileList.append(path)
        self.imageLab.w.filesCount.display(str(len(self.fileList)))

    def upscale_count(self, num):
        self.signals.upscale_counter.emit(num)

    def run_upscale(self, progress_callback=None):
        self.upscale = Upscale()
        self.upscale.signals.upscale_counter.connect(self.upscale_count)
        model_name=''
        if self.imageLab.w.ESRGAN.isChecked():
            if self.imageLab.w.RealESRGAN.isChecked():
                model_name = 'RealESRGAN_x4plus'
            if self.imageLab.w.RealESRGANAnime.isChecked():
                model_name = 'RealESRGAN_x4plus_anime_6B'

            load_upscaler(self.imageLab.w.GFPGAN.isChecked(), self.imageLab.w.ESRGAN.isChecked(), model_name)
        if len(self.fileList) > 0:
            if self.imageLab.w.ESRGAN.isChecked() or self.imageLab.w.GFPGAN.isChecked():
                self.upscale.upscale_and_reconstruct(self.fileList,
                                             upscale          = self.imageLab.w.ESRGAN.isChecked(),
                                             upscale_scale    = self.imageLab.w.esrScale.value(),
                                             upscale_strength = self.imageLab.w.esrStrength.value()/100,
                                             use_gfpgan       = self.imageLab.w.GFPGAN.isChecked(),
                                             strength         = self.imageLab.w.gfpStrength.value()/100,
                                             image_callback   = None)

        self.signals.upscale_stop.emit()

    def run_img2txt(self, progress_callback=None):
        grayscale = 0
        if self.imageLab.w.grayscaleType.isChecked():
            grayscale = 1

        if len(self.fileList) > 0:
            for path in self.fileList:
                to_ascii(path, self.imageLab.w.img2txtRatio.value()/100, grayscale)
