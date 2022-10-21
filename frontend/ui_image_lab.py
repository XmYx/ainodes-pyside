import os

from PySide6 import QtUiTools, QtCore, QtWidgets, QtGui
from PySide6.QtCore import QObject, QFile


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

class ImageLab:

    def __init__(self):
        self.imageLab = ImageLab_ui()
        self.dropWidget = DropListView()
        self.dropWidget.setAccessibleName('fileList')
        self.dropWidget.fileDropped.connect(self.pictureDropped)
        self.imageLab.w.dropZone.addWidget(self.dropWidget)

        pass

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
