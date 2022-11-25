import json

from PySide6 import QtUiTools, QtNetwork, QtCore
from PySide6.QtCore import QObject, QFile, QSize
from PySide6.QtGui import QImage, QPixmap, QIcon
from PySide6.QtWidgets import QListWidgetItem, QListView


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
