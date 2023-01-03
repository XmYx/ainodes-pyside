from PySide6 import QtNetwork, QtCore
from PySide6.QtCore import QObject, Signal, QByteArray


class Callbacks(QObject):
    web_image_retrived = Signal(QByteArray)


class WebImages:

    def __init__(self):
        self.next_models_link = None
        self.signals = Callbacks()
        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleResponse)

    def get_image(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam.get(req)


    def handleResponse(self, reply):
        er = reply.error()
        if er == QtNetwork.QNetworkReply.NoError:
            bytes_string = reply.readAll()
            self.signals.web_image_retrived.emit(bytes_string)
        else:
            print('retrive image caused error: ', er)
