import json

from PySide6 import QtUiTools, QtNetwork, QtCore
from PySide6.QtCore import QObject, QFile


class ModelDownload_UI(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/model_download.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class ModelDownload():
    def __init__(self, parent):
        self.parent = parent
        self.model_download = ModelDownload_UI()
        self.model_download.w.model_search.clicked.connect(self.models_search)
        self.model_download.w.model_list.itemClicked.connect(self.show_model_infos)
        self.actual_model_list = {}


    def show_model_infos(self,**args):
        print(self.model_download.w.model_list.currentItem().text())
        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]


        info = f"""Model Name: {model_info['item']['name']}
Version: {model_info['model']['name']}
Tags: {model_info['item']['tags']}            
Trained Words: {model_info['model']['trainedWords']}  
Type: {model_info['item']['type']} 
NSFW: {model_info['item']['nsfw']} 
"""
        print(info)
        self.model_download.w.model_informations.setPlainText(info)

    def executeRequest(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleResponse)
        self.nam.get(req)

    def handleResponse(self, reply):
        self.actual_model_list = {}
        self.model_download.w.model_list.clear()
        er = reply.error()
        #self.prompts = []
        #self.counter = 0
        if er == QtNetwork.QNetworkReply.NoError:
            bytes_string = reply.readAll()
            responseDict = json.loads(str(bytes_string, 'utf-8'))
            for item in responseDict['items']:
                tmp_item = {
                    'id': item['id'],
                    'name': item['name'],
                    'type': item['type'],
                    'nsfw': item['nsfw'],
                    'tags': item['tags']
                }
                for model in item['modelVersions']:

                    model_description = item['name'] + ' ' + ' Version: ' + model['name']
                    self.actual_model_list[model_description] = {'model':model, 'item':tmp_item}
                    self.model_download.w.model_list.addItem(model_description)
                    #print(model)

        else:
            print('error: ', er)



    def models_search(self):
        query = 'query=' + self.model_download.w.query.text()
        query += '&tag=' + self.model_download.w.tag.text() if self.model_download.w.tag.text() != '' else ''
        query += '&limit=' + str(self.model_download.w.limit.value())
        query += '&types=' + self.model_download.w.types.currentText()
        query += '&sort=' + self.model_download.w.sort.currentText()
        query += '&period=' + self.model_download.w.period.currentText()
        url = "https://civitai.com/api/v1/models?" + query
        print(url)
        self.executeRequest(url)
        print('hello')
        pass
