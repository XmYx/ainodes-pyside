import io
import json
import os
import shutil
import urllib.request
import urllib.parse

from PySide6 import QtUiTools, QtNetwork, QtCore
from PySide6.QtCore import QObject, QFile, Signal
from backend.poor_mans_wget import wget_progress
from backend.singleton import singleton
gs = singleton

class ModelDownload_UI(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("frontend/ui/model_download.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class Callbacks(QObject):
    startDownload = Signal()
    # Signal for the window to establish the maximum value
    # of the progress bar.
    setTotalProgress = Signal(int)
    # Signal to increase the progress.
    setCurrentProgress = Signal(int)
    # Signal to be emitted when the file has been downloaded successfully.
    succeeded = Signal()



class ModelDownload():
    def __init__(self, parent):
        self.parent = parent
        self.model_download = ModelDownload_UI()
        self.signals = Callbacks()
        self.model_download.w.model_search.clicked.connect(self.models_search)
        self.model_download.w.model_list.itemClicked.connect(self.show_model_infos)
        self.model_download.w.download_button.clicked.connect(self.signal_download_model)
        self.actual_model_list = {}

    def signal_download_model(self):
        self.signals.startDownload.emit()

    def show_model_infos(self,**args):

        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]
        info = f"""Model Name: {model_info['item']['name']}
Version: {model_info['model']['name']}
Tags: {model_info['item']['tags']}            
Trained Words: {model_info['model']['trainedWords']}  
Type: {model_info['item']['type']} 
NSFW: {model_info['item']['nsfw']} 
"""
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
        self.executeRequest(url)

    def sanitize(self, name):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        tmp = ''.join(filter(whitelist.__contains__, name))
        return tmp.replace(' ', '_')

    def download_model(self, progress_callback=False):
        self.model_download.w.download_button.setEnabled(False)
        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]
        config_name = ''

        if model_info['item']['type'] == 'Checkpoint':
            model_name = self.sanitize(model_info['item']['name']) + f"_{model_info['model']['name']}"
            config_name = model_name + '.yaml'
            model_name += '.ckpt'
            model_outpath = os.path.join(gs.system.customModels, model_name)
        if model_info['item']['type'] == 'TextualInversion':
            model_name = self.sanitize(model_info['item']['name']) + f"_{model_info['model']['name']}" + '.pt'
            model_outpath = os.path.join(gs.system.embeddings_dir, model_name)
        if model_info['item']['type'] == 'Hypernetwork':
            model_name = self.sanitize(model_info['item']['name']) + f"_{model_info['model']['name']}" + '.pt'
            model_outpath = os.path.join(gs.system.hypernetwork_dir, model_name)
        if model_info['item']['type'] == 'AestheticGradient':
            model_name = self.sanitize(model_info['item']['name']) + f"_{model_info['model']['name']}" + '.pt'
            model_outpath = os.path.join(gs.system.aesthetic_gradients, model_name)

        print(f"download model from url: {model_info['model']['downloadUrl']} ")
        wget_progress(model_info['model']['downloadUrl'],model_outpath, 8192, self.parent.model_download_progress_callback)
        self.model_download.w.download_button.setEnabled(True)
        #self.do_download(model_info['model']['downloadUrl'],model_outpath)
        if config_name != '':
            src = os.path.join(gs.system.default_config_yaml_path, self.model_download.w.config_yaml.currentText())
            dst = os.path.join(gs.system.customModels, config_name)
            shutil.copyfile(src, dst)
