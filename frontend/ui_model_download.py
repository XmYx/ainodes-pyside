import io
import json
import os
import re
import shutil
import urllib.request
import urllib.parse

from PySide6 import QtUiTools, QtNetwork, QtCore
from PySide6.QtCore import QObject, QFile, Signal
from backend.poor_mans_wget import wget_progress, wget_headers
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
        self.model_download.w.more_models.clicked.connect(self.get_more_models)
        self.actual_model_list = {}
        self.next_models_link = None

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

    def executeMoreRequest(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleMoreResponse)
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
            if 'metadata' in responseDict:
                if 'nextPage' in responseDict['metadata']:
                    self.next_models_link = responseDict['metadata']['nextPage']
                    self.model_download.w.more_models.setEnabled(True)
                else:
                    self.model_download.w.more_models.setEnabled(False)
                    self.next_models_link = None
            else:
                self.model_download.w.more_models.setEnabled(False)
                self.next_models_link = None
        else:
            print('error: ', er)

    def handleMoreResponse(self, reply):
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
            if 'metadata' in responseDict:
                if 'nextPage' in responseDict['metadata']:
                    self.next_models_link = responseDict['metadata']['nextPage']
                    self.model_download.w.more_models.setEnabled(True)
                else:
                    self.model_download.w.more_models.setEnabled(False)
                    self.next_models_link = None
            else:
                self.model_download.w.more_models.setEnabled(False)
                self.next_models_link = None
        else:
            print('error: ', er)

    def get_more_models(self):
        print('more', self.next_models_link)
        if self.next_models_link != None:
            self.executeMoreRequest(self.next_models_link)

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
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        tmp = ''.join(filter(whitelist.__contains__, name))
        return tmp.replace(' ', '_')

    def download_model(self, progress_callback=False):
        safetensors = False
        self.model_download.w.download_button.setEnabled(False)
        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]
        config_name = ''
        regex = re.compile(r'(.*?)\.')
        headers = wget_headers(model_info['model']['downloadUrl'])
        filename = headers['Content-Disposition'].replace('attachment; filename="','').replace('"','')
        safetensors = True if 'safetensors' in filename else False
        filename = regex.match(filename)[1]
        length = headers['Content-Length']
        model_name = 'noNameFound'
        if len(filename) < 1:
            model_Version_info = model_info['model']['name'].replace('learned embeds','')
            filename = self.sanitize(model_info['item']['name'] + f"_{model_Version_info}")
        if model_info['item']['type'] == 'Checkpoint':
            config_name = filename + '.yaml'


            if safetensors is False:
                model_name = filename + '.ckpt'
            else:
                model_name = filename + '.safetensors'

            model_outpath = os.path.join(gs.system.custom_models_dir, model_name)


        if model_info['item']['type'] == 'TextualInversion':
            model_name = filename + '.pt'
            model_outpath = os.path.join(gs.system.textual_inversion_dir, model_name)

        if model_info['item']['type'] == 'Hypernetwork':
            model_name = filename + '.pt'
            model_outpath = os.path.join(gs.system.hypernetwork_dir, model_name)

        if model_info['item']['type'] == 'AestheticGradient':
            model_name = filename + '.pt'
            model_outpath = os.path.join(gs.system.aesthetic_gradients_dir, model_name)

        print(f"download model {model_name} from url: {model_info['model']['downloadUrl']} ")

        chunk_size = 1024
        if int(length) > 102400:
            chunk_size = 8192

        try:
            wget_progress(url=model_info['model']['downloadUrl'], filename=model_outpath, length=length, chunk_size=chunk_size, callback=self.parent.model_download_progress_callback)
            self.parent.model_download_progress_callback(100)
        except Exception as e:
            print('Download failed: ', e)
            self.parent.model_download_progress_callback(0)

        self.model_download.w.download_button.setEnabled(True)
        #self.do_download(model_info['model']['downloadUrl'],model_outpath)
        if config_name != '':
            src = os.path.join(gs.system.default_config_yaml_dir, self.model_download.w.config_yaml.currentText())
            dst = os.path.join(gs.system.custom_models_dir, config_name)
            shutil.copyfile(src, dst)

        self.parent.widgets[self.parent.current_widget].update_model_list()
