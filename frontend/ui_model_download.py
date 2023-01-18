import io
import json
import os
import re
import shutil
import urllib.request
import urllib.parse

from PySide6 import QtUiTools, QtNetwork, QtCore
from PySide6.QtCore import QObject, QFile, Signal, Slot, Qt
from backend.poor_mans_wget import wget_progress, wget_headers
from backend.sqlite import model_db_civitai
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
    # signal to push the list of preview images to be shownon canvas
    show_model_preview_images = Signal(dict)
    set_download_percent = Signal(int)
    add_model_search_item = Signal(str)
    set_download_info_text = Signal(str)

class ModelDownload():
    def __init__(self, parent):
        self.parent = parent
        self.model_download = ModelDownload_UI()
        self.signals = Callbacks()
        self.model_download.w.model_search.clicked.connect(self.models_search)
        self.model_download.w.more_results.clicked.connect(self.more_results)
        self.model_download.w.model_list.itemClicked.connect(self.show_model_infos)
        self.model_download.w.download_button.clicked.connect(self.signal_download_model)
        self.model_download.w.maintain_custom_models.clicked.connect(self.maintain_custom_models)
        self.actual_model_list = {}
        self.next_models_link = None
        self.civit_ai_api = model_db_civitai.civit_ai_api()

    @Slot()
    def model_download_progress_callback_signal(self, percent):
        self.model_download.w.dl_progress.setValue(percent)

    @Slot()
    def download_model_thread(self):
        self.parent.run_as_thread(self.download_model)

    @Slot()
    def model_download_progress_callback(self, percent):
        self.signals.set_download_percent.emit(percent)

    def maintain_custom_models(self):
        self.civit_ai_api.civitai_start_model_update()
        #self.civit_ai_api.signals.civitai_start_model_update.emit()

    def signal_download_model(self):
        self.signals.startDownload.emit()

    def show_model_infos(self,**args):

        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]
        info = f"""Model Name: {model_info['item']['name']}
Base Model: {model_info['model']['baseModel']}
Version: {model_info['model']['name']}
Tags: {model_info['item']['tags']}            
Trained Words: {model_info['model']['trainedWords']}  
Type: {model_info['item']['type']} 
NSFW: {model_info['item']['nsfw']} 
"""
        self.model_download.w.model_informations.setPlainText(info)
        if self.model_download.w.preview_on_canvas.isChecked():
            self.signals.show_model_preview_images.emit(model_info['model'])
        if model_info['model']['baseModel'] == 'SD 1.5':
            if 'inpainting' in model_info['model']['name']:
                self.model_download.w.config_yaml.setCurrentIndex(4)
            else:
                self.model_download.w.config_yaml.setCurrentIndex(1)
        if model_info['model']['baseModel'] == 'SD 2.0':
            if 'inpainting' in model_info['model']['name']:
                self.model_download.w.config_yaml.setCurrentIndex(5)
            else:
                self.model_download.w.config_yaml.setCurrentIndex(2)
        if model_info['model']['baseModel'] == 'SD 2.0 768':
            self.model_download.w.config_yaml.setCurrentIndex(3)
        if model_info['model']['baseModel'] == 'SD 2.1':
            if 'inpainting' in model_info['model']['name']:
                self.model_download.w.config_yaml.setCurrentIndex(5)
            else:
                self.model_download.w.config_yaml.setCurrentIndex(2)
        if model_info['model']['baseModel'] == 'SD 2.1 768':
            self.model_download.w.config_yaml.setCurrentIndex(3)




    def executeRequest(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleResponse)
        self.nam.get(req)


    def add_model_search_item(self, model_description):
        self.model_download.w.model_list.addItem(model_description)
        self.model_download.w.model_list.sortItems(Qt.AscendingOrder)

    def executeMoreRequest(self):
        # Create a QNetworkRequest with the specified url
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(self.next_models_link))
        # Create a QNetworkAccessManager
        self.nam = QtNetwork.QNetworkAccessManager()
        # Connect the finished signal of the QNetworkAccessManager to the handleMoreResponse function
        self.nam.finished.connect(self.handleMoreResponse)
        # Send a GET request using the QNetworkAccessManager
        self.nam.get(req)



    def handleResponse(self, reply):
        # Initialize an empty dictionary to store model information
        self.actual_model_list = {}

        # Clear the model list widget
        self.model_download.w.model_list.clear()

        # Check for errors in the reply
        er = reply.error()

        # If there are no errors:
        if er == QtNetwork.QNetworkReply.NoError:
            # Read the bytes of the reply
            bytes_string = reply.readAll()
            if bytes_string != '':
                # Convert the bytes to a dictionary
                responseDict = json.loads(str(bytes_string, 'utf-8'))

                # Iterate through the items in the response
                for item in responseDict['items']:
                    # Create a temporary dictionary with the item information
                    tmp_item = {
                        'id': item['id'],
                        'name': item['name'],
                        'type': item['type'],
                        'nsfw': item['nsfw'],
                        'tags': item['tags']
                    }

                    # Iterate through the model versions of the item
                    for model in item['modelVersions']:

                        for file in model['files']:

                            if file['type'] != 'Training Data' and file['type'] != 'Config':

                                # Construct a description of the model
                                model_description = f"{item['name']}  Version: {model['name']} Type: {file['type']} Format: {file['format']}"

                                # Add the model and item information to the actual_model_list dictionary
                                self.actual_model_list[model_description] = {'model':model, 'item':tmp_item, 'file':file}

                                # Add the model description to the model list widget
                                self.signals.add_model_search_item.emit(model_description)

                # Check if there is a "next page" of models
                if 'metadata' in responseDict:
                    if 'nextPage' in responseDict['metadata']:
                        # If there is a next page, enable the "more models" button and store the link to the next page
                        self.model_download.w.more_results.setEnabled(True)
                        self.next_models_link = responseDict['metadata']['nextPage']
                    else:
                        # If there is no next page, disable the "more models" button and set the link to None
                        self.model_download.w.more_results.setEnabled(False)
                        self.next_models_link = None
                else:
                    # If there is no "metadata" key in the response, disable the "more models" button and set the link to None
                    self.model_download.w.more_results.setEnabled(False)
                    self.next_models_link = None
                info_string = f"showing {self.model_download.w.model_list.count()} results"
                info_string = info_string + ', there is more to fetch' if self.next_models_link is not None else info_string
                self.signals.set_download_info_text.emit(info_string)
        else:
            # If there are errors, print the error code
            print('error: ', er)

    def set_download_info_text(self, info_string):
        self.model_download.w.model_download_info.setText(info_string)

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

                    for file in model['files']:
                        if file['type'] != 'Training Data' and file['type'] != 'Config':

                            # Construct a description of the model
                            model_description = f"{item['name']}  Version: {model['name']} Type: {file['type']} Format: {file['format']}"

                            # Add the model and item information to the actual_model_list dictionary
                            self.actual_model_list[model_description] = {'model':model, 'item':tmp_item, 'file':file}

                            # Add the model description to the model list widget
                            self.signals.add_model_search_item.emit(model_description)

            if 'metadata' in responseDict:
                if 'nextPage' in responseDict['metadata']:
                    self.model_download.w.more_results.setEnabled(True)
                    self.next_models_link = responseDict['metadata']['nextPage']
                else:
                    self.model_download.w.more_results.setEnabled(False)
                    self.next_models_link = None
            else:
                self.model_download.w.more_results.setEnabled(False)
                self.next_models_link = None

            info_string = f"showing {self.model_download.w.model_list.count()} results"
            info_string = info_string + ', there is more to fetch' if self.next_models_link is not None else info_string
            self.signals.set_download_info_text.emit(info_string)
        else:
            print('error: ', er)


    def more_results(self):
        self.executeMoreRequest()

    def models_search_(self):
        self.parent.run_as_thread(self.models_search_thread)

    def models_search(self, progress_callback=None):
        # Build the query string from the user input fields
        query = 'query=' + self.model_download.w.query.text()
        query += '&tag=' + self.model_download.w.tag.text() if self.model_download.w.tag.text() != '' else ''
        query += '&limit=' + str(self.model_download.w.limit.value())
        query += '&types=' + self.model_download.w.types.currentText()
        query += '&sort=' + self.model_download.w.sort.currentText()
        query += '&period=' + self.model_download.w.period.currentText()

        # Construct the URL using the base URL and the query string
        url = "https://civitai.com/api/v1/models?" + query

        # Send a GET request to the URL
        self.executeRequest(url)


    def sanitize(self, name):
        whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ-')
        tmp = ''.join(filter(whitelist.__contains__, name))
        return tmp.replace(' ', '_')

    def download_model(self, progress_callback=False):

        if self.model_download.w.config_yaml.currentIndex() == 0:
            print('no yamle selected, please select config yaml before download')
            return

        safetensors = False
        self.model_download.w.download_button.setEnabled(False)
        model_info = self.actual_model_list[self.model_download.w.model_list.currentItem().text()]
        config_name = ''
        regex = re.compile(r'(.*?)\.')
        headers = wget_headers(model_info['file']['downloadUrl'])
        filename = headers['Content-Disposition'].replace('attachment; filename="','').replace('"','')
        safetensors = True if 'safetensors' in filename else False
        filename = regex.match(filename)[1]
        length = headers['Content-Length']
        model_name = 'noNameFound'
        if len(filename) < 1:
            model_Version_info = model_info['file']['name'].replace('learned embeds','')
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
            wget_progress(url=model_info['model']['downloadUrl'], filename=model_outpath, length=length, chunk_size=chunk_size, callback=self.model_download_progress_callback)
            self.model_download_progress_callback(100)
        except Exception as e:
            print('Download failed: ', e)
            self.model_download_progress_callback(0)

        self.model_download.w.download_button.setEnabled(True)
        #self.do_download(model_info['model']['downloadUrl'],model_outpath)
        if config_name != '':
            src = os.path.join(gs.system.default_config_yaml_dir, self.model_download.w.config_yaml.currentText())
            dst = os.path.join(gs.system.custom_models_dir, config_name)
            shutil.copyfile(src, dst)

        self.parent.widgets[self.parent.current_widget].update_model_list()
