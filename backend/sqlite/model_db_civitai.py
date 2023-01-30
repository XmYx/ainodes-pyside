import json
import os
import sqlite3

from PySide6 import QtNetwork, QtCore
from PySide6.QtCore import QEventLoop, QObject, Signal
from backend.singleton import singleton

gs = singleton
model_data_file = 'model_data.db'


class Callbacks(QObject):
    civitai_no_more_models = Signal()
    civitai_start_model_update = Signal()


class civit_ai_api:

    def __init__(self):
        self.next_models_link = None
        self.signals = Callbacks()
        self.nam = QtNetwork.QNetworkAccessManager()
        self.nam.finished.connect(self.handleResponse)
        self.db_file = os.path.join(gs.system.db_dir, model_data_file)

    def insert_model(self, cursor, model, item):
        sqlite_insert_with_param = (
            """insert or replace into model_data (model_name, type, nsfw, trained_words, tags, version, base_model, files, images, model_data, description) values (?,?,?,?,?,?,?,?,?,?,?);""")
        if item['nsfw'] is False:
            item['nsfw'] = 'False'
        else:
            item['nsfw'] = 'True'
        data_tuple = (model['files'][0]['name'],
                      item['type'],
                      item['nsfw'],
                      ', '.join(model['trainedWords']),
                      ', '.join(item['tags']),
                      model['name'],
                      model['baseModel'],
                      json.dumps(model['files']),
                      json.dumps(model['images']),
                      json.dumps(model),
                      item['description']
                      )
        cursor.execute(sqlite_insert_with_param, data_tuple)


    def handleResponse(self, reply):
        er = reply.error()
        if er == QtNetwork.QNetworkReply.NoError:
            model_db_con = sqlite3.connect(self.db_file)
            cursor = model_db_con.cursor()
            try:
                bytes_string = reply.readAll()
                responseDict = json.loads(str(bytes_string, 'utf-8'))
                for item in responseDict['items']:
                    tmp_item = {
                        'id': item['id'],
                        'name': item['name'],
                        'type': item['type'],
                        'nsfw': item['nsfw'],
                        'tags': item['tags'],
                        'description': item['description'],
                        'poi': item['poi'],
                        'creator': item['creator']
                    }
                    for model in item['modelVersions']:
                        self.insert_model(cursor, model, tmp_item)
                if 'metadata' in responseDict:
                    if 'nextPage' in responseDict['metadata']:
                        self.next_models_link = responseDict['metadata']['nextPage']
                    else:
                        self.next_models_link = None
                else:
                    self.next_models_link = None
            except:
                pass
            finally:
                model_db_con.commit()
                cursor.close()
                model_db_con.close()
        else:
            print('Update the model Data from civitai failed, thats no big deal, next time you run the UI we try again: ', er)
            print(reply.readAll())
        if self.next_models_link is not None:
            req = QtNetwork.QNetworkRequest(QtCore.QUrl(self.next_models_link))
            self.nam.get(req)
        else:
            print('model list update done')
            self.signals.civitai_no_more_models.emit()

    def executeRequest(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam.get(req)

    def executeIMgRequest(self, url):
        req = QtNetwork.QNetworkRequest(QtCore.QUrl(url))
        self.nam.get(req)

    def civitai_start_model_update(self, progress_callback=False):
        print('start model data update')
        url = "https://civitai.com/api/v1/models?limit=100"
        self.executeRequest(url)

    def all_civitai_model_data_loaded(self, progress_callback=False):
        print('done')

    def civitai_get_model_data(self, name):
        print('get model data for:', name)
        model_db_con = sqlite3.connect(self.db_file)
        model_db_con.row_factory = sqlite3.Row
        sqlite_select_with_param = """select * from model_data where model_name = ?"""
        cursor = model_db_con.cursor()
        cursor.execute(sqlite_select_with_param, (name,))
        records = cursor.fetchall()
        cursor.close()
        model_db_con.close()
        return records
