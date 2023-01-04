import json
import os
import sqlite3

from PySide6 import QtNetwork, QtCore
from PySide6.QtCore import QEventLoop, QObject, Signal
from backend.singleton import singleton

gs = singleton
model_data_file = 'model_data.db'
db_file = os.path.join(gs.system.db_dir, model_data_file)


def get_model_config_data(name):
    model_db_con = sqlite3.connect(db_file)
    model_db_con.row_factory = sqlite3.Row
    sqlite_select_with_param = """select * from model_sha_data where model_name = ?"""
    cursor = model_db_con.cursor()
    cursor.execute(sqlite_select_with_param, (name,))
    records = cursor.fetchall()
    cursor.close()
    model_db_con.close()
    return records


def insert_model_config_data(name, config, version, sha):
    model_db_con = sqlite3.connect(db_file)
    model_db_con.row_factory = sqlite3.Row
    sqlite_insert_with_param = """insert or replace into model_sha_data (model_name, config, version, sha) values(?,?,?,?);"""
    cursor = model_db_con.cursor()
    cursor.execute(sqlite_insert_with_param, (name,config,version,sha,))
    model_db_con.commit()
    cursor.close()
    model_db_con.close()
