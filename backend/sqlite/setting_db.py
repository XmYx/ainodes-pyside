import json
import os
import sqlite3

from backend.sqlite import model_db_civitai
from backend.singleton import singleton

gs = singleton


data_file = 'setting_data.db'
db_file = os.path.join('data/db',data_file)

def check_model_db_status():
    os.makedirs('data/db', exist_ok=True)
    db_con = sqlite3.connect(db_file)
    cursor = db_con.cursor()
    res = cursor.execute("SELECT name FROM sqlite_master WHERE name='setting_data'")
    model_data_exists = res.fetchone()
    if model_data_exists is None:
        cursor.execute("CREATE TABLE setting_data(creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,project TEXT, settings TEXT)")
    res = cursor.execute("SELECT name FROM sqlite_master WHERE name='setting_data'")
    setting_data_exists = res.fetchone()
    if len(setting_data_exists) > 0:
        gs.db_settings_present = True
    cursor.close()
    db_con.close()

def get_last_settings():
    db_con = sqlite3.connect(db_file)
    cursor = db_con.cursor()
    settings_select = """SELECT settings
                        FROM setting_data
                        WHERE project = 'default'
                        AND creation_time=(
                        SELECT MAX(creation_time) FROM setting_data WHERE  project = 'default');"""
    res = cursor.execute(settings_select)
    settings = res.fetchone()
    cursor.close()
    db_con.close()
    if settings is None:
        settings = []
    return settings

def save_settings():
    print('save settings to db')
    db_con = sqlite3.connect(db_file)
    cursor = db_con.cursor()
    data_tuple = (json.dumps({
        "system": json.dumps(gs.system.__dict__),
        "diffusion": json.dumps(gs.diffusion.__dict__)
    }, indent=4),
    'default')
    sqlite_insert_with_param = (
        """insert or replace into setting_data (settings, project) values (?,?);""")
    cursor.execute(sqlite_insert_with_param, data_tuple)

    cursor.close()
    db_con.commit()
    db_con.close()
