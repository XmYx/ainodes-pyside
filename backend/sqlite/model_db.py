import os
import sqlite3

from backend.sqlite import model_db_civitai
from backend.singleton import singleton

gs = singleton


model_data_file = 'model_data.db'

def check_model_db_status():
    os.makedirs(gs.system.db_dir, exist_ok=True)
    db_file = os.path.join(gs.system.db_dir,model_data_file)
    model_db_con = sqlite3.connect(db_file)
    model_db_cur = model_db_con.cursor()
    res = model_db_cur.execute("SELECT name FROM sqlite_master WHERE name='model_data'")
    model_data_exists = res.fetchone()
    if model_data_exists is None:
        model_db_cur.execute("CREATE TABLE model_data(model_name PRIMARY KEY, creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,type TEXT, nsfw TEXT, trained_words TEXT, tags TEXT, version TEXT, model_data BLOB, base_model TEXT,files TEXT,images TEXT, description TEXT,allowNoCredit TEXT, allowCommercialUse TEXT, allowDerivatives TEXT, allowDifferentLicense TEXT )")
    res = model_db_cur.execute("SELECT name FROM sqlite_master WHERE name='model_data'")
    model_data_exists = res.fetchone()
    if len(model_data_exists) > 0 and gs.system.update_model_data_on_startup:
        model_db_civitai_api = model_db_civitai.civit_ai_api()
        model_db_civitai_api.civitai_start_model_update()

def check_model_sha_db_status():
    os.makedirs(gs.system.db_dir, exist_ok=True)
    db_file = os.path.join(gs.system.db_dir,model_data_file)
    model_db_con = sqlite3.connect(db_file)
    model_db_cur = model_db_con.cursor()
    res = model_db_cur.execute("SELECT name FROM sqlite_master WHERE name='model_sha_data'")
    model_data_exists = res.fetchone()
    if model_data_exists is None:
        model_db_cur.execute("CREATE TABLE model_sha_data(model_name PRIMARY KEY, creation_time DATETIME DEFAULT CURRENT_TIMESTAMP,config TEXT, version TEXT, sha TEXT)")
