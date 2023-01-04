from backend.sqlite import model_db
from backend.sqlite import setting_db


def check_settings_db_status():
    setting_db.check_model_db_status()

def check_db_status():
    model_db.check_model_db_status()
