from backend.sqlite import model_db


def check_db_status():
    model_db.check_model_db_status()
