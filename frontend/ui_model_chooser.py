import os

from PySide6.QtCore import QObject

from backend.singleton import singleton
from backend import load_models


gs = singleton

class ModelChooser_UI(QObject):
    def __init__(self, parent):
        self.parent = parent
        self.parent.path_setup.w.activateModel.clicked.connect(self.set_model)
        self.parent.path_setup.w.reloadModelList.clicked.connect(self.load_folder_content)
        self.load_folder_content()
        gs.models['custom_model_name'] = ''



    def load_folder_content(self):
        self.parent.path_setup.w.modelList.clear()
        models = os.listdir(gs.system.custom_models_dir)
        self.parent.path_setup.w.activeModel.setText(gs.system.sd_model_file)
        for model in models:
            location = os.path.join(gs.system.custom_models_dir, model)
            self.parent.path_setup.w.modelList.addItem(model)

    def set_model(self):
        if 'custom_model_name' not in gs.models:
            gs.models['custom_model_name'] = ''
        load_models.load_custom_model(os.path.join(gs.system.custom_models_dir, self.parent.path_setup.w.modelList.currentText()))
