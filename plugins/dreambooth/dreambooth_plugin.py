import os

from PySide6 import QtCore, QtUiTools
from PySide6.QtWidgets import QDockWidget, QFileDialog
from PySide6.QtCore import Slot, Signal, QObject, QFile
from backend.singleton import singleton

gs = singleton

from dreambooth import DreamBooth

class FineTune(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("./finetune.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class Callbacks(QObject):
    dreambooth_start_signal = Signal()
    dreambooth_stop_signal = Signal()


class aiNodesPlugin:
    def __init__(self, parent):
        self.parent = parent
        self.dreambooth = FineTune()
        self.signals = Callbacks()
        self.connections()
        self.load_folder_content()

    def initme(self):
        print("Dreambooth")
        self.dreambooth.w.show()


    def connections(self):
        self.dreambooth.w.pathInputImages.clicked.connect(self.set_path_to_input_image)
        self.dreambooth.w.logsFolder.clicked.connect(self.set_path_to_logfiles)
        self.dreambooth.w.resumeModel.clicked.connect(self.set_path_to_resume_model)
        self.dreambooth.w.initEmbeddingManager.clicked.connect(self.set_path_to_init_embedding_manager)
        self.dreambooth.w.Start.clicked.connect(self.start_dreambooth)
        self.dreambooth.w.Stop.clicked.connect(self.stop_dreambooth)

    def load_folder_content(self):
        self.dreambooth.w.base.clear()
        models = os.listdir(gs.system.dreambooth_config)
        self.parent.path_setup.w.activeModel.setText(gs.system.sdPath)
        for model in models:
            location = os.path.join(gs.system.dreambooth_config, model)
            self.dreambooth.w.base.addItem(model)

    @Slot()
    def set_path_to_input_image(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to input Images')
        print(filename)
        self.dreambooth.w.data_root.setText(filename)

    @Slot()
    def set_path_to_logfiles(self):
        filename = QFileDialog.getExistingDirectory(caption='Logfile Path')
        print(filename)
        self.dreambooth.w.logdir.setText(filename)

    @Slot()
    def set_path_to_resume_model(self):
        filename = QFileDialog.getOpenFileName(caption='Model to resume from', filter='Checkpoint (*.ckpt)')
        self.dreambooth.w.actual_resume.setText(filename[0])

    @Slot()
    def set_path_to_init_embedding_manager(self):
        filename = QFileDialog.getOpenFileName(caption='Initialize embedding manager from a checkpoint', filter='Checkpoint (*.ckpt)')
        self.dreambooth.w.init_embedding_manager_ckpt.setText(filename[0])


    def start_dreambooth(self):
        gs.ti_grad_flag_switch = True
        self.parent.plugin_thread(self.create_dreambooth)

    def stop_dreambooth(self):
        self.signals.dreambooth_stop_signal.emit()

    def setup_view(self):

        for dock in self.parent.w.findChildren(QDockWidget):
            dock.hide()

        self.dreambooth.w.dockWidget.setWindowTitle('Textual Inversion (dreambooth)')
        self.parent.w.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.dreambooth.w.dockWidget)


    def create_dreambooth(self, progress_callback=None):
        self.dreambooth_training.dreambooth()
        """
        self.parent.ti.create_txt_inv(
            name=self.dreambooth.w.name.text(),
            data_root=self.dreambooth.w.data_root.text(),
            #base=os.path.join(gs.system.dreambooth_config,self.dreambooth.w.base.currentText()),
            logdir=self.dreambooth.w.logdir.text(),
            placeholder_tokens=self.dreambooth.w.placeholder_tokens.text(),
            init_word=self.dreambooth.w.init_word.text(),
            actual_resume=self.dreambooth.w.actual_resume.text(),
            no_test=self.dreambooth.w.no_test.isChecked(),
            train=self.dreambooth.w.train.isChecked()
        )"""
