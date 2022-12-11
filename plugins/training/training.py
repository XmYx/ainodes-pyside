import os
from types import SimpleNamespace

from PySide6 import QtCore, QtUiTools
from PySide6.QtWidgets import QDockWidget, QFileDialog
from PySide6.QtCore import Slot, Signal, QObject, QFile, QEasingCurve
from backend.singleton import singleton

gs = singleton

from plugins.training.dreambooth_class import DreamBooth
from plugins.training.sd_to_diffusers import run_translation
from plugins.training.train_lora_dreambooth import run_lora_dreambooth


class FineTune(QObject):

    def __init__(self, *args, **kwargs):
        loader = QtUiTools.QUiLoader()
        file = QFile("plugins/training/training.ui")
        file.open(QFile.ReadOnly)
        self.w = loader.load(file)
        file.close()


class Callbacks(QObject):
    dreambooth_start_signal = Signal()
    dreambooth_stop_signal = Signal()
    sd_to_diffusers_start_signal = Signal()


class aiNodesPlugin:
    def __init__(self, parent):
        self.parent = parent
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        self.training = FineTune()


    def initme(self):
        print("training")
        gs.system.dreambooth_config = "plugins/training/configs/dreambooth"
        self.signals = Callbacks()
        self.connections()
        self.init_anims()
        self.load_folder_content()
        self.dreambooth_training = DreamBooth()
        self.showAll = False
        self.show_hide_all_anim()
        self.training.w.show()



    def hideProcessCaption_anim(self):
        if self.pocHidden is True:
            self.showPocAnim.start()
        else:
            self.hidePocAnim.start()
        self.pocHidden = not self.pocHidden

    def hideFocalPointCrop_anim(self):
        if self.fpcHidden is True:
            self.showFpcAnim.start()
        else:
            self.hideFpcAnim.start()
        self.fpcHidden = not self.fpcHidden

    def hideSampler_anim(self):
        if self.aucHidden is True:
            self.showAucAnim.start()
        else:
            self.hideAucAnim.start()
        self.aucHidden = not self.aucHidden

    def hideAutoCrop_anim(self):
        if self.aucHidden is True:
            self.showAucAnim.start()
        else:
            self.hideAucAnim.start()
        self.aucHidden = not self.aucHidden

    def hideDreambooth_anim(self):
        if self.drbHidden is True:
            self.showDrbAnim.start()
        else:
            self.hideDrbAnim.start()
        self.drbHidden = not self.drbHidden

    def hideLoraDreambooth_anim(self):
        if self.ldbHidden is True:
            self.showLdbAnim.start()
        else:
            self.hideLdbAnim.start()
        self.ldbHidden = not self.ldbHidden

    def hideHypernetwork_anim(self):
        if self.hpnHidden is True:
            self.showHpnAnim.start()
        else:
            self.hideHpnAnim.start()
        self.hpnHidden = not self.hpnHidden

    def hidePrepareInput_anim(self):
        if self.pitHidden is True:
            self.showPitAnim.start()
        else:
            self.hidePitAnim.start()
        self.pitHidden = not self.pitHidden

    def hideCkpt2diff_anim(self):
        if self.cpdHidden is True:
            self.showCpdAnim.start()
        else:
            self.hideCpdAnim.start()
        self.cpdHidden = not self.cpdHidden

    def init_anims(self):
        self.showPocAnim = QtCore.QPropertyAnimation(self.training.w.processCaption, b"maximumHeight")
        self.showPocAnim.setDuration(1500)
        self.showPocAnim.setStartValue(self.training.w.processCaption.height())
        self.showPocAnim.setEndValue(self.training.w.height())
        self.showPocAnim.setEasingCurve(QEasingCurve.Linear)

        self.hidePocAnim = QtCore.QPropertyAnimation(self.training.w.processCaption, b"maximumHeight")
        self.hidePocAnim.setDuration(500)
        self.hidePocAnim.setStartValue(self.training.w.processCaption.height())
        self.hidePocAnim.setEndValue(0)
        self.hidePocAnim.setEasingCurve(QEasingCurve.Linear)

        self.showFpcAnim = QtCore.QPropertyAnimation(self.training.w.focalPointCrop, b"maximumHeight")
        self.showFpcAnim.setDuration(1500)
        self.showFpcAnim.setStartValue(self.training.w.focalPointCrop.height())
        self.showFpcAnim.setEndValue(self.training.w.height())
        self.showFpcAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideFpcAnim = QtCore.QPropertyAnimation(self.training.w.focalPointCrop, b"maximumHeight")
        self.hideFpcAnim.setDuration(500)
        self.hideFpcAnim.setStartValue(self.training.w.focalPointCrop.height())
        self.hideFpcAnim.setEndValue(0)
        self.hideFpcAnim.setEasingCurve(QEasingCurve.Linear)

        self.showAucAnim = QtCore.QPropertyAnimation(self.training.w.autoCrop, b"maximumHeight")
        self.showAucAnim.setDuration(1500)
        self.showAucAnim.setStartValue(self.training.w.autoCrop.height())
        self.showAucAnim.setEndValue(self.training.w.height())
        self.showAucAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideAucAnim = QtCore.QPropertyAnimation(self.training.w.autoCrop, b"maximumHeight")
        self.hideAucAnim.setDuration(500)
        self.hideAucAnim.setStartValue(self.training.w.autoCrop.height())
        self.hideAucAnim.setEndValue(0)
        self.hideAucAnim.setEasingCurve(QEasingCurve.Linear)

        self.showDrbAnim = QtCore.QPropertyAnimation(self.training.w.Dreambooth, b"maximumHeight")
        self.showDrbAnim.setDuration(1500)
        self.showDrbAnim.setStartValue(self.training.w.Dreambooth.height())
        self.showDrbAnim.setEndValue(self.training.w.height())
        self.showDrbAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideDrbAnim = QtCore.QPropertyAnimation(self.training.w.Dreambooth, b"maximumHeight")
        self.hideDrbAnim.setDuration(500)
        self.hideDrbAnim.setStartValue(self.training.w.Dreambooth.height())
        self.hideDrbAnim.setEndValue(0)
        self.hideDrbAnim.setEasingCurve(QEasingCurve.Linear)

        self.showLdbAnim = QtCore.QPropertyAnimation(self.training.w.LoraDreambooth, b"maximumHeight")
        self.showLdbAnim.setDuration(1500)
        self.showLdbAnim.setStartValue(self.training.w.LoraDreambooth.height())
        self.showLdbAnim.setEndValue(self.training.w.height())
        self.showLdbAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideLdbAnim = QtCore.QPropertyAnimation(self.training.w.LoraDreambooth, b"maximumHeight")
        self.hideLdbAnim.setDuration(500)
        self.hideLdbAnim.setStartValue(self.training.w.LoraDreambooth.height())
        self.hideLdbAnim.setEndValue(0)
        self.hideLdbAnim.setEasingCurve(QEasingCurve.Linear)

        self.showHpnAnim = QtCore.QPropertyAnimation(self.training.w.hypernetworks, b"maximumHeight")
        self.showHpnAnim.setDuration(1500)
        self.showHpnAnim.setStartValue(self.training.w.hypernetworks.height())
        self.showHpnAnim.setEndValue(self.training.w.height())
        self.showHpnAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideHpnAnim = QtCore.QPropertyAnimation(self.training.w.hypernetworks, b"maximumHeight")
        self.hideHpnAnim.setDuration(500)
        self.hideHpnAnim.setStartValue(self.training.w.hypernetworks.height())
        self.hideHpnAnim.setEndValue(0)
        self.hideHpnAnim.setEasingCurve(QEasingCurve.Linear)

        self.showPitAnim = QtCore.QPropertyAnimation(self.training.w.PrepareInput, b"maximumHeight")
        self.showPitAnim.setDuration(1500)
        self.showPitAnim.setStartValue(self.training.w.PrepareInput.height())
        self.showPitAnim.setEndValue(self.training.w.height())
        self.showPitAnim.setEasingCurve(QEasingCurve.Linear)

        self.hidePitAnim = QtCore.QPropertyAnimation(self.training.w.PrepareInput, b"maximumHeight")
        self.hidePitAnim.setDuration(500)
        self.hidePitAnim.setStartValue(self.training.w.PrepareInput.height())
        self.hidePitAnim.setEndValue(0)
        self.hidePitAnim.setEasingCurve(QEasingCurve.Linear)

        self.showCpdAnim = QtCore.QPropertyAnimation(self.training.w.ckptToDiff, b"maximumHeight")
        self.showCpdAnim.setDuration(1500)
        self.showCpdAnim.setStartValue(self.training.w.ckptToDiff.height())
        self.showCpdAnim.setEndValue(self.training.w.height())
        self.showCpdAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideCpdAnim = QtCore.QPropertyAnimation(self.training.w.ckptToDiff, b"maximumHeight")
        self.hideCpdAnim.setDuration(500)
        self.hideCpdAnim.setStartValue(self.training.w.ckptToDiff.height())
        self.hideCpdAnim.setEndValue(0)
        self.hideCpdAnim.setEasingCurve(QEasingCurve.Linear)

    def show_hide_all_anim(self):
        print(self.showAll)
        if self.showAll == False:
            self.hideCpdAnim.start()
            self.cpdHidden = True
            self.hidePitAnim.start()
            self.pitHidden = True
            self.hidePocAnim.start()
            self.pocHidden = True
            self.hideFpcAnim.start()
            self.fpcHidden = True
            self.hideAucAnim.start()
            self.aucHidden = True
            self.hideDrbAnim.start()
            self.drbHidden = True
            self.hideLdbAnim.start()
            self.ldbHidden = True
            self.hideHpnAnim.start()
            self.hpnHidden = True
            self.showAll = True
        elif self.showAll == True:
            self.showCpdAnim.start()
            self.cpdHidden = False
            self.showPitAnim.start()
            self.pitHidden = False
            self.showPocAnim.start()
            self.pocHidden = False
            self.showFpcAnim.start()
            self.fpcHidden = False
            self.showAucAnim.start()
            self.aucHidden = False
            self.showDrbAnim.start()
            self.drbHidden = False
            self.showLdbAnim.start()
            self.ldbHidden = False
            self.showHpnAnim.start()
            self.hpnHidden = False
            self.showAll = False

    def connections(self):
        self.training.w.pathInputImages.clicked.connect(self.set_path_to_input_image)
        self.training.w.logsFolder.clicked.connect(self.set_path_to_logfiles)
        self.training.w.resumeModel.clicked.connect(self.set_path_to_resume_model)
        self.training.w.initEmbeddingManager.clicked.connect(self.set_path_to_init_embedding_manager)
        self.training.w.Start.clicked.connect(self.start_dreambooth)
        self.training.w.Stop.clicked.connect(self.stop_dreambooth)
        self.training.w.ckpt2diff_start_process.clicked.connect(self.ckpt2diff_start_process)
        self.training.w.ckpt2diff_select_source.clicked.connect(self.ckpt2diff_select_source)
        self.training.w.ckpt2diff_select_destination.clicked.connect(self.ckpt2diff_select_destination)

        self.training.w.toggle_ckpt2diff.stateChanged.connect(self.hideCkpt2diff_anim)
        self.training.w.toggle_caption.stateChanged.connect(self.hideProcessCaption_anim)
        self.training.w.toggle_focal_crop.stateChanged.connect(self.hideFocalPointCrop_anim)
        self.training.w.toggle_split_oversize.stateChanged.connect(self.hideAutoCrop_anim)
        self.training.w.toggle_dreambooth.stateChanged.connect(self.hideDreambooth_anim)
        self.training.w.toggle_lora_dreambooth.stateChanged.connect(self.hideLoraDreambooth_anim)
        self.training.w.toggle_hypernetwork.stateChanged.connect(self.hideHypernetwork_anim)
        self.training.w.toggle_prepare_input.stateChanged.connect(self.hidePrepareInput_anim)

        self.training.w.ldb_select_pretrained_model_name_or_path.clicked.connect(self.ldb_select_pretrained_model_name_or_path)
        self.training.w.ldb_select_instance_data_dir.clicked.connect(self.ldb_select_instance_data_dir)
        self.training.w.ldb_select_output_dir.clicked.connect(self.ldb_select_output_dir)
        self.training.w.ldb_select_logging_dir.clicked.connect(self.ldb_select_logging_dir)
        self.training.w.ldb_select_class_data_dir.clicked.connect(self.ldb_select_class_data_dir)
        self.training.w.start_lora_dreambooth.clicked.connect(self.start_lora_dreambooth)


    def get_lora_dreambooth_args(self):
        args = SimpleNamespace(**{})
        args.pretrained_model_name_or_path = self.training.w.pretrained_model_name_or_path.text()
        args.instance_data_dir = self.training.w.instance_data_dir.text()
        args.output_dir = self.training.w.output_dir.text()
        args.logging_dir = self.training.w.logging_dir.text()
        args.instance_prompt = self.training.w.instance_prompt.text()
        args.with_prior_preservation = self.training.w.with_prior_preservation.isChecked()
        args.prior_loss_weight = self.training.w.prior_loss_weight.value()
        args.class_data_dir = self.training.w.class_data_dir.text()
        args.class_prompt = self.training.w.class_prompt.text()
        args.num_class_images = self.training.w.num_class_images.value()
        args.resolution = self.training.w.resolution.value()
        args.train_batch_size = self.training.w.train_batch_size.value()
        args.sample_batch_size = self.training.w.sample_batch_size.value()
        args.learning_rate = self.training.w.learning_rate.value()
        args.lr_scheduler = self.training.w.lr_scheduler.currentText()
        args.adam_beta1 = self.training.w.adam_beta1.value()
        args.adam_beta2 = self.training.w.adam_beta2.value()
        args.lr_warmup_steps = self.training.w.lr_warmup_steps.value()
        args.num_train_epochs = self.training.w.num_train_epochs.value()
        args.max_train_steps = self.training.w.max_train_steps.value()
        args.adam_epsilon = float(self.training.w.adam_epsilon.text())
        args.seed = self.training.w.seed.text()
        args.gradient_accumulation_steps = self.training.w.gradient_accumulation_steps.value()
        args.max_grad_norm = self.training.w.max_grad_norm.value()
        args.mixed_precision = self.training.w.mixed_precision.currentText()
        args.center_crop = self.training.w.center_crop.isChecked()
        args.train_text_encoder = self.training.w.train_text_encoder.isChecked()
        args.gradient_checkpointing = self.training.w.gradient_checkpointing.isChecked()
        args.scale_lr = self.training.w.scale_lr.isChecked()
        args.use_8bit_adam = self.training.w.use_8bit_adam.isChecked()
        return args

    def start_lora_dreambooth(self):
        self.parent.plugin_thread(self.start_lora_dreambooth_thread)


    def start_lora_dreambooth_thread(self, progress_callback=None):
        args = self.get_lora_dreambooth_args()
        run_lora_dreambooth(args)

    def ldb_select_class_data_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to class Images for training')
        print(filename)
        self.training.w.class_data_dir.setText(filename)

    def ldb_select_logging_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Logging Dir')
        print(filename)
        self.training.w.logging_dir.setText(filename)

    def ldb_select_output_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to save output')
        print(filename)
        self.training.w.output_dir.setText(filename)

    def ldb_select_pretrained_model_name_or_path(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to diffuser model to be trained on')
        print(filename)
        self.training.w.pretrained_model_name_or_path.setText(filename)

    def ldb_select_instance_data_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to Instance Images for training')
        print(filename)
        self.training.w.instance_data_dir.setText(filename)

    def load_folder_content(self):
        self.training.w.base.clear()
        models = os.listdir(gs.system.dreambooth_config)
        #self.parent.path_setup.w.activeModel.setText(gs.system.sdPath)
        for model in models:
            location = os.path.join(gs.system.dreambooth_config, model)
            self.training.w.base.addItem(model)

    @Slot()
    def ckpt2diff_select_source(self):
        filename = QFileDialog.getOpenFileName(caption='CKPT to translate to diffuser', filter='Checkpoint (*.ckpt)')
        print(filename)
        self.training.w.checkpoint_path.setText(filename[0])

    @Slot()
    def ckpt2diff_select_destination(self):
        filename = QFileDialog.getExistingDirectory(caption='Path save diffuser model')
        print(filename)
        self.training.w.dump_path.setText(filename)


    @Slot()
    def set_path_to_input_image(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to input Images')
        print(filename)
        self.training.w.data_root.setText(filename)

    @Slot()
    def set_path_to_logfiles(self):
        filename = QFileDialog.getExistingDirectory(caption='Logfile Path')
        print(filename)
        self.training.w.logdir.setText(filename)

    @Slot()
    def set_path_to_resume_model(self):
        filename = QFileDialog.getOpenFileName(caption='Model to resume from', filter='Checkpoint (*.ckpt)')
        self.training.w.actual_resume.setText(filename[0])

    @Slot()
    def set_path_to_init_embedding_manager(self):
        filename = QFileDialog.getOpenFileName(caption='Initialize embedding manager from a checkpoint', filter='Checkpoint (*.ckpt)')
        self.training.w.init_embedding_manager_ckpt.setText(filename[0])


    def ckpt2diff_start_process(self):
        gs.ti_grad_flag_switch = True
        self.parent.plugin_thread(self.ckpt2diff_start_process_thread)

    def ckpt2diff_start_process_thread(self, progress_callback=None):
        print('translation ckpt to diffuser started')
        run_translation(
            checkpoint_path=self.training.w.checkpoint_path.text(),
            dump_path=self.training.w.dump_path.text(),
            original_config_file=self.training.w.original_config_file.currentText(),
            num_in_channels=self.training.w.num_in_channels.value(),
            scheduler_type=self.training.w.scheduler_type.currentText(),
            image_size=int(self.training.w.image_size.currentText()),
            prediction_type=self.training.w.prediction_type.currentText(),
            extract_ema=self.training.w.extract_ema.isChecked()
        )
        print('translation ckpt to diffuser finished')

    def start_dreambooth(self):
        gs.ti_grad_flag_switch = True
        self.parent.plugin_thread(self.create_dreambooth)

    def stop_dreambooth(self):
        self.signals.dreambooth_stop_signal.emit()

    def setup_view(self):

        for dock in self.parent.w.findChildren(QDockWidget):
            dock.hide()

        self.training.w.dockWidget.setWindowTitle('Textual Inversion (dreambooth)')
        self.parent.w.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.training.w.dockWidget)


    def create_dreambooth(self, progress_callback=None):
        self.dreambooth_training.dreambooth()
        """
        self.parent.ti.create_txt_inv(
            name=self.training.w.name.text(),
            data_root=self.training.w.data_root.text(),
            #base=os.path.join(gs.system.dreambooth_config,self.training.w.base.currentText()),
            logdir=self.training.w.logdir.text(),
            placeholder_tokens=self.training.w.placeholder_tokens.text(),
            init_word=self.training.w.init_word.text(),
            actual_resume=self.training.w.actual_resume.text(),
            no_test=self.training.w.no_test.isChecked(),
            train=self.training.w.train.isChecked()
        )"""
