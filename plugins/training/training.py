import os
import shutil
from types import SimpleNamespace

from PySide6 import QtCore, QtUiTools
from PySide6.QtWidgets import QDockWidget, QFileDialog
from PySide6.QtCore import Slot, Signal, QObject, QFile, QEasingCurve
from backend.singleton import singleton

gs = singleton

from plugins.training.dreambooth_class import DreamBooth
from plugins.training.sd_to_diffusers import run_translation
from plugins.training.train_lora_dreambooth import run_lora_dreambooth
from plugins.training.lora_diffusion.cli_lora_add import lom_merge_models
from plugins.training.diffuser_to_sd import diff2sd


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
        self.training.w.lom_select_model_a_lora.setVisible(False)



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

    def hideLoraMerge_anim(self):
        if self.lomHidden is True:
            self.showLomAnim.start()
        else:
            self.hideLomAnim.start()
        self.lomHidden = not self.lomHidden

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

        self.showLomAnim = QtCore.QPropertyAnimation(self.training.w.LoraMerge, b"maximumHeight")
        self.showLomAnim.setDuration(1500)
        self.showLomAnim.setStartValue(self.training.w.LoraMerge.height())
        self.showLomAnim.setEndValue(self.training.w.height())
        self.showLomAnim.setEasingCurve(QEasingCurve.Linear)

        self.hideLomAnim = QtCore.QPropertyAnimation(self.training.w.LoraMerge, b"maximumHeight")
        self.hideLomAnim.setDuration(500)
        self.hideLomAnim.setStartValue(self.training.w.LoraMerge.height())
        self.hideLomAnim.setEndValue(0)
        self.hideLomAnim.setEasingCurve(QEasingCurve.Linear)

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
            self.hideLomAnim.start()
            self.lomHidden = True
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
            self.showLomAnim.start()
            self.lomHidden = False
            self.showAll = False

    def connections(self):
        self.training.w.pathInputImages.clicked.connect(self.set_path_to_input_image)
        self.training.w.logsFolder.clicked.connect(self.set_path_to_logfiles)
        self.training.w.resumeModel.clicked.connect(self.set_path_to_resume_model)
        #self.training.w.initEmbeddingManager.clicked.connect(self.set_path_to_init_embedding_manager)
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
        self.training.w.toggle_lora_merge.stateChanged.connect(self.hideLoraMerge_anim)
        self.training.w.ldb_select_pretrained_model_name_or_path.clicked.connect(self.ldb_select_pretrained_model_name_or_path)
        self.training.w.ldb_select_instance_data_dir.clicked.connect(self.ldb_select_instance_data_dir)
        self.training.w.ldb_select_output_dir.clicked.connect(self.ldb_select_output_dir)
        self.training.w.ldb_select_logging_dir.clicked.connect(self.ldb_select_logging_dir)
        self.training.w.ldb_select_class_data_dir.clicked.connect(self.ldb_select_class_data_dir)
        self.training.w.ldb_start_lora_dreambooth.clicked.connect(self.ldb_start_lora_dreambooth)
        self.training.w.lom_select_model_a.clicked.connect(self.lom_select_model_a)
        self.training.w.lom_select_model_a_lora.clicked.connect(self.lom_select_model_a_lora)
        self.training.w.lom_select_model_b.clicked.connect(self.lom_select_model_b)
        self.training.w.lom_select_output_dir.clicked.connect(self.lom_select_output_dir)
        self.training.w.lom_start_merge.clicked.connect(self.lom_start_merge)
        self.training.w.lora2diff.toggled.connect(self.lom_set_model_select_buttons)


    def lom_set_model_select_buttons(self):
        if self.training.w.lora2diff.isChecked():
            self.training.w.lom_select_model_a_lora.setVisible(False)
            self.training.w.lom_select_model_a.setVisible(True)
        else:
            self.training.w.lom_select_model_a_lora.setVisible(True)
            self.training.w.lom_select_model_a.setVisible(False)

    def lom_start_merge(self):
        self.parent.plugin_thread(self.lom_start_merge_thread)


    def lom_start_merge_thread(self, progress_callback=None):
        print('merge started')
        lom_output_type = 'pt'
        lom_output_type = 'ckpt' if 'ckpt' in self.training.w.path_1.text() else lom_output_type
        lom_output_type = 'ckpt' if 'ckpt' in self.training.w.path_2.text() else lom_output_type

        lom_merge_models(
            path_1=self.training.w.path_1.text(),
            path_2=self.training.w.path_2.text(),
            output_path=os.path.join(self.training.w.output_path.text(),'merged_lora.' + lom_output_type ),
            alpha=self.training.w.alpha.value(),
            mode=self.training.w.mode.currentText()
        )
        args = SimpleNamespace(**{})
        args.model_path = os.path.join(self.training.w.output_path.text(),'merged_lora')
        args.checkpoint_path = os.path.join(self.training.w.output_path.text(),'merged_lora.ckpt')
        args.half=self.training.w.ckpt_half.isChecked()
        diff2sd(args)
        shutil.rmtree(args.model_path)
        print('merge finished')

    def lom_select_model_a(self):
        filename = QFileDialog.getExistingDirectory(caption='Model A to merge')
        self.training.w.path_1.setText(filename)

    def lom_select_model_a_lora(self):
        filename = QFileDialog.getOpenFileName(caption='Lora Model A to merge', filter='Model (*.pt)')
        self.training.w.path_1.setText(filename[0])

    def lom_select_model_b(self):
        filename = QFileDialog.getOpenFileName(caption='Model B to merge', filter='Model (*.pt)')
        self.training.w.path_2.setText(filename[0])

    def lom_select_output_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to output merged model')
        self.training.w.output_path.setText(filename)


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
        args.resolution = int(self.training.w.resolution.currentText())
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
        seed = self.training.w.seed.text()
        if seed == '':
            args.seed = None
        else:
            args.seed = int(seed)
        args.gradient_accumulation_steps = self.training.w.gradient_accumulation_steps.value()
        args.max_grad_norm = self.training.w.max_grad_norm.value()
        args.mixed_precision = self.training.w.mixed_precision.currentText()
        args.center_crop = self.training.w.center_crop.isChecked()
        args.train_text_encoder = self.training.w.train_text_encoder.isChecked()
        args.gradient_checkpointing = self.training.w.gradient_checkpointing.isChecked()
        args.scale_lr = self.training.w.scale_lr.isChecked()
        args.use_8bit_adam = self.training.w.use_8bit_adam.isChecked()
        args.adam_weight_decay = self.training.w.adam_weight_decay.value()
        args.save_steps = self.training.w.save_steps.value()
        return args

    def ldb_start_lora_dreambooth(self):
        self.parent.plugin_thread(self.ldb_start_lora_dreambooth_thread)


    def ldb_start_lora_dreambooth_thread(self, progress_callback=None):
        args = self.get_lora_dreambooth_args()
        run_lora_dreambooth(args)

    def ldb_select_class_data_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to class Images for training')
        self.training.w.class_data_dir.setText(filename)

    def ldb_select_logging_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Logging Dir')
        self.training.w.logging_dir.setText(filename)

    def ldb_select_output_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to save output')
        self.training.w.output_dir.setText(filename)

    def ldb_select_pretrained_model_name_or_path(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to diffuser model to be trained on')
        self.training.w.pretrained_model_name_or_path.setText(filename)

    def ldb_select_instance_data_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Dir to Instance Images for training')
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
        self.training.w.checkpoint_path.setText(filename[0])

    @Slot()
    def ckpt2diff_select_destination(self):
        filename = QFileDialog.getExistingDirectory(caption='Path save diffuser model')
        self.training.w.dump_path.setText(filename)


    @Slot()
    def set_path_to_input_image(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to input Images')
        self.training.w.data_root.setText(filename)

    @Slot()
    def set_path_to_logfiles(self):
        filename = QFileDialog.getExistingDirectory(caption='Logfile Path')
        self.training.w.logdir.setText(filename)

    @Slot()
    def set_path_to_resume_model(self):
        filename = QFileDialog.getOpenFileName(caption='Model to resume from', filter='Checkpoint (*.ckpt)')
        self.training.w.actual_resume.setText(filename[0])

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
        self.dreambooth_training.dreambooth(
            accelerator=self.training.w.accelerator.currentText(),                                   # Previously known as distributed_backend (dp, ddp, ddp2, etc...).
            # Can also take in an accelerator object for custom hardware.
            accumulate_grad_batches=None,                        # Accumulates grads every k batches or as set up in the dict.
            amp_backend=self.training.w.amp_backend.currentText(),                                # The mixed precision backend to use ("native" or "apex")
            amp_level=None,                                      # The optimization level to use (O1, O2, etc...).
            auto_lr_find=self.training.w.auto_lr_find.isChecked(),                                  # If set to True, will make trainer.tune() run a learning rate finder,
            # trying to optimize initial learning for faster convergence. trainer.tune() method will
            # set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
            # To use a different key set a string instead of True with the key name.
            auto_scale_batch_size=self.training.w.auto_scale_batch_size.isChecked(),                         # If set to True, will `initially` run a batch size
            # finder trying to find the largest batch size that fits into memory.
            # The result will be stored in self.batch_size in the LightningModule.
            # Additionally, can be set to either `power` that estimates the batch size through
            # a power search or `binsearch` that estimates the batch size through a binary search.
            auto_select_gpus=self.training.w.auto_select_gpus.isChecked(),                              # If enabled and `gpus` is an integer, pick available
            # gpus automatically. This is especially useful when
            # GPUs are configured to be in "exclusive mode", such
            # that only one process at a time can access them.

            actual_resume=self.training.w.actual_resume.text(),







            benchmark=self.training.w.benchmark.isChecked(),                                     # If true enables cudnn.benchmark.
            base=['plugins/training/configs/v1-finetune_unfrozen.yaml'],
            callbacks=None,                                      # Add a callback or list of callbacks.
            checkpoint_callback=self.training.w.checkpoint_callback.isChecked(),                           # If ``True``, enable checkpointing.
            # It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
            # :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
            check_val_every_n_epoch=self.training.w.check_val_every_n_epoch.value(),                           # Check val every n train epochs.
            class_word='<xxx>',
            default_root_dir=None,                               # Default path for logs and weights when no logger/ckpt_callback passed.
            # Default: ``os.getcwd()``.
            # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            deterministic=self.training.w.deterministic.isChecked(),                                 # If true enables cudnn.deterministic.
            devices=None,                                        # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
            # based on the accelerator type.
            debug=False,
            datadir_in_name=True,
            data_root=self.training.w.data_root.text(),
            detect_anomaly=False,
            enable_checkpointing=True,
            enable_model_summary=True,
            enable_progress_bar=True,
            embedding_manager_ckpt='',




            fast_dev_run=False,                                  # runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
            # of train, val and test to find any bugs (ie: a sort of unit test).
            flush_logs_every_n_steps=self.training.w.flush_logs_every_n_steps.value(),                        # How often to flush logs to disk (defaults to every 100 steps).
            gpus='0,',                                           # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
            gradient_clip_val=self.training.w.gradient_clip_val.value(),                                 # 0 means don't clip.
            gradient_clip_algorithm=self.training.w.gradient_clip_algorithm.currentText(),                      # 'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'
            ipus=None,                                           # How many IPUs to train on.
            init_word=self.training.w.init_word.text(),

            limit_train_batches=self.training.w.limit_train_batches.value(),                             # How much of training dataset to check (float = fraction, int = num_batches)
            limit_val_batches=self.training.w.limit_val_batches.value(),                               # How much of validation dataset to check (float = fraction, int = num_batches)
            limit_test_batches=self.training.w.limit_test_batches.value(),                              # How much of test dataset to check (float = fraction, int = num_batches)
            limit_predict_batches=self.training.w.limit_predict_batches.value(),                           # How much of prediction dataset to check (float = fraction, int = num_batches)



            logger=True,                                         # Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
            # the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
            # provided and the `save_dir` property of that logger is not set, local files (checkpoints,
            # profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
            # of the individual loggers.
            log_gpu_memory=None,                                 # 'min_max', 'all'. Might slow performance

            log_every_n_steps=50,                                # How often to log within steps (defaults to every 50 steps).
            logdir=self.training.w.logdir.text(),
            move_metrics_to_cpu=self.training.w.move_metrics_to_cpu.isChecked(),                           # Whether to force internal logged metrics to be moved to cpu.
            # This can save some gpu memory, but can make training slower. Use with attention.
            multiple_trainloader_mode=self.training.w.multiple_trainloader_mode.currentText(),          # How to loop over the datasets when there are multiple train loaders.
            # In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
            # and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
            # reload when reaching the minimum length of datasets.
            max_epochs=None if self.training.w.max_epochs.value() == 0 else self.training.w.max_epochs.value(),                                     # Stop training once this number of epochs is reached. Disabled by default (None).
            # If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.
            min_epochs=None if self.training.w.min_epochs.value() == 0 else self.training.w.min_epochs.value(),                                     # Force training for at least these many epochs. Disabled by default (None).
            # If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.
            max_steps=self.training.w.max_steps.value(),                                        # Stop training after this number of steps. Disabled by default (None).
            min_steps=None if self.training.w.min_steps.value() == 0 else self.training.w.min_steps.value(),                                      # Force training for at least these number of steps. Disabled by default (None).
            max_time=None if self.training.w.max_time.value() == 0 else self.training.w.max_time.value(),                                       # Stop training after this amount of time has passed. Disabled by default (None).
            # The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
            # :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
            # :class:`datetime.timedelta`.
            name=self.training.w.name.text(),
            num_nodes=1,                                         # number of GPU nodes for distributed training.
            num_processes=1,                                     # number of processes for distributed training with distributed_backend="ddp_cpu"
            num_sanity_val_steps=None if self.training.w.num_sanity_val_steps.value() == 0 else self.training.w.num_sanity_val_steps.value(),                           # Sanity check runs n validation batches before starting the training routine.
            # Set it to `-1` to run all batches in all validation dataloaders.

            no_test=False,
            overfit_batches=0.0,                                 # Overfit a fraction of training data (float) or a set number of batches (int).
            project=None,
            postfix='',

            prepare_data_per_node=None,                          # If True, each LOCAL_RANK=0 will call prepare data.
            # Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
            process_position=0,                                  # orders the progress bar when running multiple models on same machine.
            progress_bar_refresh_rate=None,                      # How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
            # Ignored when a custom progress bar is passed to :paramref:`~Trainer.callbacks`. Default: None, means
            # a suitable value will be chosen based on the environment (terminal, Google COLAB, etc.).
            profiler=None,                                       # To profile individual steps during training and assist in identifying bottlenecks.

            plugins=None,                                        # Plugins allow modification of core behavior like ddp and amp, and enable custom lightning plugins.
            progress_callback=None,

            precision=int(self.training.w.precision.currentText()),                                        # Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or
            # TPUs.

            reload_dataloaders_every_n_epochs=self.training.w.reload_dataloaders_every_n_epochs.value(),                 # Set to a non-negative integer to reload dataloaders every n epochs.
            # Default: 0

            resume_from_checkpoint=None,                         # Path/URL of the checkpoint from which training is resumed. If there is
            # no checkpoint file at the path, start from scratch. If resuming from mid-epoch checkpoint,
            # training will start from the beginning of the next epoch.
            replace_sampler_ddp=True,                            # Explicitly enables or disables sampler replacement. If not specified this
            # will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
            # train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
            # you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
            resume='',
            reg_data_root='data/input/regularization/images',
            stochastic_weight_avg=self.training.w.stochastic_weight_avg.isChecked(),                         # Whether to use `Stochastic Weight Averaging (SWA)
            # <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_
            strategy=None,
            seed=23,
            scale_lr=False,
            sync_batchnorm=False,                                # Synchronize batch norm layers between process groups/whole world.
            terminate_on_nan=self.training.w.stochastic_weight_avg.isChecked(),                               # If set to True, will terminate training (by raising a `ValueError`) at the
            # end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
            tpu_cores=None,                                      # How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            track_grad_norm=-1,                                  # -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

            train=True,

            weights_summary='top',                               # Prints a summary of the weights when training begins.
            weights_save_path=None,                              # Where to save weights if specified. Will override default_root_dir
            # for checkpoints only. Use this if for whatever reason you need the checkpoints
            # stored in a different place than the logs written in `default_root_dir`.
            # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            # Defaults to `default_root_dir`.
            val_check_interval=self.training.w.val_check_interval.value()                               # How often to check the validation set. Use float to check within a training epoch,
            # use int to check every n steps (batches).
        )
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
