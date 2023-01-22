import os
import shutil
from types import SimpleNamespace
from pytorch_lightning import seed_everything
from PySide6 import QtCore, QtUiTools
from PySide6.QtWidgets import QDockWidget, QFileDialog
from PySide6.QtCore import Slot, Signal, QObject, QFile

from backend.torch_gc import torch_gc
from backend.singleton import singleton

gs = singleton

from plugins.training.dreambooth_class import DreamBooth
from plugins.training.sd_to_diffusers import run_translation
from plugins.training.train_lora_dreambooth import run_lora_dreambooth
from plugins.training.diffusers.cli_lora_add import add as lom_merge_models
from plugins.training.diffuser_to_sd import diff2sd
from plugins.training.txt_inv.textual_inversion import create_txt_inv
from plugins.training.preprocess.preprocess import preprocess
from plugins.training.diffusers.dreambooth import run_diff_dreambooth

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
        #self.txt_invers = TI()
        self.parent = parent
        os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
        self.training = FineTune()
        sshFile="frontend/style/elegantDark.stylesheet"
        with open(sshFile,"r") as fh:
            self.training.w.setStyleSheet(fh.read())


    def initme(self):
        print("training")
        gs.system.dreambooth_config = "plugins/training/configs/dreambooth"
        self.signals = Callbacks()
        self.connections()
        self.load_folder_content()
        self.dreambooth_training = DreamBooth()
        self.showAll = False
        self.training.w.show()
        self.training.w.lom_select_model_a_lora.setVisible(False)




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

        self.training.w.ti_start_textual_inversion.clicked.connect(self.create_textual_inversion)
        self.training.w.ti_stop_textual_inversion.clicked.connect(self.ti_stop_textual_inversion)
        self.training.w.ti_select_log_dir.clicked.connect(self.ti_select_log_dir)
        self.training.w.ti_select_image_dir.clicked.connect(self.ti_select_image_dir)
        self.training.w.ti_select_output_dir.clicked.connect(self.ti_select_output_dir)
        self.training.w.ti_select_model.clicked.connect(self.ti_select_model)

        self.training.w.select_process_input_files.clicked.connect(self.select_process_input_files)
        self.training.w.select_process_destination_folder.clicked.connect(self.select_procerss_destination_folder)
        self.training.w.runPreprocess.clicked.connect(self.start_preprocess)


        self.training.w.df_select_model.clicked.connect(self.df_select_model)
        self.training.w.df_select_images.clicked.connect(self.df_select_images)
        self.training.w.df_select_class_images.clicked.connect(self.df_select_class_images)
        self.training.w.df_select_output_folder.clicked.connect(self.df_select_output_folder)
        self.training.w.df_select_log_folder.clicked.connect(self.df_select_log_folder)



        self.training.w.start_diffuser_dreambooth.clicked.connect(self.run_df_dreambooth)


    @Slot()
    def df_select_model(self):
        filename = QFileDialog.getExistingDirectory(caption='Model Path')
        self.training.w.df_pretrained_model_name_or_path.setText(filename)

    @Slot()
    def df_select_images(self):
        filename = QFileDialog.getExistingDirectory(caption='Input Images Path')
        self.training.w.df_instance_data_dir.setText(filename)

    @Slot()
    def df_select_class_images(self):
        filename = QFileDialog.getExistingDirectory(caption='Class Images Path')
        self.training.w.df_class_data_dir.setText(filename)

    @Slot()
    def df_select_output_folder(self):
        filename = QFileDialog.getExistingDirectory(caption='Output Path')
        self.training.w.df_output_dir.setText(filename)

    @Slot()
    def df_select_log_folder(self):
        filename = QFileDialog.getExistingDirectory(caption='Logging Path')
        self.training.w.df_logging_dir.setText(filename)


    @Slot()
    def select_process_input_files(self):
        filename = QFileDialog.getExistingDirectory(caption='Input Images Path')
        self.training.w.process_src.setText(filename)

    @Slot()
    def select_procerss_destination_folder(self):
        filename = QFileDialog.getExistingDirectory(caption='Output Images Path')
        self.training.w.process_dst.setText(filename)

    def start_preprocess(self):
        torch_gc()
        self.parent.run_as_thread(self.run_preprocess)

    def run_preprocess(self, progress_callback=False):
        gs.state.interrupted = False
        preprocess(
            process_src=self.training.w.process_src.text(),
            process_dst=self.training.w.process_dst.text(),
            process_width=self.training.w.process_width.value(),
            process_height=self.training.w.process_height.value(),
            preprocess_txt_action=self.training.w.preprocess_txt_action.currentText(),
            process_flip=self.training.w.process_flip.isChecked(),
            process_split=self.training.w.toggle_split_oversize.isChecked(),
            process_caption=self.training.w.process_caption.isChecked(),
            process_caption_deepbooru=self.training.w.process_caption_deepbooru.isChecked(),
            split_threshold=self.training.w.split_threshold.value()/100,
            overlap_ratio=self.training.w.overlap_ratio.value()/100,
            process_focal_crop=self.training.w.toggle_focal_crop.isChecked(),
            process_focal_crop_face_weight=self.training.w.process_focal_crop_face_weight.value()/10,
            process_focal_crop_entropy_weight=self.training.w.process_focal_crop_entropy_weight.value()/10,
            process_focal_crop_edges_weight=self.training.w.process_focal_crop_edges_weight.value()/10,
            process_focal_crop_debug=self.training.w.process_focal_crop_debug.isChecked(),
            style_caption=self.training.w.style_caption.isChecked())


    def ti_select_log_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to store logs')
        self.training.w.ti_logging_dir.setText(filename)

    def ti_select_image_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to training images')
        self.training.w.ti_train_data_dir.setText(filename)

    def ti_select_output_dir(self):
        filename = QFileDialog.getExistingDirectory(caption='Path to reqularization images')
        self.training.w.ti_output_dir.setText(filename)

    def ti_select_model(self):
        filename = QFileDialog.getExistingDirectory(caption='Select model')
        self.training.w.ti_pretrained_model_name_or_path.setText(filename)

    def ti_stop_textual_inversion(self):
        self.txt_invers.stop_textual_inversion()

    def lom_set_model_select_buttons(self):
        if self.training.w.lora2diff.isChecked():
            self.training.w.lom_select_model_a_lora.setVisible(False)
            self.training.w.lom_select_model_a.setVisible(True)
        else:
            self.training.w.lom_select_model_a_lora.setVisible(True)
            self.training.w.lom_select_model_a.setVisible(False)

    def lom_start_merge(self):
        torch_gc()
        self.parent.run_as_thread(self.lom_start_merge_thread)



    def lom_start_merge_thread(self, progress_callback=None):
        print('merge started')
        lom_output_type = 'pt'
        lom_output_type = 'ckpt' if 'ckpt' in self.training.w.path_1.text() else lom_output_type
        lom_output_type = 'ckpt' if 'ckpt' in self.training.w.path_2.text() else lom_output_type

        mode = self.training.w.mode.currentText()

        lom_merge_models(
            path_1=self.training.w.path_1.text(),
            path_2=self.training.w.path_2.text(),
            output_path=os.path.join(self.training.w.output_path.text(),'merged_lora.' + lom_output_type ),
            alpha=self.training.w.alpha.value(),
            mode=mode,
            with_text_lora= self.training.w.with_text_lora.isChecked(),
            half = self.training.w.ckpt_half.isChecked()
        )
        args = SimpleNamespace(**{})
        args.model_path = os.path.join(self.training.w.output_path.text(),'merged_lora')
        args.checkpoint_path = os.path.join(self.training.w.output_path.text(),'merged_lora.ckpt')
        args.half=self.training.w.ckpt_half.isChecked()
        if mode != "upl-ckpt-v2":
            diff2sd(args)
            shutil.rmtree(args.model_path)
        print('merge finished')
        torch_gc()

    def lom_select_model_a(self):
        filename = QFileDialog.getExistingDirectory(caption='Model A to merge')
        self.training.w.path_1.setText(filename)

    def lom_select_model_a_lora(self):
        filename = QFileDialog.getOpenFileName(caption='Lora Model A to merge', filter='Model (*.pt)')
        self.training.w.path_1.setText(filename[0])

    def lom_select_model_b(self):
        filename = QFileDialog.c(caption='Model B to merge', filter='Model (*.pt)')
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
        args.color_jitter = self.training.w.color_jitter.isChecked()
        args.learning_rate_text = self.training.w.learning_rate_text.value()
        args.push_to_hub = False
        args.hub_token = None
        args.hub_model_id = None
        args.tokenizer_name = None
        args.revision = None

        return args

    def ldb_start_lora_dreambooth(self):
        torch_gc()
        self.parent.run_as_thread(self.ldb_start_lora_dreambooth_thread)


    def ldb_start_lora_dreambooth_thread(self, progress_callback=None):
        args = self.get_lora_dreambooth_args()
        run_lora_dreambooth(args)
        torch_gc()

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
        self.parent.run_as_thread(self.ckpt2diff_start_process_thread)

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
        self.parent.run_as_thread(self.create_dreambooth_thread)

    def stop_dreambooth(self):
        self.signals.dreambooth_stop_signal.emit()

    def setup_view(self):

        for dock in self.parent.w.findChildren(QDockWidget):
            dock.hide()

        self.training.w.dockWidget.setWindowTitle('Textual Inversion (dreambooth)')
        self.parent.w.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.training.w.dockWidget)


    def create_dreambooth_thread(self, progress_callback=None):
        torch_gc()
        self.dreambooth_training.dreambooth(
            accelerator=None if self.training.w.accelerator.currentText() == 'None' else self.training.w.accelerator.currentText(),                                   # Previously known as distributed_backend (dp, ddp, ddp2, etc...).
            # Can also take in an accelerator object for custom hardware.
            accumulate_grad_batches=self.training.w.accumulate_grad_batches.value(),                        # Accumulates grads every k batches or as set up in the dict.
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
            base=['plugins/training/configs/dreambooth/v1-finetune_unfrozen.yaml'],
            callbacks=None,                                      # Add a callback or list of callbacks.
            checkpoint_callback=self.training.w.checkpoint_callback.isChecked(),                           # If ``True``, enable checkpointing.
            # It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
            # :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
            check_val_every_n_epoch=self.training.w.check_val_every_n_epoch.value(),                           # Check val every n train epochs.
            class_word=self.training.w.class_word.text(),    #'<xxx>'
            default_root_dir=self.training.w.logdir.text(),                               # Default path for logs and weights when no logger/ckpt_callback passed.
            # Default: ``os.getcwd()``.
            # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            deterministic=self.training.w.deterministic.isChecked(),                                 # If true enables cudnn.deterministic.
            devices=None,                                        # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
            # based on the accelerator type.
            debug=self.training.w.debug.isChecked(),   #False
            datadir_in_name=self.training.w.datadir_in_name.isChecked(),   #True,
            data_root=self.training.w.data_root.text(),
            detect_anomaly=self.training.w.detect_anomaly.isChecked(),   #False,
            enable_checkpointing=self.training.w.enable_checkpointing.isChecked(),   #True,
            enable_model_summary=self.training.w.enable_model_summary.isChecked(),   #True,
            enable_progress_bar=self.training.w.enable_progress_bar.isChecked(),   #True,
            embedding_manager_ckpt='',




            fast_dev_run=self.training.w.fast_dev_run.value(),                                  # runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
            # of train, val and test to find any bugs (ie: a sort of unit test).
            flush_logs_every_n_steps=self.training.w.flush_logs_every_n_steps.value(),                        # How often to flush logs to disk (defaults to every 100 steps).
            gpus=self.training.w.gpus.text()+ ',',                                           # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
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

            log_every_n_steps=self.training.w.log_every_n_steps.value(),                                # How often to log within steps (defaults to every 50 steps).
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
            num_sanity_val_steps=self.training.w.num_sanity_val_steps.value(),                           # Sanity check runs n validation batches before starting the training routine.
            # Set it to `-1` to run all batches in all validation dataloaders.

            no_test=self.training.w.no_test.isChecked(),
            overfit_batches=int(self.training.w.overfit_batches.value()) if int(self.training.w.overfit_batches.value()) == self.training.w.overfit_batches.value() else self.training.w.overfit_batches.value(),                                 # Overfit a fraction of training data (float) or a set number of batches (int).
            project=None,
            postfix='',

            prepare_data_per_node=None,                          # If True, each LOCAL_RANK=0 will call prepare data.
            # Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data
            process_position=0,                                  # orders the progress bar when running multiple models on same machine.
            progress_bar_refresh_rate=self.training.w.progress_bar_refresh_rate.value(),                      # How often to refresh progress bar (in steps). Value ``0`` disables progress bar.
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
            replace_sampler_ddp=self.training.w.replace_sampler_ddp.isChecked(),                            # Explicitly enables or disables sampler replacement. If not specified this
            # will toggled automatically when DDP is used. By default it will add ``shuffle=True`` for
            # train sampler and ``shuffle=False`` for val/test sampler. If you want to customize it,
            # you can set ``replace_sampler_ddp=False`` and add your own distributed sampler.
            resume='',
            reg_data_root=self.training.w.reg_data_root.text(),
            stochastic_weight_avg=self.training.w.stochastic_weight_avg.isChecked(),                         # Whether to use `Stochastic Weight Averaging (SWA)
            # <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_
            strategy=None,
            seed=seed_everything(int(self.training.w.seed.text()) if self.training.w.seed.text() != '' else -1),
            scale_lr=self.training.w.db_scale_lr.isChecked(),
            sync_batchnorm=self.training.w.sync_batchnorm.isChecked(),                                # Synchronize batch norm layers between process groups/whole world.
            terminate_on_nan=self.training.w.stochastic_weight_avg.isChecked(),                               # If set to True, will terminate training (by raising a `ValueError`) at the
            # end of each training batch, if any of the parameters or the loss are NaN or +/-inf.
            tpu_cores=None,                                      # How many TPU cores to train on (1 or 8) / Single TPU to train on [1]

            track_grad_norm=-1,                                  # -1 no tracking. Otherwise tracks that p-norm. May be set to 'inf' infinity-norm.

            train=self.training.w.train.isChecked(),

            weights_summary='top',                               # Prints a summary of the weights when training begins.
            weights_save_path=None,                              # Where to save weights if specified. Will override default_root_dir
            # for checkpoints only. Use this if for whatever reason you need the checkpoints
            # stored in a different place than the logs written in `default_root_dir`.
            # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
            # Defaults to `default_root_dir`.
            val_check_interval=self.training.w.val_check_interval.value(),                               # How often to check the validation set. Use float to check within a training epoch,
            # use int to check every n steps (batches).
            base_lr=self.training.w.base_lr.value(),
            bs=self.training.w.bs.value(),
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

    def create_textual_inversion(self):
        gs.ti_grad_flag_switch = True
        self.parent.run_as_thread(self.create_textual_inversion_thread)

    def create_textual_inversion_thread(self, progress_callback=None):
        print('Textual Inversion training started')
        create_txt_inv(name=self.training.w.ti_name.text(),
                                       save_steps=self.training.w.ti_save_steps.value(),
                                       only_save_embeds=self.training.w.ti_only_save_embeds.isChecked(),
                                       pretrained_model_name_or_path=self.training.w.ti_pretrained_model_name_or_path.text(),
                                       revision=self.training.w.ti_revision.text(),
                                       tokenizer_name=self.training.w.ti_revision.text(),
                                       train_data_dir=self.training.w.ti_train_data_dir.text(),
                                       placeholder_token=self.training.w.ti_placeholder_token.text(),
                                       initializer_token=self.training.w.ti_initializer_token.text(),
                                       learnable_property=self.training.w.ti_learnable_property.currentText(),
                                       repeats=self.training.w.ti_repeats.value(),
                                       output_dir=self.training.w.ti_output_dir.text(),
                                       resolution=self.training.w.ti_resolution.value(),
                                       seed=-1 if self.training.w.ti_seed.text() == '' else int(self.training.w.ti_seed.text()),
                                       center_crop=self.training.w.ti_center_crop.isChecked(),
                                       train_batch_size=self.training.w.ti_train_batch_size.value(),
                                       num_train_epochs=self.training.w.ti_num_train_epochs.value(),
                                       max_train_steps=self.training.w.ti_max_train_steps.value(),
                                       gradient_accumulation_steps=self.training.w.ti_gradient_accumulation_steps.value(),
                                       learning_rate=self.training.w.ti_learning_rate.value(),
                                       scale_lr=self.training.w.ti_scale_lr.isChecked(),
                                       lr_scheduler=self.training.w.ti_lr_scheduler.currentText(),
                                       lr_warmup_steps=self.training.w.ti_lr_warmup_steps.value(),
                                       adam_beta1=self.training.w.ti_adam_beta1.value(),
                                       adam_beta2=self.training.w.ti_adam_beta2.value(),
                                       adam_weight_decay=self.training.w.ti_adam_weight_decay.value(),
                                       adam_epsilon=float(self.training.w.ti_adam_epsilon.text()),
                                       logging_dir=self.training.w.ti_logging_dir.text(),
                                       mixed_precision=self.training.w.ti_mixed_precision.currentText())
        print('Textual Inversion training finished')





    def run_df_dreambooth(self):
        gs.ti_grad_flag_switch = True
        self.parent.run_as_thread(self.run_df_dreambooth_thread)
    def run_df_dreambooth_thread(self, progress_callback=False):
        run_diff_dreambooth(
            pretrained_model_name_or_path=self.training.w.df_pretrained_model_name_or_path.text(),       # Path to pretrained model or model identifier from huggingface.co/models.
            revision=None,                           # Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be float32 precision.
            tokenizer_name=None if self.training.w.df_tokenizer_name.text() == '' else self.training.w.df_tokenizer_name.text(),                    # Pretrained tokenizer name or path if not the same as model_name
            instance_data_dir=self.training.w.df_instance_data_dir.text(),        # A folder containing the training data of instance images.
            class_data_dir=None if self.training.w.df_class_data_dir.text() == '' else self.training.w.df_class_data_dir.text(),                    # A folder containing the training data of class images.
            instance_prompt=self.training.w.df_instance_prompt.text(),                   # The prompt with identifier specifying the instance
            class_prompt=None if self.training.w.df_class_prompt.text() == '' else self.training.w.df_class_prompt.text(),                      # The prompt to specify images in the same class as provided instance images.
            with_prior_preservation=None,           # Flag to add prior preservation loss.
            prior_loss_weight=self.training.w.df_prior_loss_weight.value(),                  #
            num_class_images=self.training.w.df_num_class_images.value(),                   # Minimal class images for prior preservation loss. If there are not enough images already present in
            # class_data_dir, additional images will be sampled with class_prompt.
            output_dir=self.training.w.df_output_dir.text(),      # The output directory where the model predictions and checkpoints will be written.
            seed=seed_everything(self.training.w.df_seed.text()),                              # A seed for reproducible training.
            resolution=self.training.w.df_resolution.value(),                         # The resolution for input images, all the images in the train/validation dataset will be resized to this resolution
            center_crop=self.training.w.df_center_crop.isChecked(),                      # Whether to center crop images before resizing to resolution
            train_text_encoder=self.training.w.df_train_text_encoder.isChecked(),               # Whether to train the text encoder. If set, the text encoder should be float32 precision.
            # will take additional VRAM
            train_batch_size=self.training.w.df_train_batch_size.value(),                     # Batch size (per device) for the training dataloader.
            sample_batch_size=self.training.w.df_sample_batch_size.value(),                    # Batch size (per device) for sampling images.
            num_train_epochs=self.training.w.df_num_train_epochs.value(),                     # how many epochs are to be trained
            max_train_steps=None if self.training.w.df_max_train_steps.value() == 0 else self.training.w.df_max_train_steps.value(),                   # Total number of training steps to perform. If provided, overrides num_train_epochs.
            checkpointing_steps=self.training.w.df_checkpointing_steps.value(),                # Save a checkpoint of the training state every X updates. These checkpoints can be used both as final
            # checkpoints in case they are better than the last checkpoint, and are also suitable for resuming
            # training using `--resume_from_checkpoint`.
            resume_from_checkpoint=None,            # Whether training should be resumed from a previous checkpoint. Use a path saved by
            # `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.
            gradient_accumulation_steps=self.training.w.df_gradient_accumulation_steps.value(),          # Number of updates steps to accumulate before performing a backward/update pass.
            gradient_checkpointing=self.training.w.df_gradient_checkpointing.isChecked(),           # Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.
            learning_rate=self.training.w.df_learning_rate.value(),                     # Initial learning rate (after the potential warmup period) to use.
            scale_lr=self.training.w.df_scale_lr.isChecked(),                         # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.
            lr_scheduler=self.training.w.df_lr_scheduler.currentText(),                # The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",
            # "constant", "constant_with_warmup"]
            lr_warmup_steps=self.training.w.df_lr_warmup_steps.value(),                      # Number of steps for the warmup in the lr scheduler.
            lr_num_cycles=self.training.w.df_lr_num_cycles.value(),                        # Number of hard resets of the lr in cosine_with_restarts scheduler.
            lr_power=self.training.w.df_lr_power.value(),                           # Power factor of the polynomial scheduler.
            use_8bit_adam=False,                    # Whether to use 8-bit Adam from bitsandbytes.
            adam_beta1=self.training.w.df_adam_beta1.value(),                         # The beta1 parameter for the Adam optimizer.
            adam_beta2=self.training.w.df_adam_beta2.value(),                       # The beta2 parameter for the Adam optimizer.
            adam_weight_decay=self.training.w.df_adam_weight_decay.value(),                 # Weight decay to use.
            adam_epsilon=float(self.training.w.df_adam_epsilon.text()),                     # Epsilon value for the Adam optimizer
            max_grad_norm=self.training.w.df_max_grad_norm.value(),                      # Max gradient norm.
            push_to_hub=False,                      # Whether to push the model to the Hub.
            hub_token=None,                         # The token to use to push to the Model Hub.
            hub_model_id=None,                      # The name of the repository to keep in sync with the local `output_dir`.
            logging_dir=self.training.w.df_logging_dir.text(),                # [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to
            # *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.
            allow_tf32=self.training.w.df_allow_tf32.isChecked(),                       # Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see
            # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
            report_to="tensorboard",                # The integration to report the results and logs to. Supported platforms are `"tensorboard"`
            # (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.
            mixed_precision=None if self.training.w.df_mixed_precision.currentText() == 'no' else self.training.w.df_mixed_precision.currentText(),                   # ["no", "fp16", "bf16"]
            # Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=
            # 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the
            # flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config.
            prior_generation_precision=None if self.training.w.df_prior_generation_precision.currentText() == 'no' else self.training.w.df_prior_generation_precision.currentText(),        # ["no", "fp32", "fp16", "bf16"]
            # Choose prior generation precision between fp32, fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=
            # 1.10.and an Nvidia Ampere GPU.  Default to  fp16 if a GPU is available else fp32.
            local_rank=-1,                          # For distributed training: local_rank
            enable_xformers_memory_efficient_attention=self.training.w.df_enable_xformers_memory_efficient_attention.isChecked())
