import argparse, os, sys, datetime, glob, importlib, csv
from types import SimpleNamespace

import numpy as np
import time
import torch

import torchvision
import pytorch_lightning as pl

from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image

from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info

from plugins.training.callbacks import *
from plugins.training.ldm_db.data.base import Txt2ImgIterableBaseDataset
from plugins.training.ldm_db.util import instantiate_from_config
from backend.singleton import singleton

gs = singleton



def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    return model



class DreamBooth:


    def stop_dreambooth(self):
        # run all checkpoint hooks
        try:
            if self.trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
                self.trainer.save_checkpoint(ckpt_path)
                del self.trainer
        except:
            pass
        finally:
            gs.ti_grad_flag_switch = False

    def get_trainer_config(self, opt):
        trainer_config = {}
        trainer_config['accelerator'] = opt.accelerator
        trainer_config['accumulate_grad_batches'] = opt.accumulate_grad_batches
        trainer_config['amp_backend'] = opt.amp_backend
        trainer_config['amp_level'] = opt.amp_level
        trainer_config['auto_lr_find'] = opt.auto_lr_find
        trainer_config['auto_scale_batch_size'] = opt.auto_scale_batch_size
        trainer_config['auto_select_gpus'] = opt.auto_select_gpus
        trainer_config['benchmark'] = opt.benchmark
        trainer_config['callbacks'] = opt.callbacks
        trainer_config['checkpoint_callback'] = opt.checkpoint_callback
        trainer_config['check_val_every_n_epoch'] = opt.check_val_every_n_epoch
        trainer_config['default_root_dir'] = opt.default_root_dir
        trainer_config['deterministic'] = opt.deterministic
        trainer_config['devices'] = opt.devices
        trainer_config['fast_dev_run'] = opt.fast_dev_run
        trainer_config['flush_logs_every_n_steps'] = opt.flush_logs_every_n_steps
        trainer_config['gpus'] = opt.gpus
        trainer_config['gradient_clip_val'] = opt.gradient_clip_val
        trainer_config['gradient_clip_algorithm'] = opt.gradient_clip_algorithm
        trainer_config['limit_train_batches'] = opt.limit_train_batches
        trainer_config['limit_val_batches'] = opt.limit_val_batches
        trainer_config['limit_test_batches'] = opt.limit_test_batches
        trainer_config['limit_predict_batches'] = opt.limit_predict_batches
        trainer_config['logger'] = opt.logger
        trainer_config['log_gpu_memory'] = opt.log_gpu_memory
        trainer_config['log_every_n_steps'] = opt.log_every_n_steps
        trainer_config['prepare_data_per_node'] = opt.prepare_data_per_node
        trainer_config['process_position'] = opt.process_position
        trainer_config['progress_bar_refresh_rate'] = opt.progress_bar_refresh_rate
        trainer_config['profiler'] = opt.profiler
        trainer_config['overfit_batches'] = opt.overfit_batches
        trainer_config['plugins'] = opt.plugins
        trainer_config['precision'] = opt.precision
        trainer_config['max_epochs'] = opt.max_epochs
        trainer_config['min_epochs'] = opt.min_epochs
        trainer_config['max_steps'] = opt.max_steps
        trainer_config['min_steps'] = opt.min_steps
        trainer_config['max_time'] = opt.max_time
        trainer_config['num_nodes'] = opt.num_nodes
        trainer_config['num_processes'] = opt.num_processes
        trainer_config['num_sanity_val_steps'] = opt.num_sanity_val_steps
        trainer_config['reload_dataloaders_every_n_epochs'] = opt.reload_dataloaders_every_n_epochs
        trainer_config['replace_sampler_ddp'] = opt.replace_sampler_ddp
        trainer_config['resume_from_checkpoint'] = opt.resume_from_checkpoint
        trainer_config['sync_batchnorm'] = opt.sync_batchnorm
        trainer_config['terminate_on_nan'] = opt.terminate_on_nan
        trainer_config['tpu_cores'] = opt.tpu_cores
        trainer_config['ipus'] = opt.ipus
        trainer_config['track_grad_norm'] = opt.track_grad_norm
        trainer_config['val_check_interval'] = opt.val_check_interval
        trainer_config['weights_summary'] = opt.weights_summary
        trainer_config['weights_save_path'] = opt.weights_save_path
        trainer_config['move_metrics_to_cpu'] = opt.move_metrics_to_cpu
        trainer_config['multiple_trainloader_mode'] = opt.multiple_trainloader_mode
        trainer_config['stochastic_weight_avg'] = opt.stochastic_weight_avg
        return trainer_config



    def dreambooth(self,
                   accelerator='ddp',                                   # Previously known as distributed_backend (dp, ddp, ddp2, etc...).
                                                                        # Can also take in an accelerator object for custom hardware.
                   accumulate_grad_batches=None,                        # Accumulates grads every k batches or as set up in the dict.
                   amp_backend='native',                                # The mixed precision backend to use ("native" or "apex")
                   amp_level=None,                                      # The optimization level to use (O1, O2, etc...).
                   auto_lr_find=False,                                  # If set to True, will make trainer.tune() run a learning rate finder,
                                                                        # trying to optimize initial learning for faster convergence. trainer.tune() method will
                                                                        # set the suggested learning rate in self.lr or self.learning_rate in the LightningModule.
                                                                        # To use a different key set a string instead of True with the key name.
                   auto_scale_batch_size=False,                         # If set to True, will `initially` run a batch size
                                                                        # finder trying to find the largest batch size that fits into memory.
                                                                        # The result will be stored in self.batch_size in the LightningModule.
                                                                        # Additionally, can be set to either `power` that estimates the batch size through
                                                                        # a power search or `binsearch` that estimates the batch size through a binary search.
                   auto_select_gpus=False,                              # If enabled and `gpus` is an integer, pick available
                                                                        # gpus automatically. This is especially useful when
                                                                        # GPUs are configured to be in "exclusive mode", such
                                                                        # that only one process at a time can access them.

                   actual_resume='data/models/sd-v1-4-full-ema.ckpt',







                   benchmark=False,                                     # If true enables cudnn.benchmark.
                   base=['plugins/training/configs/dreambooth/v1-finetune_unfrozen.yaml'],
                   callbacks=None,                                      # Add a callback or list of callbacks.
                   checkpoint_callback=False,                           # If ``True``, enable checkpointing.
                                                                        # It will configure a default ModelCheckpoint callback if there is no user-defined ModelCheckpoint in
                                                                        # :paramref:`~pytorch_lightning.trainer.trainer.Trainer.callbacks`.
                   check_val_every_n_epoch=1,                           # Check val every n train epochs.
                   class_word='<xxx>',
                   default_root_dir=None,                               # Default path for logs and weights when no logger/ckpt_callback passed.
                                                                        # Default: ``os.getcwd()``.
                                                                        # Can be remote file paths such as `s3://mybucket/path` or 'hdfs://path/'
                   deterministic=False,                                 # If true enables cudnn.deterministic.
                   devices=None,                                        # Will be mapped to either `gpus`, `tpu_cores`, `num_processes` or `ipus`,
                                                                        # based on the accelerator type.
                   debug=False,
                   datadir_in_name=True,
                   data_root='data/input/testTraining/dog',
                   detect_anomaly=False,
                   enable_checkpointing=True,
                   enable_model_summary=True,
                   enable_progress_bar=True,
                   embedding_manager_ckpt='',




                   fast_dev_run=False,                                  # runs n if set to ``n`` (int) else 1 if set to ``True`` batch(es)
                                                                        # of train, val and test to find any bugs (ie: a sort of unit test).
                   flush_logs_every_n_steps=100,                        # How often to flush logs to disk (defaults to every 100 steps).
                   gpus='0,',                                           # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
                   gradient_clip_val=0,                                 # 0 means don't clip.
                   gradient_clip_algorithm='norm',                      # 'value' means clip_by_value, 'norm' means clip_by_norm. Default: 'norm'
                   ipus=None,                                           # How many IPUs to train on.
                   init_word=None,

                   limit_train_batches=1.0,                             # How much of training dataset to check (float = fraction, int = num_batches)
                   limit_val_batches=1.0,                               # How much of validation dataset to check (float = fraction, int = num_batches)
                   limit_test_batches=1.0,                              # How much of test dataset to check (float = fraction, int = num_batches)
                   limit_predict_batches=1.0,                           # How much of prediction dataset to check (float = fraction, int = num_batches)



                   logger=True,                                         # Logger (or iterable collection of loggers) for experiment tracking. A ``True`` value uses
                                                                        # the default ``TensorBoardLogger``. ``False`` will disable logging. If multiple loggers are
                                                                        # provided and the `save_dir` property of that logger is not set, local files (checkpoints,
                                                                        # profiler traces, etc.) are saved in ``default_root_dir`` rather than in the ``log_dir`` of any
                                                                        # of the individual loggers.
                   log_gpu_memory=None,                                 # 'min_max', 'all'. Might slow performance

                   log_every_n_steps=50,                                # How often to log within steps (defaults to every 50 steps).
                   logdir='output/ti/logs',
                   move_metrics_to_cpu=False,                           # Whether to force internal logged metrics to be moved to cpu.
                                                                        # This can save some gpu memory, but can make training slower. Use with attention.
                   multiple_trainloader_mode='max_size_cycle',          # How to loop over the datasets when there are multiple train loaders.
                                                                        # In 'max_size_cycle' mode, the trainer ends one epoch when the largest dataset is traversed,
                                                                        # and smaller datasets reload when running out of their data. In 'min_size' mode, all the datasets
                                                                        # reload when reaching the minimum length of datasets.
                   max_epochs=None,                                     # Stop training once this number of epochs is reached. Disabled by default (None).
                                                                        # If both max_epochs and max_steps are not specified, defaults to ``max_epochs`` = 1000.
                   min_epochs=None,                                     # Force training for at least these many epochs. Disabled by default (None).
                                                                        # If both min_epochs and min_steps are not specified, defaults to ``min_epochs`` = 1.
                   max_steps=-1,                                        # Stop training after this number of steps. Disabled by default (None).
                   min_steps=None,                                      # Force training for at least these number of steps. Disabled by default (None).
                   max_time=None,                                       # Stop training after this amount of time has passed. Disabled by default (None).
                                                                        # The time duration can be specified in the format DD:HH:MM:SS (days, hours, minutes seconds), as a
                                                                        # :class:`datetime.timedelta`, or a dictionary with keys that will be passed to
                                                                        # :class:`datetime.timedelta`.
                   name='some_job',
                   num_nodes=1,                                         # number of GPU nodes for distributed training.
                   num_processes=1,                                     # number of processes for distributed training with distributed_backend="ddp_cpu"
                   num_sanity_val_steps=None,                           # Sanity check runs n validation batches before starting the training routine.
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

                   precision=16,                                        # Double precision (64), full precision (32) or half precision (16). Can be used on CPU, GPU or
                                                                        # TPUs.

                   reload_dataloaders_every_n_epochs=0,                 # Set to a non-negative integer to reload dataloaders every n epochs.
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
                   stochastic_weight_avg=False,                         # Whether to use `Stochastic Weight Averaging (SWA)
                                                                        # <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging/>_
                   strategy=None,
                   seed=23,
                   scale_lr=False,
                   sync_batchnorm=False,                                # Synchronize batch norm layers between process groups/whole world.
                   terminate_on_nan=None,                               # If set to True, will terminate training (by raising a `ValueError`) at the
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
                   val_check_interval=1.0                               # How often to check the validation set. Use float to check within a training epoch,
                                                                        # use int to check every n steps (batches).
                   ):
        distributed_backend=None                # deprecated. Please use 'accelerator'
        reload_dataloaders_every_epoch=False    # deprecated. Please use ``reload_dataloaders_every_n_epochs``.
        truncated_bptt_steps=None               # Deprecated in v1.3 to be removed in 1.5.
                                                # Please use :paramref:`~pytorch_lightning.core.lightning.LightningModule.truncated_bptt_steps` instead.

        opt = SimpleNamespace(**locals())



        self.run_dreambooth(opt)




    def run_dreambooth(self, opt):
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # add cwd for convenience and to make classes in this file available when
        # running as `python main.py`
        # (in particular `main.DataModuleFromConfig`)
        sys.path.append(os.getcwd())

        unknown = []

        if opt.name and opt.resume:
            raise ValueError(
                "-n/--name and -r/--resume cannot be specified both."
                "If you want to resume training in a new log folder, "
                "use -n/--name in combination with --resume_from_checkpoint"
            )
        if opt.resume:
            if not os.path.exists(opt.resume):
                raise ValueError("Cannot find {}".format(opt.resume))
            if os.path.isfile(opt.resume):
                paths = opt.resume.split("/")
                # idx = len(paths)-paths[::-1].index("logs")+1
                # logdir = "/".join(paths[:idx])
                logdir = "/".join(paths[:-2])
                ckpt = opt.resume
            else:
                assert os.path.isdir(opt.resume), opt.resume
                logdir = opt.resume.rstrip("/")
                ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

            opt.resume_from_checkpoint = ckpt
            base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
            opt.base = base_configs + opt.base
            _tmp = logdir.split("/")
            nowname = _tmp[-1]
        else:
            if opt.name:
                name = "_" + opt.name
            elif opt.base:
                cfg_fname = os.path.split(opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = "_" + cfg_name
            else:
                name = ""

            if opt.datadir_in_name:
                now = os.path.basename(os.path.normpath(opt.data_root)) + now

            nowname = now + name + opt.postfix
            logdir = os.path.join(opt.logdir, nowname)

        ckptdir = os.path.join(logdir, "checkpoints")
        cfgdir = os.path.join(logdir, "configs")
        seed_everything(opt.seed)

        try:
            # init and save configs
            configs = [OmegaConf.load(cfg) for cfg in opt.base]
            cli = OmegaConf.from_dotlist(unknown)
            config = OmegaConf.merge(*configs, cli)
            lightning_config = config.pop("lightning", OmegaConf.create())
            # merge trainer cli with config
            #trainer_config = lightning_config.get("trainer", OmegaConf.create())
            trainer_config = self.get_trainer_config(opt)
            # default to ddp
            trainer_config["accelerator"] = "ddp"
            for k in nondefault_trainer_args(opt):
                trainer_config[k] = getattr(opt, k)
            if not "gpus" in trainer_config:
                del trainer_config["accelerator"]
                cpu = True
            else:
                gpuinfo = trainer_config["gpus"]
                print(f"Running on GPUs {gpuinfo}")
                cpu = False
            trainer_opt = argparse.Namespace(**trainer_config)
            lightning_config.trainer = trainer_config

            # model

            # config.model.params.personalization_config.params.init_word = opt.init_word
            # config.model.params.personalization_config.params.embedding_manager_ckpt = opt.embedding_manager_ckpt
            # config.model.params.personalization_config.params.placeholder_tokens = opt.placeholder_tokens

            # if opt.init_word:
            #     config.model.params.personalization_config.params.initializer_words[0] = opt.init_word

            config.data.params.train.params.placeholder_token = opt.class_word
            config.data.params.reg.params.placeholder_token = opt.class_word
            config.data.params.validation.params.placeholder_token = opt.class_word

            if opt.actual_resume:
                model = load_model_from_config(config, opt.actual_resume)
            else:
                model = instantiate_from_config(config.model)

            # trainer and callbacks
            trainer_kwargs = dict()

            # default logger configs
            default_logger_cfgs = {
                "wandb": {
                    "target": "pytorch_lightning.loggers.WandbLogger",
                    "params": {
                        "name": nowname,
                        "save_dir": logdir,
                        "offline": opt.debug,
                        "id": nowname,
                    }
                },
                "testtube": {
                    "target": "pytorch_lightning.loggers.TestTubeLogger",
                    "params": {
                        "name": "testtube",
                        "save_dir": logdir,
                    }
                },
            }
            default_logger_cfg = default_logger_cfgs["testtube"]
            if "logger" in lightning_config:
                logger_cfg = lightning_config.logger
            else:
                logger_cfg = OmegaConf.create()
            logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
            trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

            # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
            # specify which metric is used to determine best models
            default_modelckpt_cfg = {
                "target": "pytorch_lightning.callbacks.ModelCheckpoint",
                "params": {
                    "dirpath": ckptdir,
                    "filename": "{epoch:06}",
                    "verbose": True,
                    "save_last": True,
                }
            }
            if hasattr(model, "monitor"):
                print(f"Monitoring {model.monitor} as checkpoint metric.")
                default_modelckpt_cfg["params"]["monitor"] = model.monitor
                default_modelckpt_cfg["params"]["save_top_k"] = 1

            if "modelcheckpoint" in lightning_config:
                modelckpt_cfg = lightning_config.modelcheckpoint
            else:
                modelckpt_cfg =  OmegaConf.create()
            modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
            print(f"Merged modelckpt-cfg: \n{modelckpt_cfg}")
            if version.parse(pl.__version__) < version.parse('1.4.0'):
                trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

            # add callback which sets up log directory
            default_callbacks_cfg = {
                "setup_callback": {
                    "target": "plugins.training.callbacks.SetupCallback",
                    "params": {
                        "resume": opt.resume,
                        "now": now,
                        "logdir": logdir,
                        "ckptdir": ckptdir,
                        "cfgdir": cfgdir,
                        "config": config,
                        "lightning_config": lightning_config,
                    }
                },
                "image_logger": {
                    "target": "plugins.training.callbacks.ImageLogger",
                    "params": {
                        "batch_frequency": 750,
                        "max_images": 4,
                        "clamp": True
                    }
                },
                "learning_rate_logger": {
                    "target": "plugins.training.callbacks.LearningRateMonitor",
                    "params": {
                        "logging_interval": "step",
                        # "log_momentum": True
                    }
                },
                "cuda_callback": {
                    "target": "plugins.training.callbacks.CUDACallback"
                },
            }
            if version.parse(pl.__version__) >= version.parse('1.4.0'):
                default_callbacks_cfg.update({'checkpoint_callback': modelckpt_cfg})

            if "callbacks" in lightning_config:
                callbacks_cfg = lightning_config.callbacks
            else:
                callbacks_cfg = OmegaConf.create()

            if 'metrics_over_trainsteps_checkpoint' in callbacks_cfg:
                print(
                    'Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
                default_metrics_over_trainsteps_ckpt_dict = {
                    'metrics_over_trainsteps_checkpoint':
                        {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                         'params': {
                             "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                             "filename": "{epoch:06}-{step:09}",
                             "verbose": True,
                             'save_top_k': -1,
                             'every_n_train_steps': 10000,
                             'save_weights_only': True
                         }
                         }
                }
                default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

            callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
            if 'ignore_keys_callback' in callbacks_cfg and hasattr(trainer_opt, 'resume_from_checkpoint'):
                callbacks_cfg.ignore_keys_callback.params['ckpt_path'] = trainer_opt.resume_from_checkpoint
            elif 'ignore_keys_callback' in callbacks_cfg:
                del callbacks_cfg['ignore_keys_callback']

            trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
            trainer_kwargs["max_steps"] = trainer_opt.max_steps

            self.trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
            self.trainer.logdir = logdir  ###

            # data
            config.data.params.train.params.data_root = opt.data_root
            config.data.params.reg.params.data_root = opt.reg_data_root
            config.data.params.validation.params.data_root = opt.data_root
            #data = instantiate_from_config(config.data)

            print(config.data)
            data = instantiate_from_config(config.data)
            # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
            # calling these ourselves should not be necessary but it is.
            # lightning still takes care of proper multiprocessing though
            data.prepare_data()
            data.setup()
            print("#### Data #####")
            for k in data.datasets:
                print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

            # configure learning rate
            bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
            if not cpu:
                ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
            else:
                ngpu = 1
            if 'accumulate_grad_batches' in lightning_config.trainer:
                accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
            else:
                accumulate_grad_batches = 1
            print(f"accumulate_grad_batches = {accumulate_grad_batches}")
            lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
            if opt.scale_lr:
                model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
                print(
                    "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
                        model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
            else:
                model.learning_rate = base_lr
                print("++++ NOT USING LR SCALING ++++")
                print(f"Setting learning rate to {model.learning_rate:.2e}")


            # allow checkpointing via USR1
            def melk(*args, **kwargs):
                # run all checkpoint hooks
                if self.trainer.global_rank == 0:
                    print("Summoning checkpoint.")
                    ckpt_path = os.path.join(ckptdir, "last.ckpt")
                    self.trainer.save_checkpoint(ckpt_path)


            def divein(*args, **kwargs):
                if self.trainer.global_rank == 0:
                    import pudb;
                    pudb.set_trace()


            # run
            if opt.train:
                try:
                    self.trainer.fit(model, data)
                except Exception as e:
                    print('exception ', e)
                    melk()
                    raise
            if not opt.no_test and not self.trainer.interrupted:
                self.trainer.test(model, data)
        except Exception as e:
            print('exception ', e)
            if opt.debug and self.trainer.global_rank == 0:
                try:
                    import pudb as debugger
                except ImportError:
                    import pdb as debugger
                debugger.post_mortem()
            raise
        finally:
            # move newly created debug project to debug_runs
            if opt.debug and not opt.resume and self.trainer.global_rank == 0:
                dst, name = os.path.split(logdir)
                dst = os.path.join(dst, "debug_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                os.rename(logdir, dst)
            if self.trainer.global_rank == 0:
                print(self.trainer.profiler.summary())
