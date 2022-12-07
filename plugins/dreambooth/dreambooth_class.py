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

from plugins.dreambooth.callbacks import *
from plugins.dreambooth.ldm_db.data.base import Txt2ImgIterableBaseDataset
from plugins.dreambooth.ldm_db.util import instantiate_from_config
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


    def dreambooth(self,
                   name='some_job',
                   resume='',
                   base=['plugins/dreambooth/configs/v1-finetune_unfrozen.yaml'],
                   train=True,
                   no_test=False,
                   project=None,
                   debug=False,
                   seed=23,
                   postfix='',
                   logdir='output/ti/logs',
                   scale_lr=False,
                   datadir_in_name=True,
                   actual_resume='data/models/sd-v1-4.ckpt',
                   data_root='data/input/testTraining/dog',
                   reg_data_root='data/input/regularization/images',
                   embedding_manager_ckpt='',
                   class_word='<xxx>',
                   init_word=None,
                   logger=True,
                   checkpoint_callback=True,
                   default_root_dir=None,
                   gradient_clip_val=0.0,
                   gradient_clip_algorithm='norm',
                   process_position=0,
                   num_nodes=1,
                   num_processes=1,
                   devices=None,
                   gpus='0,',
                   auto_select_gpus=False,
                   tpu_cores=None,
                   ipus=None,
                   log_gpu_memory=None,
                   progress_bar_refresh_rate=None,
                   overfit_batches=0.0,
                   track_grad_norm=-1,
                   check_val_every_n_epoch=1,
                   fast_dev_run=False,
                   accumulate_grad_batches=1,
                   max_epochs=None,
                   min_epochs=None,
                   max_steps=None,
                   min_steps=None,
                   max_time=None,
                   limit_train_batches=1.0,
                   limit_val_batches=1.0,
                   limit_test_batches=1.0,
                   limit_predict_batches=1.0,
                   val_check_interval=1.0,
                   flush_logs_every_n_steps=100,
                   log_every_n_steps=50,
                   accelerator=None,
                   sync_batchnorm=False,
                   precision=32,
                   weights_summary='top',
                   weights_save_path=None,
                   num_sanity_val_steps=2,
                   truncated_bptt_steps=None,
                   resume_from_checkpoint=None,
                   profiler=None,
                   benchmark=False,
                   deterministic=False,
                   reload_dataloaders_every_n_epochs=0,
                   reload_dataloaders_every_epoch=False,
                   auto_lr_find=False,
                   replace_sampler_ddp=True,
                   terminate_on_nan=False,
                   auto_scale_batch_size=False,
                   prepare_data_per_node=True,
                   plugins=None,
                   amp_backend='native',
                   amp_level='O2',
                   distributed_backend=None,
                   move_metrics_to_cpu=False,
                   multiple_trainloader_mode='max_size_cycle',
                   stochastic_weight_avg=False,
                   progress_callback=None):
        opt=SimpleNamespace(**locals())
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
            trainer_config = lightning_config.get("trainer", OmegaConf.create())
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
                    "target": "plugins.dreambooth.callbacks.SetupCallback",
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
                    "target": "plugins.dreambooth.callbacks.ImageLogger",
                    "params": {
                        "batch_frequency": 750,
                        "max_images": 4,
                        "clamp": True
                    }
                },
                "learning_rate_logger": {
                    "target": "plugins.dreambooth.callbacks.LearningRateMonitor",
                    "params": {
                        "logging_interval": "step",
                        # "log_momentum": True
                    }
                },
                "cuda_callback": {
                    "target": "plugins.dreambooth.callbacks.CUDACallback"
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
