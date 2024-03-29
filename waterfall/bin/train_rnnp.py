#!/usr/bin/env python3

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import argparse
import yaml
import random
import numpy as np
import logging
from waterfall import rnnp
from waterfall.utils import datapipe
from waterfall.utils.specaug import SpecAugment
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'conf'), config_name="config")
def main(cfg):
    pl.seed_everything(cfg.training.seed, workers=True)

    batch_size = cfg.training.batch_size

    if cfg.training.spec_aug:
        spec_aug = SpecAugment(resize_mode=cfg.specaug.mode,
                               max_time_warp=cfg.specaug.max_time_warp,
                               max_freq_width=cfg.specaug.max_freq_width,
                               n_freq_mask=cfg.specaug.n_freq_mask,
                               max_time_width=cfg.specaug.max_time_width,
                               n_time_mask=cfg.specaug.n_time_mask,
                               inplace=cfg.specaug.inplace,
                               replace_with_zero=cfg.specaug.replace_with_zero)
    else:
        spec_aug = None

    ctc_target = False
    if cfg.training.loss == 'builtin_ctc':
        ctc_target = True

    train_data = datapipe.Dataset(to_absolute_path(cfg.data.train_set),
                                  to_absolute_path(cfg.data.lang_dir),
                                  ctc_target=ctc_target,
                                  load_feats=True,
                                  transforms=spec_aug)
    dev_data = datapipe.Dataset(to_absolute_path(cfg.data.dev_set),
                                to_absolute_path(cfg.data.lang_dir),
                                ctc_target=ctc_target,
                                load_feats=True,
                                transforms=None)

    train_gen = DataLoader(train_data,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=cfg.training.num_workers,
                           persistent_workers=True,
                           collate_fn=datapipe.collate_fn_sorted)
    dev_gen = DataLoader(dev_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=cfg.training.num_workers,
                         persistent_workers=True,
                         collate_fn=datapipe.collate_fn_sorted)
    model = rnnp.RNNPModel(
        cfg.model.idim, train_data.lang.num_nn_output, cfg=cfg, lang_dir=cfg.data.lang_dir)

    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='valid_loss',
                                                    save_top_k=1 if "save_top_k" not in cfg.training.keys() else cfg.training.save_top_k,
                                                    every_n_epochs=1,
                                                    dirpath=os.path.join(
                                                        cfg.training.output_dir, 'checkpoints'),
                                                    filename='{epoch}-{valid_loss:.4f}',
                                                    mode='min')

    callbacks = [model_checkpoint,
                 pl.callbacks.LearningRateMonitor(logging_interval='step'),
                 pl.callbacks.RichProgressBar(),
                 pl.callbacks.RichModelSummary(max_depth=2)]

    if cfg.training.early_stopping:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='valid_loss',
                                                    mode='min',
                                                    patience=cfg.training.patience,
                                                    verbose=False))

    accumulate_grad_batches = 1 if "accumulate_grad_batches" not in cfg.training.keys() else cfg.training.accumulate_grad_batches
    logger = pl.loggers.WandbLogger(
        project='waterfall-%s-%s' % (os.path.basename(os.path.dirname(get_original_cwd())),
                                     os.path.basename(get_original_cwd())),
        name=cfg.training.name,
        save_dir=os.path.join(cfg.training.output_dir))
    logger.watch(model, log='all', log_graph=False)

    if 'checkpoint' in cfg.training.keys() and cfg.training.checkpoint is not None:
        if 'load_weights_only' in cfg.training.keys() and not cfg.training.load_weights_only:
            trainer = pl.Trainer(devices=cfg.training.gpus,
                                 strategy=cfg.training.strategy,
                                 deterministic=False,
                                 resume_from_checkpoint=cfg.training.checkpoint,
                                 max_epochs=cfg.training.max_epochs,
                                 logger=logger,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 callbacks=callbacks,
                                 sync_batchnorm=True,
                                 val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
                                 )
        else:
            checkpoint = torch.load(
                cfg.training.checkpoint, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['state_dict'])
            del checkpoint
            torch.cuda.empty_cache()
            trainer = pl.Trainer(devices=cfg.training.gpus,
                                 strategy=cfg.training.strategy,
                                 deterministic=False,
                                 max_epochs=cfg.training.max_epochs,
                                 logger=logger,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 callbacks=callbacks,
                                 sync_batchnorm=True,
                                 val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
                                 )
    else:
        trainer = pl.Trainer(devices=cfg.training.gpus,
                             strategy=cfg.training.strategy,
                             deterministic=False,
                             max_epochs=cfg.training.max_epochs,
                             logger=logger,
                             accumulate_grad_batches=accumulate_grad_batches,
                             callbacks=callbacks,
                             sync_batchnorm=True,
                             val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
                             )

    trainer.fit(model, train_gen, dev_gen)

    logger.log_metrics({'best_model_path': os.path.join(os.getcwd(), model_checkpoint.best_model_path),
                        'best_model_loss': model_checkpoint.best_model_score.item()})

    wandb.finish()


if __name__ == '__main__':
    main()
