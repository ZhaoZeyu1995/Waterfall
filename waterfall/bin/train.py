#!/usr/bin/env python3

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
import argparse
import yaml
import random
import numpy as np
import logging
from waterfall import models
from waterfall.utils import datapipe, datapipe_manual_ctc, datapipe_k2
from waterfall.manual_ctc import eta_scheduler
import wandb


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.loader.SafeLoader)
    pl.seed_everything(cfg['seed'], workers=True)

    batch_size = cfg['batch_size'] if args.batch_size == 0 else args.batch_size

    if cfg['loss'] == 'ctc':
        Dataset = datapipe.Dataset
        collate_fn = datapipe.collate_fn
    elif cfg['loss'] in ['ctc_fb', 'ctc_softmax']:
        Dataset = datapipe_manual_ctc.Dataset
        collate_fn = datapipe_manual_ctc.collate_fn
    elif cfg['loss'] in ['ctc_k2', 'k2']:
        Dataset = datapipe_k2.Dataset
        collate_fn = datapipe_k2.collate_fn_sorted

    if cfg['loss'] in ['ctc_k2', 'k2']:
        train_data = Dataset(args.train_set,
                             args.lang_dir, token_type='phones')
        dev_data = Dataset(args.dev_set,
                           args.lang_dir, token_type='phones')
    else:
        train_data = Dataset(args.train_set,
                             args.lang_dir)
        dev_data = Dataset(args.dev_set,
                           args.lang_dir)

    train_gen = DataLoader(train_data,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=cfg['num_workers'],
                           persistent_workers=True,
                           collate_fn=collate_fn)
    dev_gen = DataLoader(dev_data,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=cfg['num_workers'],
                         persistent_workers=True,
                         collate_fn=collate_fn)

    model = models.Wav2VecFineTuningDiverse(
        train_data.lang.num_nn_output, cfg=cfg, lang_dir=args.lang_dir)

    os.makedirs('exp/%s' % (args.name), exist_ok=True)
    model_checkpoint= pl.callbacks.ModelCheckpoint(monitor='valid_loss',
                                              save_top_k=1 if 'save_top_k' not in cfg.keys(
                                              ) else cfg['save_top_k'],
                                              every_n_epochs=1,
                                              filename='{epoch}-{valid_loss:.3f}',
                                              mode='min')
    callbacks = [model_checkpoint,
                 pl.callbacks.RichProgressBar(),
                 pl.callbacks.RichModelSummary(max_depth=2)]

    if cfg['early_stopping']:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='valid_loss',
                                                    mode='min',
                                                    patience=cfg['patience'],
                                                    verbose=True))

    if 'auto_eta_scheduler' in cfg.keys() and cfg['auto_eta_scheduler']:
        callbacks.append(eta_scheduler.AutoEtaScheduler('valid_loss',
                                                        delta_eta=cfg['delta_eta'],
                                                        final_eta=cfg['final_eta'],
                                                        patience=cfg['patience_eta'],
                                                        verbose=True))

    accumulate_grad_batches = 1 # by default 1, args.accumulate_grad_batches has more priority than cfg['accumulate_grad_batches']
    if args.accumulate_grad_batches != 1:
        accumulate_grad_batches = args.accumulate_grad_batches
    elif 'accumulate_grad_batches' in cfg.keys():
        accumulate_grad_batches = cfg['accumulate_grad_batches']

    logger = pl.loggers.WandbLogger(
        project=args.name, save_dir='exp/%s' % (args.name))
    logger.watch(model, log='all')

    if args.checkpoint:
        if not args.load_weights_only:
            trainer = pl.Trainer(gpus=args.gpus,
                                 strategy=cfg['strategy'],
                                 deterministic=False,
                                 resume_from_checkpoint=args.checkpoint,
                                 max_epochs=cfg['max_epochs'],
                                 logger=logger,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 callbacks=callbacks)
        else:
            model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
            trainer = pl.Trainer(gpus=args.gpus,
                                 strategy=cfg['strategy'],
                                 deterministic=False,
                                 max_epochs=cfg['max_epochs'],
                                 logger=logger,
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 callbacks=callbacks)
    else:
        trainer = pl.Trainer(gpus=args.gpus,
                             strategy=cfg['strategy'],
                             deterministic=False,
                             max_epochs=cfg['max_epochs'],
                             logger=logger,
                             accumulate_grad_batches=accumulate_grad_batches,
                             callbacks=callbacks)

    trainer.fit(model, train_gen, dev_gen)

    logger.log_metrics({'best_model_path': os.path.join(os.getcwd(), model_checkpoint.best_model_path),
                        'best_model_loss': model_checkpoint.best_model_score.item()})
    wandb.finish()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train_set', help='Training set directory.', type=str)
    parser.add_argument('--dev_set', help='Dev set directory.', type=str)
    parser.add_argument('--lang_dir', help='Lang directory.', type=str)
    parser.add_argument('--config', help='Configuration file path.', type=str)
    parser.add_argument(
        '--name', help='Experiment name. Models will be stored in exp/$name/version*', type=str, default='ctc')
    parser.add_argument(
        '--gpus', help='Number of GPUs that used for training.', type=int, default=1)
    parser.add_argument(
        '--checkpoint', help='Resume from checkpoint.', type=str, default=None)
    parser.add_argument('--load_weights_only',
                        help='Whether or not load weights only from checkpoint.', type=bool, default=False)
    parser.add_argument('--batch_size', help='The batch_size for training.', type=int, default=0)
    parser.add_argument(
        '--accumulate_grad_batches', help='The number of batches for gradient accumulation.', type=int, default=1)

    args = parser.parse_args()
    main(args)
