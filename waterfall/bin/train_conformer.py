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
from waterfall import conformer
from waterfall.utils import datapipe, datapipe_manual_ctc, datapipe_k2
from waterfall.manual_ctc import eta_scheduler
from waterfall.utils.specaug import SpecAugment


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.loader.SafeLoader)
    pl.seed_everything(cfg['seed'], workers=True)

    batch_size = cfg['batch_size'] if args.batch_size == 0 else args.batch_size

    if cfg['spec_aug']:
        spec_aug = SpecAugment(resize_mode=cfg['mode'],
                               max_time_warp=cfg['max_time_warp'],
                               max_freq_width=cfg['max_freq_width'],
                               n_freq_mask=cfg['n_freq_mask'],
                               max_time_width=cfg['max_time_width'],
                               n_time_mask=cfg['n_time_mask'],
                               inplace=cfg['inplace'],
                               replace_with_zero=cfg['replace_with_zero'])
    else:
        spec_aug = None

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
                             args.lang_dir, token_type='phones', load_wav=False, load_feats=True, transforms=spec_aug)
        dev_data = Dataset(args.dev_set,
                           args.lang_dir, token_type='phones', load_wav=False, load_feats=True)
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

    if 'nowarmup' in cfg and cfg['nowarmup']:
        model = conformer.ConformerModelNoWarmup(
            cfg['idim'], train_data.lang.num_nn_output, cfg=cfg, lang_dir=args.lang_dir)
    else:
        model = conformer.ConformerModel(
            cfg['idim'], train_data.lang.num_nn_output, cfg=cfg, lang_dir=args.lang_dir)

    callbacks = [pl.callbacks.ModelCheckpoint(monitor='valid_loss',
                                              save_top_k=5 if 'save_top_k' not in cfg.keys(
                                              ) else cfg['save_top_k'],
                                              every_n_epochs=1,
                                              filename='{epoch}-{valid_loss:.3f}',
                                              mode='min')]

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

    accumulate_grad_batches = 1 if 'accumulate_grad_batches' not in cfg.keys(
    ) else cfg['accumulate_grad_batches']
    if args.checkpoint:
        if not args.load_weights_only:
            trainer = pl.Trainer(gpus=args.gpus,
                                 strategy=cfg['strategy'],
                                 deterministic=False,
                                 resume_from_checkpoint=args.checkpoint,
                                 max_epochs=cfg['max_epochs'],
                                 logger=pl.loggers.TensorBoardLogger(
                                     'exp', name=args.name),
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 gradient_clip_val=cfg['grad-clip'],
                                 gradient_clip_algorithm='norm',
                                 callbacks=callbacks)
        else:
            model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
            trainer = pl.Trainer(gpus=args.gpus,
                                 strategy=cfg['strategy'],
                                 deterministic=False,
                                 max_epochs=cfg['max_epochs'],
                                 logger=pl.loggers.TensorBoardLogger(
                                     'exp', name=args.name),
                                 accumulate_grad_batches=accumulate_grad_batches,
                                 gradient_clip_val=cfg['grad-clip'],
                                 gradient_clip_algorithm='norm',
                                 callbacks=callbacks)
    else:
        trainer = pl.Trainer(gpus=args.gpus,
                             strategy=cfg['strategy'],
                             deterministic=False,
                             max_epochs=cfg['max_epochs'],
                             logger=pl.loggers.TensorBoardLogger(
                                 'exp', name=args.name),
                             accumulate_grad_batches=accumulate_grad_batches,
                             gradient_clip_val=cfg['grad-clip'],
                             gradient_clip_algorithm='norm',
                             callbacks=callbacks)

    trainer.fit(model, train_gen, dev_gen)


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

    args = parser.parse_args()
    main(args)
