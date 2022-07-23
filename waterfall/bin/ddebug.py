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


class DebuggerModel(models.Wav2VecFineTuningDiverse):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        for para in self.wav2vec.parameters():
            para.requires_grad = False
        for i in range(1, self.cfg['finetune_layers']+1):
            for para in self.wav2vec.encoder.transformer.layers[-i].parameters():
                para.requires_grad = True

    def division(self, batch, batch_idx=None, optimizer_idx=None):
        if self.cfg['loss'] == 'ctc':
            wavs = batch['wavs']
            lengths = batch['lengths']
            target_lengths = batch['target_lengths']
            targets = batch['targets']
            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            print('\nxlens\n', xlens)
            print('\ntarget_lengths\n', target_lengths)

            loss = F.ctc_loss(log_probs=log_probs.permute(1, 0, 2),
                              input_lengths=xlens,
                              target_lengths=target_lengths,
                              targets=targets,
                              reduction='sum')
            print('\nloss\n', loss)
        elif self.cfg['loss'] == 'ctc_fb':
            wavs = batch['wavs']
            lengths = batch['lengths']
            trans = batch['trans']
            trans_lengths = batch['trans_lengths']
            transform = batch['transform']
            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            loss = ctc.ctc_loss(log_probs,
                                input_lengths=xlens,
                                trans=trans,
                                trans_lengths=trans_lengths,
                                transform=transform,
                                reduction='sum')
        elif self.cfg['loss'] == 'ctc_softmax':
            wavs = batch['wavs']
            lengths = batch['lengths']
            trans = batch['trans']
            trans_lengths = batch['trans_lengths']
            transform = batch['transform']
            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            loss = softmax_ctc.ctc_softmax(log_probs,
                                           input_lengths=xlens,
                                           trans=trans,
                                           trans_lengths=trans_lengths,
                                           transform=transform,
                                           eta=self.eta,
                                           reduction='sum')
        elif self.cfg['loss'] in ['ctc_k2']:
            wavs = batch['wavs']
            lengths = batch['lengths']
            targets = batch['targets']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=log_probs,
                                           supervision_segments=supervision_segments)

            trans_fsa = k2.linear_fsa(targets, device=log_probs.device)

            trans_fsa = k2.remove_epsilon_and_add_self_loops(trans_fsa)

            trans_fsa = k2.arc_sort(trans_fsa)

            topo = self.lang.topo.to(log_probs.device)

            decoding_graph = k2.compose(
                topo, trans_fsa, treat_epsilons_specially=False)

            assert decoding_graph.requires_grad == False

            loss = graph.graphloss(decoding_graph=decoding_graph,
                                   dense_fsa_vec=dense_fsa_vec,
                                   output_beam=self.cfg['output_beam'],
                                   reduction='sum')
        elif self.cfg['loss'] in ['k2']:
            wavs = batch['wavs']
            lengths = batch['lengths']
            targets = batch['targets']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=log_probs,
                                           supervision_segments=supervision_segments)

            trans_fsa = k2.linear_fsa(targets, device=log_probs.device)

            trans_fsa = k2.remove_epsilon_and_add_self_loops(trans_fsa)

            trans_fsa = k2.arc_sort(trans_fsa)

            topo = self.lang.topo.to(log_probs.device)

            decoding_graph = k2.compose(
                topo, trans_fsa, treat_epsilons_specially=False)

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(decoding_graph=decoding_graph,
                                        dense_fsa_vec=dense_fsa_vec,
                                        output_beam=self.cfg['output_beam'],
                                        reduction='sum')
            denominator = graph.graphloss(decoding_graph=topo,
                                          dense_fsa_vec=dense_fsa_vec,
                                          output_beam=self.cfg['output_beam'],
                                          reduction='sum')
            loss = numerator - denominator
        return loss/batch_num

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self.division(batch, batch_idx, optimizer_idx)
        self.log('loss', loss, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        if 'same_optimiser_for_finetuning' in self.cfg.keys() and self.cfg['same_optimiser_for_finetuning']:
            optimiser = torch.optim.Adam(self.parameters())
            if self.cfg['loss'] == 'ctc_softmax' and 'auto_eta_scheduler' in self.cfg.keys() and self.cfg['auto_eta_scheduler']:
                return [optimiser], [{'scheduler': eta_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=self.cfg['patience_eta'], verbose=True, factor=0.2),
                                      'monitor': 'valid_loss'}]
            return [optimiser], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True),
                                  'monitor': 'valid_loss'}]
        else:
            optimiser = torch.optim.Adam(self.wav2vec.parameters(), lr=1e-4)
            optimiser_output = torch.optim.Adam(self.output_layer.parameters())
            if self.cfg['loss'] == 'ctc_softmax' and 'auto_eta_scheduler' in self.cfg.keys():
                if self.cfg['auto_eta_scheduler']:
                    return [optimiser, optimiser_output], [{'scheduler': eta_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=self.cfg['patience_eta'], verbose=True, factor=0.2),
                                                            'monitor': 'valid_loss'},
                                                           {'scheduler': eta_scheduler.ReduceLROnPlateau(optimiser_output, 'min', patience=self.cfg['patience_eta'], verbose=True, factor=0.2),
                                                            'monitor': 'valid_loss'}]
            return [optimiser, optimiser_output], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True),
                                                   'monitor': 'valid_loss'},
                                                   {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_output, 'min', patience=2, verbose=True),
                                                   'monitor': 'valid_loss'}]


def main(args):
    cfg = yaml.load(open(args.config), Loader=yaml.loader.SafeLoader)
    pl.seed_everything(cfg['seed'], workers=True)

    if cfg['loss'] == 'ctc':
        Dataset = datapipe.Dataset
        collate_fn = datapipe.collate_fn
    elif cfg['loss'] in ['ctc_fb', 'ctc_softmax']:
        Dataset = datapipe_manual_ctc.Dataset
        collate_fn = datapipe_manual_ctc.collate_fn
    elif cfg['loss'] in ['ctc_k2', 'k2']:
        Dataset = datapipe_k2.Dataset
        collate_fn = datapipe_k2.collate_fn_sorted

    train_data = Dataset(args.train_set,
                         args.lang_dir)
    dev_data = Dataset(args.dev_set,
                       args.lang_dir)

    train_gen = DataLoader(train_data,
                           batch_size=cfg['batch_size'],
                           shuffle=True,
                           num_workers=cfg['num_workers'],
                           persistent_workers=True,
                           collate_fn=collate_fn)
    dev_gen = DataLoader(dev_data,
                         batch_size=cfg['batch_size'],
                         shuffle=False,
                         num_workers=cfg['num_workers'],
                         persistent_workers=True,
                         collate_fn=collate_fn)

    model = DebuggerModel(
        train_data.lang.num_nn_output, cfg=cfg, lang_dir=args.lang_dir)
    callbacks = [pl.callbacks.ModelCheckpoint(monitor='valid_loss',
                                              save_top_k=5,
                                              every_n_epochs=1,
                                              filename='{epoch}-{valid_loss:.3f}',
                                              mode='min')]

    if cfg['early_stopping']:
        callbacks.append(pl.callbacks.EarlyStopping(monitor='valid_loss',
                                                    mode='min',
                                                    patience=cfg['patience'],
                                                    verbose=True))

    if 'auto_eta_scheduler' in cfg.keys():
        if cfg['auto_eta_scheduler']:
            callbacks.append(eta_scheduler.AutoEtaScheduler('valid_loss',
                                                            delta_eta=cfg['delta_eta'],
                                                            final_eta=cfg['final_eta'],
                                                            patience=cfg['patience_eta'],
                                                            verbose=True))

    trainer = pl.Trainer(gpus=args.gpus,
                         strategy=cfg['strategy'],
                         deterministic=False,
                         resume_from_checkpoint=args.checkpoint,
                         max_epochs=cfg['max_epochs'],
                         logger=pl.loggers.TensorBoardLogger(
                             'exp', name=args.name),
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

    args = parser.parse_args()
    main(args)
