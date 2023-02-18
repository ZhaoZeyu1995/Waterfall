import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
from waterfall import graph
import k2
from waterfall.utils.datapipe import Lang
import time

from espnet.nets.pytorch_backend.rnn.encoders import Encoder


class RNNPModel(pl.LightningModule):
    def __init__(self,
                 input_dim,
                 output_dim,
                 lang_dir=None,
                 cfg=None):

        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.save_hyperparameters()

        self.encoder = Encoder(
            etype=cfg['etype'],
            idim=input_dim,
            elayers=cfg['elayers'],
            eunits=cfg['eunits'],
            eprojs=cfg['eprojs'],
            subsample=cfg['subsample'],
            dropout=cfg['dropout']
        )

        self.batch_norm = nn.BatchNorm1d(cfg['eprojs'])

        self.output_layer = nn.Linear(cfg['eprojs'], self.output_dim)

        self.lang = Lang(lang_dir, load_topo=True, load_lexicon=True)

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):

        if self.cfg['loss'] in ['k2']:
            wavs = batch['feats']
            lengths = batch['feats_lens']
            word_ids = batch['word_ids']
            target_lengths = batch['target_lengths']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=log_probs,
                                           supervision_segments=supervision_segments)

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device)

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(decoding_graph=decoding_graph,
                                        dense_fsa_vec=dense_fsa_vec,
                                        target_lengths=target_lengths,
                                        reduction='mean')

            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            elif 'no_den_grad' in self.cfg.keys() and self.cfg['no_den_grad']:
                with torch.no_grad():
                    den_decoding_graph = k2.create_fsa_vec(
                        [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                    assert den_decoding_graph.requires_grad == False

                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  target_lengths=target_lengths,
                                                  reduction='mean')
                loss = numerator - denominator
            else:

                den_decoding_graph = k2.create_fsa_vec(
                    [self.lang.topo.to(log_probs.device) for _ in range(batch_num)])

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                              dense_fsa_vec=dense_fsa_vec,
                                              target_lengths=target_lengths,
                                              reduction='mean')
                loss = numerator - denominator
        elif self.cfg['loss'] == 'builtin_ctc':
            wavs = batch['feats']
            lengths = batch['feats_lens']
            target_lengths = batch['target_lengths']
            targets_ctc = batch['targets_ctc']
            log_probs, xlens = self(wavs, lengths)
            loss = F.ctc_loss(log_probs.permute(1, 0, 2), targets_ctc, xlens,
                              target_lengths, reduction='mean')

        else:
            raise Exception('Unrecognised Loss Function %s' %
                            (self.cfg['loss']))

        return loss

    def forward(self, x, xlens):
        x, xlens, _ = self.encoder(x, xlens)
        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self.compute_loss(batch, batch_idx, optimizer_idx)
        self.log('loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('valid_loss', loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        feats = batch['feats']
        feats_lens = batch['feats_lens']
        targets = batch['targets']
        names = batch['names']
        spks = batch['spks']
        texts = batch['texts']
        log_probs, xlens = self(feats, feats_lens)
        return log_probs, xlens, names, spks, texts

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=float(self.cfg['lr']))
        return [optimiser], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,
                                                                                      'min',
                                                                                      patience=self.cfg['lr_patience'],
                                                                                      verbose=True,
                                                                                      min_lr=float(self.cfg['min_lr']),
                                                                                      factor=self.cfg['factor']),
                              'monitor': 'valid_loss'}]
