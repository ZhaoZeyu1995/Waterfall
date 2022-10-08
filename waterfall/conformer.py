import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
import torchaudio
from waterfall import graph
from waterfall.manual_ctc import ctc, prior, softmax_ctc, eta_scheduler
import k2
from waterfall.utils.datapipe import Lang
import time
from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask


class ConformerModel(pl.LightningModule):
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
            idim=input_dim,
            attention_dim=cfg['adim'],
            attention_heads=cfg['aheads'],
            linear_units=cfg['eunits'],
            num_blocks=cfg['elayers'],
            input_layer=cfg['transformer-input-layer'],
            dropout_rate=cfg['dropout-rate'],
            positional_dropout_rate=cfg['dropout-rate'],
            attention_dropout_rate=cfg['transformer-attn-dropout-rate'],
            pos_enc_layer_type=cfg['transformer-encoder-pos-enc-layer-type'],
            selfattention_layer_type=cfg['transformer-encoder-selfattn-layer-type'],
            activation_type=cfg['transformer-encoder-activation-type'],
            macaron_style=cfg['macaron-style'],
            use_cnn_module=cfg['use-cnn-module'],
            zero_triu=False if 'zero-triu' not in cfg.keys(
            ) else cfg['zero-triu'],
            cnn_module_kernel=cfg['cnn-module-kernel'],
            stochastic_depth_rate=0.0 if 'stochastic-depth-rate' not in cfg.keys() else cfg['stochastic-depth-rate'])

        # self.freese_and_init() # This is legacy of wav2vec 2.0

        self.output_layer = nn.Linear(cfg['adim'], self.output_dim)

        self.lang = Lang(lang_dir, load_topo=True,
                         load_lexicon=True, load_den_graph=True)

    def freese_and_init(self):
        # TODO
        pass

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):

        if self.cfg['loss'] in ['k2']:
            wavs = batch['feats']
            lengths = batch['feats_lens']
            word_ids = batch['word_ids']

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
                                        output_beam=self.cfg['output_beam'],
                                        reduction='sum')
            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            else:
                if 'den_with_lexicon' in self.cfg.keys() and self.cfg['den_with_lexicon']:
                    den_decoding_graph = self.lang.den_graph.to(
                        log_probs.device)
                else:
                    den_decoding_graph = self.lang.topo.to(log_probs.device)

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                              dense_fsa_vec=dense_fsa_vec,
                                              output_beam=self.cfg['output_beam'] if 'output_beam_den' not in self.cfg.keys(
                                              ) else self.cfg['output_beam_den'],
                                              reduction='sum')
                loss = numerator - denominator

        return loss/batch_num

    def forward(self, x, xlens):
        src_mask = make_non_pad_mask(xlens.tolist()).to(
            x.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(x, src_mask)
        x = self.output_layer(hs_pad)
        x = F.log_softmax(x, dim=-1)
        return x, torch.sum(hs_mask, dim=-1, dtype=torch.long)

    def debugging(self):
        norm_sum_4 = 0.
        norm_sum_3 = 0.
        norm_sum_2 = 0.
        norm_sum_1 = 0.
        norm_sum_lin = 0.
        for para in self.wav2vec.encoder.transformer.layers[-4].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_4 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-3].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_3 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-2].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_2 += torch.norm(para.grad).item()
        for para in self.wav2vec.encoder.transformer.layers[-1].parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_1 += torch.norm(para.grad).item()
        for para in self.output_layer.parameters():
            # print('para', para)
            if para.grad is not None:
                # print('para.grad', para.grad)
                # print('para.grad.max()', para.grad.max())
                # print('para.grad.min()', para.grad.min())
                norm_sum_lin += torch.norm(para.grad).item()
        print('norm_sum_4', norm_sum_4)
        print('norm_sum_3', norm_sum_3)
        print('norm_sum_2', norm_sum_2)
        print('norm_sum_1', norm_sum_1)
        print('norm_sum_lin', norm_sum_lin)

        # print('torch.max(para.grad)', torch.max(para.grad))
        # print('torch.min(para.grad)', torch.min(para.grad))

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
        # optimiser = get_std_opt(
            # self.parameters(),
            # self.cfg['adim'],
            # self.cfg['transformer-warmup-steps'],
            # self.cfg['transformer-lr'],
        # )

        optimiser= torch.optim.Adam(self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
        return [optimiser], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True),
                              'monitor': 'valid_loss'}]

    def optimizer_step(self,
                       epoch,
                       batch_idx,
                       optimizer,
                       optimizer_idx,
                       optimizer_closure,
                       on_tpu=False,
                       using_native_amp=False,
                       using_lbfgs=False):
        optimizer.step(closure=optimizer_closure)

        lr = (
            self.cfg['transformer-lr']
            * self.cfg['adim'] ** (-0.5)
            * min((self.trainer.global_step+1) ** (-0.5), (self.trainer.global_step+1) * self.cfg['transformer-warmup-steps'] ** (-1.5))
        )

        self.log('lr', lr, sync_dist=True)

        # skip the first 500 steps
        for pg in optimizer.param_groups:
            pg["lr"] = lr


def get_model(input_dim, output_dim):
    model = ConformerModel(input_dim, output_dim)
    return model


def proto_model(input_dim, output_dim):
    model = ConformerModel(input_dim, output_dim)
    return model
