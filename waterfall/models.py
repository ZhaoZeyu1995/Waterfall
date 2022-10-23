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


class Wav2VecFineTuning(pl.LightningModule):
    def __init__(self,
                 output_size,
                 cfg=None):

        super().__init__()
        self.output_size = output_size
        self.cfg = cfg
        self.save_hyperparameters()
        bundle = getattr(torchaudio.pipelines, cfg['model'])
        self.wav2vec = bundle.get_model()
        self.freese_and_init()
        self.output_layer = nn.Linear(1024, self.output_size)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        for para in self.wav2vec.parameters():
            para.requires_grad = False
        for i in range(1, self.cfg['finetune_layers']+1):
            for para in self.wav2vec.encoder.transformer.layers[-i].parameters():
                # para.normal_(mean=0.0, std=1.0)
                para.requires_grad = True

    def forward(self, x, xlens):
        x, xlens = self.wav2vec(x, xlens)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

    def training_step(self, batch, batch_idx, optimizer_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        target_lengths = batch['target_lengths']
        targets = batch['targets']
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = F.ctc_loss(log_probs=log_probs.permute(1, 0, 2),
                          input_lengths=xlens,
                          target_lengths=target_lengths,
                          targets=targets,
                          reduction='sum')

        self.log('loss', loss/batch_num, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss/batch_num

    def validation_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        target_lengths = batch['target_lengths']
        targets = batch['targets']
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = F.ctc_loss(log_probs=log_probs.permute(1, 0, 2),
                          input_lengths=xlens,
                          target_lengths=target_lengths,
                          targets=targets,
                          reduction='sum')

        self.log('valid_loss', loss/batch_num, prog_bar=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        return loss/batch_num

    def test_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        target_lengths = batch['target_lengths']
        targets = batch['targets']
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = F.ctc_loss(log_probs=log_probs.permute(1, 0, 2),
                          input_lengths=xlens,
                          target_lengths=target_lengths,
                          targets=targets,
                          reduction='sum')
        self.log('test_loss', loss/batch_num, on_step=True,
                 on_epoch=True, sync_dist=True)
        return loss/batch_num

    def predict_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        targets = batch['targets']
        names = batch['names']
        spks = batch['spks']
        texts = batch['texts']
        log_probs, xlens = self(wavs, lengths)
        return log_probs, xlens, names, spks, texts

    def align(self, batch):
        with torch.no_grad():
            wavs = batch['wavs']
            lengths = batch['lengths']
            trans = batch['trans']
            trans_lengths = batch['trans_lengths']
            transform = batch['transform']
            names = batch['names']
            log_probs, xlens = self(wavs, lengths)

            log_gamma_norm = prior.compute_align(log_probs=log_probs,
                                                 input_lengths=xlens,
                                                 trans=trans,
                                                 trans_lengths=trans_lengths,
                                                 transform=transform)
            return log_gamma_norm, xlens, trans_lengths, names

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.wav2vec.parameters(), lr=1e-4)
        optimiser_output = torch.optim.Adam(self.output_layer.parameters())
        return [optimiser, optimiser_output], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True),
                                               'monitor': 'valid_loss'},
                                               {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_output, 'min', patience=2, verbose=True),
                                               'monitor': 'valid_loss'}]


class Wav2VecFineTuningAlign(Wav2VecFineTuning):
    def __init__(self,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

    def predict_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        trans = batch['trans']
        trans_lengths = batch['trans_lengths']
        transform = batch['transform']
        names = batch['names']
        log_probs, xlens = self(wavs, lengths)

        log_gamma_norm = prior.compute_align(log_probs=log_probs,
                                             input_lengths=xlens,
                                             trans=trans,
                                             trans_lengths=trans_lengths,
                                             transform=transform)
        return log_gamma_norm, xlens, trans_lengths, names


class Wav2VecFineTuningDiverse(pl.LightningModule):
    def __init__(self,
                 output_size,
                 lang_dir=None,
                 cfg=None):

        super().__init__()
        self.output_size = output_size
        self.cfg = cfg
        self.save_hyperparameters()

        bundle = getattr(torchaudio.pipelines, cfg['model'])
        self.wav2vec = bundle.get_model()
        self.freese_and_init()
        if 'encoder_output_size' in self.cfg.keys():
            self.encoder_output_size = self.cfg['encoder_output_size']
        else:
            self.encoder_output_size = 1024
        self.output_layer = nn.Linear(
            self.encoder_output_size, self.output_size)

        if self.cfg['loss'] == 'ctc_softmax':
            if 'init_eta' in self.cfg.keys():
                self.eta = self.cfg['init_eta']
                print('eta has been initialised as %f' % (self.eta))
            else:
                self.eta = 1.

        if lang_dir and self.cfg['loss'] in ['ctc_k2']:
            self.lang = Lang(lang_dir, load_topo=True)

        if lang_dir and self.cfg['loss'] in ['k2']:
            self.lang = Lang(lang_dir, load_topo=True,
                             load_lexicon=True, load_den_graph=True)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        for para in self.wav2vec.parameters():
            para.requires_grad = False
        if 'train_output_layer_only' in self.cfg.keys() and self.cfg['train_output_layer_only']:
            return
        else:
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

            loss = F.ctc_loss(log_probs=log_probs.permute(1, 0, 2),
                              input_lengths=xlens,
                              target_lengths=target_lengths,
                              targets=targets,
                              reduction='sum')
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

            decoding_graph = k2.compose(
                self.lang.topo.to(log_probs.device), trans_fsa, treat_epsilons_specially=False)

            assert decoding_graph.requires_grad == False

            loss = graph.graphloss(decoding_graph=decoding_graph,
                                   dense_fsa_vec=dense_fsa_vec,
                                   output_beam=self.cfg['output_beam'],
                                   reduction='sum')
        elif self.cfg['loss'] in ['k2']:
            wavs = batch['wavs']
            lengths = batch['lengths']
            targets = batch['targets']
            word_ids = batch['word_ids']

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor([[i, 0, xlens[i]] for i in range(batch_num)],
                                                device='cpu',
                                                dtype=torch.int32)

            dense_fsa_vec = k2.DenseFsaVec(log_probs=torch.cat([torch.zeros_like(log_probs[:, :, :1], dtype=log_probs.dtype), log_probs], dim=-1),
                                           supervision_segments=supervision_segments)

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device)

            assert decoding_graph.requires_grad == False

            if 'mask_inf' not in self.cfg.keys() or not self.cfg['mask_inf']:
                numerator = graph.graphloss(decoding_graph=decoding_graph,
                                            dense_fsa_vec=dense_fsa_vec,
                                            output_beam=self.cfg['output_beam'],
                                            reduction='sum')
            else:
                numerator = graph.graphloss(decoding_graph=decoding_graph,
                                            dense_fsa_vec=dense_fsa_vec,
                                            output_beam=self.cfg['output_beam'],
                                            reduction='none')
                inf_mask = torch.logical_not(torch.isinf(numerator))
                if False in inf_mask:
                    logging.warn(
                        'There are utterances whose inputs are shorter than labels..')
                numerator = torch.masked_select(numerator, inf_mask).sum()
            if 'no_den' in self.cfg.keys() and self.cfg['no_den']:
                loss = numerator
            else:
                if 'den_with_lexicon' in self.cfg.keys() and self.cfg['den_with_lexicon']:
                    den_decoding_graph = self.lang.den_graph.to(
                        log_probs.device)
                else:
                    den_decoding_graph = self.lang.topo.to(log_probs.device)

                assert den_decoding_graph.requires_grad == False

                if 'mask_inf' not in self.cfg.keys() or not self.cfg['mask_inf']:
                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  output_beam=self.cfg['output_beam'] if 'output_beam_den' not in self.cfg.keys(
                                                  ) else self.cfg['output_beam_den'],
                                                  reduction='sum')
                else:
                    denominator = graph.graphloss(decoding_graph=den_decoding_graph,
                                                  dense_fsa_vec=dense_fsa_vec,
                                                  output_beam=self.cfg['output_beam'] if 'output_beam_den' not in self.cfg.keys(
                                                  ) else self.cfg['output_beam_den'],
                                                  reduction='none')
                    denominator = torch.masked_select(
                        denominator, inf_mask).sum()
                loss = numerator - denominator

        return loss/batch_num

    def forward(self, x, xlens):
        x, xlens = self.wav2vec(x, xlens)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

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
        loss = self.division(batch, batch_idx, optimizer_idx)
        self.log('loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.division(batch, batch_idx)
        self.log('valid_loss', loss, prog_bar=True,
                 on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.division(batch, batch_idx)
        self.log('test_loss', loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        targets = batch['targets']
        names = batch['names']
        spks = batch['spks']
        texts = batch['texts']
        log_probs, xlens = self(wavs, lengths)
        return log_probs, xlens, names, spks, texts

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
            if self.cfg['loss'] == 'ctc_softmax' and 'auto_eta_scheduler' in self.cfg.keys() and self.cfg['auto_eta_scheduler']:
                return [optimiser, optimiser_output], [{'scheduler': eta_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=self.cfg['patience_eta'], verbose=True, factor=0.2),
                                                        'monitor': 'valid_loss'},
                                                       {'scheduler': eta_scheduler.ReduceLROnPlateau(optimiser_output, 'min', patience=self.cfg['patience_eta'], verbose=True, factor=0.2),
                                                        'monitor': 'valid_loss'}]
            return [optimiser, optimiser_output], [{'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, 'min', patience=2, verbose=True),
                                                   'monitor': 'valid_loss'},
                                                   {'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser_output, 'min', patience=2, verbose=True),
                                                   'monitor': 'valid_loss'}]


class Wav2VecFineTuningDiverseAlign(Wav2VecFineTuningDiverse):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def predict_step(self, batch, batch_idx):
        wavs = batch['wavs']
        lengths = batch['lengths']
        trans = batch['trans']
        trans_lengths = batch['trans_lengths']
        transform = batch['transform']
        names = batch['names']
        log_probs, xlens = self(wavs, lengths)

        log_gamma_norm = prior.compute_align(log_probs=log_probs,
                                             input_lengths=xlens,
                                             trans=trans,
                                             trans_lengths=trans_lengths,
                                             transform=transform)
        return log_gamma_norm, xlens, trans_lengths, names


def get_model(num_tokens):
    model = Wav2VecFineTuning(num_tokens)
    return model


def proto_model(num_tokens):
    model = Wav2VecFineTuning(num_tokens)
    return model
