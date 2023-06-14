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


class Wav2VecModelNoWarmup(pl.LightningModule):
    def __init__(self, output_dim, lang_dir=None, cfg=None):
        """
        Args:

        output_dim: int, the number of output units
        lang_dir: str, the directory of the language data, e.g, data/lang
        cfg: dict, actually this is a hydra config object, which contains all the configurations for the model

        """

        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.save_hyperparameters()

        bundle = getattr(torchaudio.pipelines, cfg["model"])
        self.wav2vec = bundle.get_model()
        self.freese_and_init()
        self.encoder_output_size = self.cfg["encoder_output_size"]
        self.batch_norm = nn.BatchNorm1d(cfg.model.adim)
        self.output_layer = nn.Linear(self.encoder_output_size, self.output_dim)

        if self.cfg.training.loss == "builtin_ctc":
            self.lang = Lang(lang_dir)
        elif self.cfg.training.loss == "k2":
            self.lang = Lang(lang_dir, load_topo=True, load_lexicon=True)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        # By default, only fine-tune the encoder part of the wav2vec model and fix the feature_extractor part
        for para in self.wav2vec.feature_extractor.parameters():
            para.requires_grad = False
        # for i in range(1, self.cfg['finetune_layers']+1):
        # for para in self.wav2vec.encoder.transformer.layers[-i].parameters():
        # para.requires_grad = True

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):
        if self.cfg.training.loss in ["k2"]:
            wavs = batch["wavs"]
            lengths = batch["wav_lens"]
            word_ids = batch["word_ids"]
            target_lengths = batch["target_lengths"]

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor(
                [[i, 0, xlens[i]] for i in range(batch_num)],
                device="cpu",
                dtype=torch.int32,
            )

            dense_fsa_vec = k2.DenseFsaVec(
                log_probs=log_probs, supervision_segments=supervision_segments
            )

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device
            )

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(
                decoding_graph=decoding_graph,
                dense_fsa_vec=dense_fsa_vec,
                target_lengths=target_lengths,
                reduction="mean",
            )

            if "no_den" in self.cfg.training.keys() and self.cfg.training.no_den:
                loss = numerator
            elif (
                "no_den_grad" in self.cfg.training.keys()
                and self.cfg.training.no_den_grad
            ):
                with torch.no_grad():
                    den_decoding_graph = k2.create_fsa_vec(
                        [self.lang.topo.to(log_probs.device) for _ in range(batch_num)]
                    )

                    assert den_decoding_graph.requires_grad == False

                    denominator = graph.graphloss(
                        decoding_graph=den_decoding_graph,
                        dense_fsa_vec=dense_fsa_vec,
                        target_lengths=target_lengths,
                        reduction="mean",
                    )
                loss = numerator - denominator
            else:
                den_decoding_graph = k2.create_fsa_vec(
                    [self.lang.topo.to(log_probs.device) for _ in range(batch_num)]
                )

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(
                    decoding_graph=den_decoding_graph,
                    dense_fsa_vec=dense_fsa_vec,
                    target_lengths=target_lengths,
                    reduction="mean",
                )
                loss = numerator - denominator
        elif self.cfg.training.loss == "builtin_ctc":
            wavs = batch["feats"]
            lengths = batch["feats_lens"]
            target_lengths = batch["target_lengths"]
            targets_ctc = batch["targets_ctc"]
            log_probs, xlens = self(wavs, lengths)
            loss = F.ctc_loss(
                log_probs.permute(1, 0, 2),
                targets_ctc,
                xlens,
                target_lengths,
                reduction="mean",
            )

        else:
            raise Exception(
                "Unrecognised Loss Function %s" % (self.cfg["training"]["loss"])
            )

        return loss

    def forward(self, x, xlens):
        x, xlens = self.wav2vec(x, xlens)
        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

    def debugging(self):
        norm_sum_4 = 0.0
        norm_sum_3 = 0.0
        norm_sum_2 = 0.0
        norm_sum_1 = 0.0
        norm_sum_lin = 0.0
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
        print("norm_sum_4", norm_sum_4)
        print("norm_sum_3", norm_sum_3)
        print("norm_sum_2", norm_sum_2)
        print("norm_sum_1", norm_sum_1)
        print("norm_sum_lin", norm_sum_lin)

        # print('torch.max(para.grad)', torch.max(para.grad))
        # print('torch.min(para.grad)', torch.min(para.grad))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self.compute_loss(batch, batch_idx, optimizer_idx)
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["wav_lens"]
        targets = batch["targets"]
        names = batch["names"]
        spks = batch["spks"]
        texts = batch["texts"]
        log_probs, xlens = self(wavs, lengths)
        return log_probs, xlens, names, spks, texts

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters(), lr=self.cfg["training"]["lr"])
        return [optimiser], [
            {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser,
                    "min",
                    patience=self.cfg["training"]["lr_patience"],
                    verbose=True,
                    factor=self.cfg["training"]["factor"],
                    min_lr=self.cfg["training"]["min_lr"],
                ),
                "monitor": "valid_loss",
            }
        ]


class Wav2VecModel(pl.LightningModule):
    def __init__(self, output_dim, lang_dir=None, cfg=None):
        """
        Args:

        output_dim: int, the number of output units
        lang_dir: str, the directory of the language data, e.g, data/lang
        cfg: dict, actually this is a hydra config object, which contains all the configurations for the model

        """

        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.save_hyperparameters()

        bundle = getattr(torchaudio.pipelines, cfg["model"])
        self.wav2vec = bundle.get_model()
        self.freese_and_init()
        self.encoder_output_size = self.cfg["encoder_output_size"]
        self.batch_norm = nn.BatchNorm1d(cfg.model.adim)
        self.output_layer = nn.Linear(self.encoder_output_size, self.output_dim)

        if self.cfg.training.loss == "builtin_ctc":
            self.lang = Lang(lang_dir)
        elif self.cfg.training.loss == "k2":
            self.lang = Lang(lang_dir, load_topo=True, load_lexicon=True)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        # By default, only fine-tune the encoder part of the wav2vec model and fix the feature_extractor part
        for para in self.wav2vec.feature_extractor.parameters():
            para.requires_grad = False
        # for i in range(1, self.cfg['finetune_layers']+1):
        # for para in self.wav2vec.encoder.transformer.layers[-i].parameters():
        # para.requires_grad = True

    def compute_loss(self, batch, batch_idx=None, optimizer_idx=None):
        if self.cfg.training.loss in ["k2"]:
            wavs = batch["wavs"]
            lengths = batch["wav_lens"]
            word_ids = batch["word_ids"]
            target_lengths = batch["target_lengths"]

            batch_num = int(wavs.shape[0])
            log_probs, xlens = self(wavs, lengths)

            supervision_segments = torch.tensor(
                [[i, 0, xlens[i]] for i in range(batch_num)],
                device="cpu",
                dtype=torch.int32,
            )

            dense_fsa_vec = k2.DenseFsaVec(
                log_probs=log_probs, supervision_segments=supervision_segments
            )

            decoding_graph = self.lang.compile_training_graph(
                word_ids, log_probs.device
            )

            assert decoding_graph.requires_grad == False

            numerator = graph.graphloss(
                decoding_graph=decoding_graph,
                dense_fsa_vec=dense_fsa_vec,
                target_lengths=target_lengths,
                reduction="mean",
            )

            if "no_den" in self.cfg.training.keys() and self.cfg.training.no_den:
                loss = numerator
            elif (
                "no_den_grad" in self.cfg.training.keys()
                and self.cfg.training.no_den_grad
            ):
                with torch.no_grad():
                    den_decoding_graph = k2.create_fsa_vec(
                        [self.lang.topo.to(log_probs.device) for _ in range(batch_num)]
                    )

                    assert den_decoding_graph.requires_grad == False

                    denominator = graph.graphloss(
                        decoding_graph=den_decoding_graph,
                        dense_fsa_vec=dense_fsa_vec,
                        target_lengths=target_lengths,
                        reduction="mean",
                    )
                loss = numerator - denominator
            else:
                den_decoding_graph = k2.create_fsa_vec(
                    [self.lang.topo.to(log_probs.device) for _ in range(batch_num)]
                )

                assert den_decoding_graph.requires_grad == False

                denominator = graph.graphloss(
                    decoding_graph=den_decoding_graph,
                    dense_fsa_vec=dense_fsa_vec,
                    target_lengths=target_lengths,
                    reduction="mean",
                )
                loss = numerator - denominator
        elif self.cfg.training.loss == "builtin_ctc":
            wavs = batch["feats"]
            lengths = batch["feats_lens"]
            target_lengths = batch["target_lengths"]
            targets_ctc = batch["targets_ctc"]
            log_probs, xlens = self(wavs, lengths)
            loss = F.ctc_loss(
                log_probs.permute(1, 0, 2),
                targets_ctc,
                xlens,
                target_lengths,
                reduction="mean",
            )

        else:
            raise Exception(
                "Unrecognised Loss Function %s" % (self.cfg["training"]["loss"])
            )

        return loss

    def forward(self, x, xlens):
        x, xlens = self.wav2vec(x, xlens)
        x = self.batch_norm(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

    def debugging(self):
        norm_sum_4 = 0.0
        norm_sum_3 = 0.0
        norm_sum_2 = 0.0
        norm_sum_1 = 0.0
        norm_sum_lin = 0.0
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
        print("norm_sum_4", norm_sum_4)
        print("norm_sum_3", norm_sum_3)
        print("norm_sum_2", norm_sum_2)
        print("norm_sum_1", norm_sum_1)
        print("norm_sum_lin", norm_sum_lin)

        # print('torch.max(para.grad)', torch.max(para.grad))
        # print('torch.min(para.grad)', torch.min(para.grad))

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        loss = self.compute_loss(batch, batch_idx, optimizer_idx)
        self.log("loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("valid_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        self.log("test_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def predict_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["wav_lens"]
        targets = batch["targets"]
        names = batch["names"]
        spks = batch["spks"]
        texts = batch["texts"]
        log_probs, xlens = self(wavs, lengths)
        return log_probs, xlens, names, spks, texts

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(
            self.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )
        return optimiser

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        optimizer.step(closure=optimizer_closure)
        lr = self.cfg.training["final_lr"] * min(
            (self.trainer.global_step + 1) ** (-0.5)
            * self.cfg.training["transformer-warmup-steps"] ** (0.5),
            (self.trainer.global_step + 1)
            * self.cfg.training["transformer-warmup-steps"] ** (-1),
        )

        for pg in optimizer.param_groups:
            pg["lr"] = lr
