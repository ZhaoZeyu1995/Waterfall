import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
import torchaudio
from waterfall.manual_ctc.ctc import ctc_loss


class Wav2VecFineTuning(pl.LightningModule):
    def __init__(self, output_size, cfg=None):
        super().__init__()
        self.output_size = output_size
        self.cfg = cfg
        self.save_hyperparameters()
        bundle = torchaudio.pipelines.WAV2VEC2_LARGE_LV60K
        self.wav2vec = bundle.get_model()
        self.freese_and_init()
        self.output_layer = nn.Linear(1024, self.output_size)

    def freese_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        for para in self.wav2vec.parameters():
            para.requires_grad = False
        self.wav2vec_last_layer = self.wav2vec.encoder.transformer.layers[-1]
        for para in self.wav2vec_last_layer.parameters():
            para.normal_(mean=0.0, std=1.0)
            para.requires_grad = True

    def forward(self, x, xlens):
        x, xlens = self.wav2vec(x, xlens)
        x = self.output_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x, xlens

    def training_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["lengths"]
        trans = batch["trans"]
        trans_lengths = batch["trans_lengths"]
        transform = batch["transform"]
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = ctc_loss(
            log_probs,
            input_lengths=xlens,
            trans=trans,
            trans_lengths=trans_lengths,
            transform=transform,
            reduction="sum",
        )

        self.log("loss", loss / batch_num, on_step=True, on_epoch=True, sync_dist=True)
        return loss / batch_num

    def validation_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["lengths"]
        trans = batch["trans"]
        trans_lengths = batch["trans_lengths"]
        transform = batch["transform"]
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = ctc_loss(
            log_probs,
            input_lengths=xlens,
            trans=trans,
            trans_lengths=trans_lengths,
            transform=transform,
            reduction="sum",
        )

        self.log(
            "valid_loss",
            loss / batch_num,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss / batch_num

    def test_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["lengths"]
        trans = batch["trans"]
        trans_lengths = batch["trans_lengths"]
        transform = batch["transform"]
        batch_num = int(wavs.shape[0])
        log_probs, xlens = self(wavs, lengths)

        loss = ctc_loss(
            log_probs,
            input_lengths=xlens,
            trans=trans,
            trans_lengths=trans_lengths,
            transform=transform,
            reduction="sum",
        )

        self.log(
            "test_loss", loss / batch_num, on_step=True, on_epoch=True, sync_dist=True
        )
        return loss / batch_num

    def predict_step(self, batch, batch_idx):
        wavs = batch["wavs"]
        lengths = batch["lengths"]
        targets = batch["targets"]
        names = batch["names"]
        spks = batch["spks"]
        texts = batch["texts"]
        log_probs, xlens = self(wavs, lengths)
        return log_probs, xlens, names, spks, texts

    def configure_optimizers(self):
        optimiser = torch.optim.Adam(self.parameters())
        return {
            "optimizer": optimiser,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimiser, "min", patience=2, verbose=True
                ),
                "monitor": "valid_loss",
            },
        }


def get_model(num_tokens):
    model = Wav2VecFineTuning(num_tokens)
    return model


def proto_model(num_tokens):
    model = Wav2VecFineTuning(num_tokens)
    return model
