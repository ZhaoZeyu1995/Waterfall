import torch
import os
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import logging
import torchaudio
from waterfall import graph
import k2
from waterfall.utils.datapipe import Lang
import time


class Wav2VecModelNoWarmup(pl.LightningModule):
    def __init__(self, output_dim, lang=None, cfg=None):
        """
        Args:

        output_dim: int, the number of output units
        lang: str or a Lang object, if str, the directory of the language data, e.g, data/lang
        cfg: dict, actually this is a hydra config object, which contains all the configurations for the model

        """

        super().__init__()
        self.output_dim = output_dim
        self.cfg = cfg
        self.save_hyperparameters(ignore=["lang"])

        bundle = getattr(torchaudio.pipelines, cfg.model["model"])
        if (
            "wav2vec2_save_path" in self.cfg.model.keys()
            and self.cfg.model.wav2vec2_save_path is not None
        ):
            assert type(self.cfg.model.wav2vec2_save_path) == str
            os.makedirs(self.cfg.model.wav2vec2_save_path, exist_ok=True)
            logging.info(
                "Download and save wav2vec2 model to {}".format(
                    self.cfg.model.wav2vec2_save_path
                )
            )
            logging.info(
                "Load wav2vec2 model from {} if it already exists there.".format(
                    self.cfg.model.wav2vec2_save_path
                )
            )
            wav2vec = bundle.get_model(dl_kwargs={"model_dir": self.cfg.model.wav2vec2_save_path})
        else:
            wav2vec = bundle.get_model()
        # Deal with different torchaudio versions
        if hasattr(wav2vec, "model"):
            self.wav2vec = wav2vec.model
        else:
            self.wav2vec = wav2vec

        self.freeze_and_init()
        self.encoder_output_size = self.cfg.model["encoder_output_size"]
        self.batch_norm = nn.BatchNorm1d(self.encoder_output_size)
        self.output_layer = nn.Sequential(
            nn.Linear(self.encoder_output_size, self.encoder_output_size),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_output_size, self.encoder_output_size),
            nn.LeakyReLU(),
            nn.Linear(self.encoder_output_size, self.output_dim),
        )
        if isinstance(lang, str):
            if self.cfg.training.loss == "builtin_ctc":
                self.lang = Lang(lang)
            elif self.cfg.training.loss == "k2":
                self.lang = Lang(lang, load_topo=True, load_lexicon=True)
        elif isinstance(lang, Lang):
            self.lang = lang
        elif lang is None:
            logging.info(
                "No lang object is provided. This is fine if you are not training the model."
            )
            self.lang = None
        else:
            raise ValueError(
                "lang should be a str or a Lang object but got {}".format(lang)
            )

        # SpecAugment with TimeMasking and FrequencyMasking only but no TimeStretching
        # Implemented by torchaudio.transforms.TimeMasking and torchaudio.transforms.FrequencyMasking
        if "spec_augment" in self.cfg.model.keys() and self.cfg.model["spec_augment"]:
            self.time_masking = torchaudio.transforms.TimeMasking(
                time_mask_param=cfg.model["time_mask_param"], p=0.3
            )
            self.freq_masking = torchaudio.transforms.FrequencyMasking(
                freq_mask_param=cfg.model["freq_mask_param"]
            )

        self.automatic_optimization = False
        if (
            "accumulate_grad_batches" in self.cfg.training.keys()
            and self.cfg.training["accumulate_grad_batches"] != 1
        ):
            self.acc_loss = 0

    def freeze_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        # By default, only fine-tune the encoder part of the wav2vec model and fix the feature_extractor part
        # if we don't set model.finetune_layers with a positive integer
        for para in self.wav2vec.feature_extractor.parameters():
            para.requires_grad = False
        if "finetune_layers" in self.cfg.model.keys():
            logging.info(
                "Finetune the last {} layers of the wav2vec model".format(
                    self.cfg.model["finetune_layers"]
                )
            )
            for para in self.wav2vec.encoder.parameters():
                para.requires_grad = False
            if self.cfg.model["finetune_layers"] > 0:
                for i in range(1, self.cfg.model["finetune_layers"] + 1):
                    for para in self.wav2vec.encoder.transformer.layers[
                        -i
                    ].parameters():
                        para.requires_grad = True

    def compute_loss(self, batch, batch_idx=None):
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
            wavs = batch["wavs"]
            lengths = batch["wav_lens"]
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

    def training_step(self, batch, batch_idx):
        if self.cfg.model["finetune_layers"] > 0:
            opt_output, opt_wav2vec = self.optimizers()
        else:
            opt_output = self.optimizers()

        batch_size = int(batch["wavs"].shape[0])
        if (
            "accumulate_grad_batches" not in self.cfg.training.keys()
            or self.cfg.training["accumulate_grad_batches"] == 1
        ):
            loss = self.compute_loss(batch, batch_idx)

            self.log(
                "loss",
                loss,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                prog_bar=True,
            )

            opt_output.zero_grad()
            if self.cfg.model["finetune_layers"] > 0:
                opt_wav2vec.zero_grad()
            self.manual_backward(loss)

            self.clip_gradients(
                opt_output,
                gradient_clip_val=0.5
                if "grad_clip" not in self.cfg.training.keys()
                else self.cfg.training["grad_clip"],
                gradient_clip_algorithm="norm"
                if "grad_clip_algorithm" not in self.cfg.training.keys()
                else self.cfg.training["grad_clip_algorithm"],
            )
            opt_output.step()
            if self.cfg.model["finetune_layers"] > 0:
                self.clip_gradients(
                    opt_wav2vec,
                    gradient_clip_val=0.5
                    if "grad_clip" not in self.cfg.training.keys()
                    else self.cfg.training["grad_clip"],
                    gradient_clip_algorithm="norm"
                    if "grad_clip_algorithm" not in self.cfg.training.keys()
                    else self.cfg.training["grad_clip_algorithm"],
                )
                opt_wav2vec.step()
        else:
            loss = (
                self.compute_loss(batch, batch_idx)
                / self.cfg.training["accumulate_grad_batches"]
            )

            self.acc_loss += loss.item()

            self.manual_backward(loss)

            if (batch_idx + 1) % self.cfg.training[
                "accumulate_grad_batches"
            ] == 0 or self.trainer.is_last_batch:
                self.log(
                    "loss",
                    self.acc_loss,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    prog_bar=True,
                )
                opt_output.step()
                if self.cfg.model["finetune_layers"] > 0:
                    self.clip_gradients(
                        opt_wav2vec,
                        gradient_clip_val=0.5
                        if "grad_clip" not in self.cfg.training.keys()
                        else self.cfg.training["grad_clip"],
                        gradient_clip_algorithm="norm"
                        if "grad_clip_algorithm" not in self.cfg.training.keys()
                        else self.cfg.training["grad_clip_algorithm"],
                    )
                    opt_wav2vec.step()
                opt_output.zero_grad()
                if self.cfg.model["finetune_layers"] > 0:
                    opt_wav2vec.zero_grad()
                self.acc_loss = 0

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()
        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])

    def validation_step(self, batch, batch_idx):
        loss = self.compute_loss(batch, batch_idx)
        batch_size = int(batch["wavs"].shape[0])
        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
            prog_bar=True,
        )
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
        optimisers = []

        optimiser = torch.optim.Adadelta(
            self.output_layer.parameters(),
            lr=self.cfg["training"]["lr"],
            rho=self.cfg["training"]["rho"],
            eps=self.cfg["training"]["eps"],
        )
        optimisers.append(
            {
                "optimizer": optimiser,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimiser,
                        "min",
                        patience=self.cfg["training"]["lr_patience"],
                        verbose=True,
                        factor=self.cfg["training"]["factor"],
                        min_lr=self.cfg["training"]["min_lr"],
                    )
                },
            }
        )
        if self.cfg.model["finetune_layers"] > 0:
            optimiser_wav2vec = torch.optim.Adam(
                self.wav2vec.parameters(), lr=self.cfg["training"]["wav2vec_lr"]
            )
            optimisers.append({"optimizer": optimiser_wav2vec})
        return optimisers


class Wav2VecModel(Wav2VecModelNoWarmup):
    def __init__(self, output_dim, lang=None, cfg=None):
        super().__init__(output_dim, lang, cfg)
        assert (
            "final_lr" in self.cfg.training.keys()
        ), "final_lr must be specified in the config file"
        self.final_lr = self.cfg.training["final_lr"]

    def freeze_and_init(self):
        self.wav2vec.aux = None  # get rid of the output linear layer in wav2vec model
        # By default, only fine-tune the encoder part of the wav2vec model and fix the feature_extractor part
        # if we don't set model.finetune_layers with a positive integer
        for para in self.wav2vec.feature_extractor.parameters():
            para.requires_grad = False
        if "finetune_layers" in self.cfg.model.keys():
            logging.info(
                "Finetune the last {} layers of the wav2vec model".format(
                    self.cfg.model["finetune_layers"]
                )
            )
            for para in self.wav2vec.encoder.parameters():
                para.requires_grad = False
            if self.cfg.model["finetune_layers"] > 0:
                for i in range(1, self.cfg.model["finetune_layers"] + 1):
                    for para in self.wav2vec.encoder.transformer.layers[
                        -i
                    ].parameters():
                        para.requires_grad = True

        if (
            "train_output_layer_first" in self.cfg.training.keys()
            and self.cfg.training["train_output_layer_first"]
        ):
            assert (
                "num_training_output_layer_steps" in self.cfg.training.keys()
            ), "num_training_output_layer_steps must be specified if train_output_layer_first is True"
            logging.info(
                "Freeze the last {} layers except the output layer for the first {} steps".format(
                    self.cfg.model["finetune_layers"],
                    self.cfg.training["num_training_output_layer_steps"],
                )
            )
            self.finished_flag = False
            # Take a note of which layers are trainable
            self.trainable_wav2vec_paras = []
            for name, para in self.wav2vec.named_parameters():
                if para.requires_grad:
                    self.trainable_wav2vec_paras.append(name)
                    para.requires_grad = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        batch_size = int(batch["wavs"].shape[0])
        if (
            "accumulate_grad_batches" not in self.cfg.training.keys()
            or self.cfg.training["accumulate_grad_batches"] == 1
        ):
            loss = self.compute_loss(batch, batch_idx)
            self.log(
                "loss",
                loss,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
                prog_bar=True,
            )

            opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(
                opt,
                gradient_clip_val=0.5
                if "grad_clip" not in self.cfg.training.keys()
                else self.cfg.training["grad_clip"],
                gradient_clip_algorithm="norm"
                if "grad_clip_algorithm" not in self.cfg.training.keys()
                else self.cfg.training["grad_clip_algorithm"],
            )
            opt.step()
            self.adjustLR(opt)

        else:
            loss = (
                self.compute_loss(batch, batch_idx)
                / self.cfg.training["accumulate_grad_batches"]
            )
            self.acc_loss += loss.item()

            self.manual_backward(loss)

            if (batch_idx + 1) % self.cfg.training[
                "accumulate_grad_batches"
            ] == 0 or self.trainer.is_last_batch:
                self.log(
                    "loss",
                    self.acc_loss,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                    prog_bar=True,
                )
                opt.step()
                self.clip_gradients(
                    opt,
                    gradient_clip_val=0.5
                    if "grad_clip" not in self.cfg.training.keys()
                    else self.cfg.training["grad_clip"],
                    gradient_clip_algorithm="norm"
                    if "grad_clip_algorithm" not in self.cfg.training.keys()
                    else self.cfg.training["grad_clip_algorithm"],
                )
                opt.zero_grad()
                self.acc_loss = 0
                self.adjustLR(opt)

    def on_validation_epoch_end(self):
        pass

    def configure_optimizers(self):
        self.start_lr = (
            self.cfg["training"]["start_lr"]
            if "start_lr" in self.cfg["training"].keys()
            else 0.0
        )
        optimiser = torch.optim.Adam(
            self.parameters(), lr=self.start_lr, betas=(0.9, 0.98), eps=1e-9
        )
        return optimiser

    def adjustLR(self, optimizer):
        if (
            "train_output_layer_first" in self.cfg.training.keys()
            and self.cfg.training["train_output_layer_first"]
            and self.finished_flag is False
        ):
            if (
                self.trainer.global_step
                == self.cfg.training["num_training_output_layer_steps"]
            ):
                logging.info(
                    "Unfreezing wav2vec parameters in the last {} layers".format(
                        self.cfg.model["finetune_layers"]
                    )
                )
                for name, para in self.wav2vec.named_parameters():
                    if name in self.trainable_wav2vec_paras:
                        para.requires_grad = True
                self.finished_flag = True

        lr = min(
            self.cfg.training["final_lr"]
            * (self.trainer.global_step + 1) ** (-0.5)
            * self.cfg.training["transformer-warmup-steps"] ** (0.5),
            (self.trainer.global_step + 1)
            * (self.final_lr - self.start_lr)
            * self.cfg.training["transformer-warmup-steps"] ** (-1)
            + self.start_lr,
        )

        for pg in optimizer.param_groups:
            pg["lr"] = lr
