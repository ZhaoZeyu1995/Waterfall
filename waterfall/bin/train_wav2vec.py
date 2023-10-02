#!/usr/bin/env python3

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import argparse
import yaml
import random
import numpy as np
import logging
from waterfall import wav2vec
from waterfall.utils import datapipe
import wandb
import hydra
from hydra.utils import get_original_cwd, to_absolute_path


@hydra.main(
    version_base=None,
    config_path=os.path.join(os.getcwd(), "conf"),
    config_name="config",
)
def main(cfg):
    pl.seed_everything(cfg.training.seed, workers=True)

    batch_size = cfg.training.batch_size

    ctc_target = False
    if cfg.training.loss == "builtin_ctc":
        ctc_target = True

    if "sort" in cfg.training.keys():
        if cfg.training.sort == "ascending":
            logging.info("Sorting data by ascending order of duration")
        elif cfg.training.sort == "descending":
            logging.info("Sorting data by descending order of duration")
    else:
        logging.info("Not sorting data")

    if cfg.training.loss == "builtin_ctc":
        lang = datapipe.Lang(to_absolute_path(cfg.data.lang_dir))
    elif cfg.training.loss == "k2":
        lang = datapipe.Lang(to_absolute_path(cfg.data.lang_dir), load_topo=True, load_lexicon=True)

    train_data = datapipe.Dataset(
        to_absolute_path(cfg.data.train_set),
        lang,
        ctc_target=ctc_target,
        load_wav=True,
        transforms=None,
        ratio_th=cfg.model.ratio_th,
        min_frames=cfg.model.min_frames,
        max_duration=cfg.model.max_duration,
        sort=cfg.training.sort,
    )
    dev_data = datapipe.Dataset(
        to_absolute_path(cfg.data.dev_set),
        lang,
        ctc_target=ctc_target,
        load_wav=True,
        transforms=None,
        ratio_th=cfg.model.ratio_th,
        min_frames=cfg.model.min_frames,
        max_duration=cfg.model.max_duration,
    )

    if "split_dev" in cfg.training.keys() and cfg.training.split_dev > 0:
        logging.info("Splitting dev set to make it smaller")
        num_kept = int(cfg.training.split_dev * len(dev_data))
        num_discarded = len(dev_data) - num_kept
        dev_data = random_split(dev_data, [num_kept, num_discarded])[0]
        logging.info("Kept %d examples, discarded %d examples" % (num_kept, num_discarded))

    logging.info("Training set size: %d" % len(train_data))
    shuffle = True if cfg.training.sort is None else False
    if shuffle:
        logging.info("Shuffling training set")
    else:
        logging.info("Not shuffling training set as it is already sorted {} by duration".format(cfg.training.sort))
    train_gen = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        collate_fn=datapipe.collate_fn_sorted,
        pin_memory=True,
    )
    dev_gen = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        persistent_workers=True,
        collate_fn=datapipe.collate_fn_sorted,
    )


    if cfg.training.nowarmup:
        logging.info("Training without warmup")
        model = wav2vec.Wav2VecModelNoWarmup(
            lang.num_nn_output,
            cfg=cfg,
            lang=lang,
        )
    else:
        model = wav2vec.Wav2VecModel(
            lang.num_nn_output,
            cfg=cfg,
            lang=lang,
        )

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="valid_loss",
        save_top_k=cfg.training.save_top_k,
        every_n_epochs=1,
        dirpath=os.path.join(cfg.training.output_dir, "checkpoints"),
        filename="{epoch}-{valid_loss:.4f}",
        mode="min",
    )

    callbacks = [
        model_checkpoint,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        pl.callbacks.RichProgressBar(),
        pl.callbacks.RichModelSummary(max_depth=2),
    ]

    if cfg.training.early_stopping:
        callbacks.append(
            pl.callbacks.EarlyStopping(
                monitor="valid_loss",
                mode="min",
                patience=cfg.training.patience,
                verbose=False,
            )
        )

    logger = pl.loggers.WandbLogger(
        project="waterfall-%s-%s"
        % (
            os.path.basename(os.path.dirname(get_original_cwd())),
            os.path.basename(get_original_cwd()),
        ),
        name=cfg.training.name,
        save_dir=os.path.join(cfg.training.output_dir),
    )
    logger.watch(model, log_freq=500, log_graph=False)

    if "checkpoint" in cfg.training.keys() and cfg.training.checkpoint is not None:
        if (
            "load_weights_only" in cfg.training.keys()
            and not cfg.training.load_weights_only
        ):
            trainer = pl.Trainer(
                accelerator="gpu" if "accelerator" not in cfg.training.keys() else cfg.training.accelerator,
                strategy="auto" if "strategy" not in cfg.training.keys() else cfg.training.strategy,
                precision=32 if "precision" not in cfg.training.keys() else cfg.training.precision,
                devices=cfg.training.gpus,
                deterministic=False,
                resume_from_checkpoint=cfg.training.checkpoint,
                max_epochs=cfg.training.max_epochs,
                logger=logger,
                callbacks=callbacks,
                sync_batchnorm=True,
                val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
            )
        else:
            checkpoint = torch.load(
                cfg.training.checkpoint, map_location=torch.device("cpu")
            )
            model.load_state_dict(checkpoint["state_dict"])
            del checkpoint
            torch.cuda.empty_cache()
            trainer = pl.Trainer(
                accelerator="gpu" if "accelerator" not in cfg.training.keys() else cfg.training.accelerator,
                strategy="auto" if "strategy" not in cfg.training.keys() else cfg.training.strategy,
                precision=32 if "precision" not in cfg.training.keys() else cfg.training.precision,
                devices=cfg.training.gpus,
                deterministic=False,
                max_epochs=cfg.training.max_epochs,
                logger=logger,
                callbacks=callbacks,
                sync_batchnorm=True,
                val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
            )
    else:
        trainer = pl.Trainer(
            accelerator="gpu" if "accelerator" not in cfg.training.keys() else cfg.training.accelerator,
            strategy="auto" if "strategy" not in cfg.training.keys() else cfg.training.strategy,
            precision=32 if "precision" not in cfg.training.keys() else cfg.training.precision,
            devices=cfg.training.gpus,
            deterministic=False,
            max_epochs=cfg.training.max_epochs,
            logger=logger,
            callbacks=callbacks,
            sync_batchnorm=True,
            val_check_interval=1.0 if 'val_check_interval' not in cfg.training.keys() else cfg.training.val_check_interval,
        )

    trainer.fit(model, train_gen, dev_gen)

    logger.log_metrics(
        {
            "best_model_path": os.path.join(
                os.getcwd(), model_checkpoint.best_model_path
            ),
            "best_model_loss": model_checkpoint.best_model_score.item(),
        }
    )

    wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        level=logging.INFO,
    )
    main()
