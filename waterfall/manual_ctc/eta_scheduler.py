from pytorch_lightning import callbacks
import logging
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

import types
import math
from functools import wraps
import warnings
import weakref
from collections import Counter
from bisect import bisect_right

from torch.optim import Optimizer
from torch._six import inf
import warnings


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    """Reduce learning rate when a metric has stopped improving.
    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = ReduceLROnPlateau(optimizer, 'min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step(val_loss)
    """

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10,
                 threshold=1e-4, threshold_mode='rel', cooldown=0,
                 min_lr=0, eps=1e-8, verbose=False, reset=True):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        if isinstance(min_lr, list) or isinstance(min_lr, tuple):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError("expected {} min_lrs, got {}".format(
                    len(optimizer.param_groups), len(min_lr)))
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = 0
        self.reset = reset
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            if old_lr - new_lr > self.eps:
                param_group['lr'] = new_lr
                if self.verbose:
                    print('Epoch {:5d}: reducing learning rate'
                          ' of group {} to {:.4e}.'.format(epoch, i, new_lr))
                if self.reset:
                    self._reset()

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' +
                             threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


class AutoEtaScheduler(Callback):
    r"""
    Monitor a metric and stop training when it stops improving.

    Args:
        monitor: quantity to be monitored.
        min_delta: minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than or equal to `min_delta`, will count as no improvement.
        patience: number of checks with no improvement
            after which training will be stopped. Under the default configuration, one check happens after
            every training epoch. However, the frequency of validation can be modified by setting various parameters on
            the ``Trainer``, for example ``check_val_every_n_epoch`` and ``val_check_interval``.

            .. note::

                It must be noted that the patience parameter counts the number of validation checks with
                no improvement, and not the number of training epochs. Therefore, with parameters
                ``check_val_every_n_epoch=10`` and ``patience=3``, the trainer will perform at least 40 training
                epochs before being stopped.

        verbose: verbosity mode.
        mode: one of ``'min'``, ``'max'``. In ``'min'`` mode, training will stop when the quantity
            monitored has stopped decreasing and in ``'max'`` mode it will stop when the quantity
            monitored has stopped increasing.
        strict: whether to crash the training if `monitor` is not found in the validation metrics.
        check_finite: When set ``True``, stops training when the monitor becomes NaN or infinite.
        stopping_threshold: Stop training immediately once the monitored quantity reaches this threshold.
        divergence_threshold: Stop training as soon as the monitored quantity becomes worse than this threshold.
        check_on_train_epoch_end: whether to run early stopping at the end of the training epoch.
            If this is ``False``, then the check runs at the end of the validation.

    Raises:
        MisconfigurationException:
            If ``mode`` is none of ``"min"`` or ``"max"``.
        RuntimeError:
            If the metric ``monitor`` is not available.
    """
    mode_dict = {"min": torch.lt, "max": torch.gt}

    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str,
        init_eta: float = 1.0,
        delta_eta: float = 0.2,
        final_eta: float = 4.0,
        min_delta: float = 0.0,
        patience: int = 3,
        verbose: bool = False,
        mode: str = "min",
        strict: bool = True,
        check_finite: bool = True,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        check_on_train_epoch_end: Optional[bool] = None,
    ):
        super().__init__()
        self.monitor = monitor
        self.init_eta = init_eta
        self.eta = self.init_eta
        self.delta_eta = delta_eta
        self.final_eta = final_eta
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.strict = strict
        self.check_finite = check_finite
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.wait_count = 0
        self.stopped_epoch = 0
        self._check_on_train_epoch_end = check_on_train_epoch_end

        if self.mode not in self.mode_dict:
            raise MisconfigurationException(
                f"`mode` can be {', '.join(self.mode_dict.keys())}, got {self.mode}")

        self.min_delta *= 1 if self.monitor_op == torch.gt else -1
        torch_inf = torch.tensor(np.Inf)
        self.best_score = torch_inf if self.monitor_op == torch.lt else -torch_inf

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode)

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        if self._check_on_train_epoch_end is None:
            # if the user runs validation multiple times per training epoch or multiple training epochs without
            # validation, then we run after validation instead of on train epoch end
            self._check_on_train_epoch_end = trainer.val_check_interval == 1.0 and trainer.check_val_every_n_epoch == 1

    def _validate_condition_metric(self, logs: Dict[str, float]) -> bool:
        monitor_val = logs.get(self.monitor)

        error_msg = (
            f"AutoEtaScheduler conditioned on metric `{self.monitor}` which is not available."
            " Pass in or modify your `AutoEtaScheduler` callback to use any of the following:"
            f' `{"`, `".join(list(logs.keys()))}`'
        )

        if monitor_val is None:
            if self.strict:
                raise RuntimeError(error_msg)
            if self.verbose > 0:
                rank_zero_warn(error_msg, category=RuntimeWarning)

            return False

        return True

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "wait_count": self.wait_count,
            "stopped_epoch": self.stopped_epoch,
            "best_score": self.best_score,
            "patience": self.patience,
            "delta_eta": self.delta_eta,
            "final_eta": self.final_eta,
            "init_eta": self.init_eta,
            "eta": self.eta
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.wait_count = state_dict["wait_count"]
        self.stopped_epoch = state_dict["stopped_epoch"]
        self.best_score = state_dict["best_score"]
        self.patience = state_dict["patience"]
        self.delta_eta = state_dict['delta_eta']
        self.final_eta = state_dict['final_eta']
        self.init_eta = state_dict['init_eta']
        self.eta = state_dict['eta']

    def _should_skip_check(self, trainer: "pl.Trainer") -> bool:
        from pytorch_lightning.trainer.states import TrainerFn

        return trainer.state.fn != TrainerFn.FITTING or trainer.sanity_checking

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        should_change = self._run_early_stopping_check(trainer)
        if should_change:
            print('Change eta from %f to %f' % (pl_module.eta, self.eta))
            pl_module.eta = self.eta

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        should_change = self._run_early_stopping_check(trainer)
        if should_change:
            print('Change eta from %f to %f' % (pl_module.eta, self.eta))
            pl_module.eta = self.eta

    def _run_early_stopping_check(self, trainer: "pl.Trainer") -> None:
        """Checks whether the eta changing condition is met and if so tells the trainer to change the eta."""
        logs = trainer.callback_metrics

        if trainer.fast_dev_run or not self._validate_condition_metric(  # disable early_stopping with fast_dev_run
            logs
        ):  # short circuit if metric not present
            return

        current = logs[self.monitor].squeeze()
        should_stop, should_change, reason = self._evaluate_stopping_criteria(
            current)

        # stop every ddp process if any world process decides to stop
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop)
        trainer.should_stop = trainer.should_stop or should_stop
        if should_stop:
            self.stopped_epoch = trainer.current_epoch
        if reason and self.verbose:
            self._log_info(trainer, reason)
        return should_change

    def _evaluate_stopping_criteria(self, current: torch.Tensor) -> Tuple[bool, Optional[str]]:
        should_stop = False
        should_change = False
        reason = None
        if self.check_finite and not torch.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}. Signaling Trainer to stop."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
                " Signaling Trainer to stop."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score.to(current.device)):
            should_stop = False
            reason = self._improvement_message(current)
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                if self.eta + self.delta_eta > self.final_eta:
                    should_stop = True
                    reason = (
                        f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                        f" Best score: {self.best_score:.3f}. Signaling Trainer to stop."
                        f" Final eta : {self.eta:.3f}."
                    )
                else:
                    should_stop = False
                    should_change = True
                    reason = (
                        f"Monitored metric {self.monitor} did not improve in the last {self.wait_count} records."
                        f" Best score: {self.best_score:.3f}. Signaling Trainer to change eta."
                        f" Change Eta : from {self.eta:.3f} to {self.eta + self.delta_eta:.3f}."
                    )
                    self.eta += self.delta_eta
                    self.best_score = torch.tensor(np.Inf) if self.monitor_op == torch.lt else -torch.tensor(np.Inf)
                    self.wait_count = 0
        return should_stop, should_change, reason

    def _improvement_message(self, current: torch.Tensor) -> str:
        """Formats a log message that informs the user about an improvement in the monitored score."""
        if torch.isfinite(self.best_score):
            msg = (
                f"Metric {self.monitor} improved by {abs(self.best_score - current):.3f} >="
                f" min_delta = {abs(self.min_delta)}. New best score: {current:.3f}"
                f" the current eta = {self.eta}"
            )
        else:
            msg = f"Metric {self.monitor} improved. New best score: {current:.3f} the current eta = {self.eta}"
        return msg

    @staticmethod
    def _log_info(trainer: Optional["pl.Trainer"], message: str) -> None:
        if trainer is not None and trainer.world_size > 1:
            log.info(f"[rank: {trainer.global_rank}] {message}")
            print(f"[rank: {trainer.global_rank}] {message}")
        else:
            log.info(message)
