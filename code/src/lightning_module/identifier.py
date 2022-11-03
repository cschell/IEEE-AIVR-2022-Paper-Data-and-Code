from collections import defaultdict
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import wandb
from torch import nn as nn

from src.metrics import DatasetPurpose, MetricType, initialize_metric


class Identifier(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_weights=None, optimizer_options: dict = None, metrics: List[dict] = None, *, labels: List[str] = None):
        super().__init__()
        self.labels = labels
        self.optimizer_options = {} if optimizer_options is None else optimizer_options
        self.hyperparameters = model.hparams.__dict__
        self.model = model
        self.save_hyperparameters()
        self.best_logged_metrics = defaultdict(lambda: np.nan)

        # the Lightning Trainer#tune expects "batch_size" to be set, even though we don't need it here
        self.batch_size = 1

        self._initialize_metrics([] if metrics is None else metrics)

        if loss_weights is not None:
            self.loss_weights = torch.from_numpy(loss_weights).float().to(self.device)
        else:
            self.loss_weights = None

    def _initialize_metrics(self, metrics):
        self.metrics = nn.ModuleDict({
            f"{DatasetPurpose.TRAIN.value}_metrics": nn.ModuleDict({}),
            f"{DatasetPurpose.VALIDATION.value}_metrics": nn.ModuleDict({}),
        })
        for metric in metrics:
            self.metrics[f"{metric['dataset']}_metrics"][metric["name"]] = initialize_metric(metric["name"], num_out_classes=self.model.num_out_classes)

    def forward(self, X):
        return self.model.forward(X)

    def training_step(self, batch, _batch_idx):
        X = batch["data"].float()
        y = batch["targets"].long()

        h = self.forward(X)

        loss = F.cross_entropy(h, y.long(), weight=self.loss_weights.to(self.device)).mean()
        self.log(f"loss/{DatasetPurpose.TRAIN.value}", loss, on_step=False, on_epoch=True)

        for metric_name, metric_fn in self.metrics[f"{DatasetPurpose.TRAIN.value}_metrics"].items():
            self.log(f"{metric_name}/{DatasetPurpose.TRAIN.value}", metric_fn.cpu()(h.cpu(), y.cpu()), on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer_name = self.optimizer_options.pop("name", "Adam")
        optimizer = getattr(torch.optim, optimizer_name)(params=self.parameters(), **self.optimizer_options)

        return optimizer

    def validation_step(self, batch, batch_idx):
        X = batch["data"].float()
        y = batch["targets"].long()

        h = self.forward(X)
        return h, y

    def validation_epoch_end(self, validation_step_outputs):
        h = torch.cat([h_ for h_, y_ in validation_step_outputs])
        y = torch.cat([y_ for h_, y_ in validation_step_outputs])

        val_loss_weights = torch.from_numpy(self.trainer.val_dataloaders[0].dataset.loss_weights).float().to(self.device)
        loss = F.cross_entropy(h, y.long(), weight=val_loss_weights).mean()
        self.log(f"loss/{DatasetPurpose.VALIDATION.value}", loss, on_step=False, on_epoch=True, )

        preds = h.softmax(axis=1)

        torch.use_deterministic_algorithms(False)
        for metric_name, metric_fn in self.metrics[f"{DatasetPurpose.VALIDATION.value}_metrics"].items():
            self.log(f"{metric_name}/{DatasetPurpose.VALIDATION.value}", metric_fn(preds, y), on_step=False, on_epoch=True, prog_bar=metric_name == MetricType.MIN_ACCURACY.value)

        wandb.log({"confusion_matrix/val": wandb.plot.confusion_matrix(probs=h.cpu().detach().numpy(), y_true=y.cpu().detach().numpy())})

        torch.use_deterministic_algorithms(True)
        self._note_best_metric_values()
        return loss

    def _note_best_metric_values(self):
        for metric_name, value in self.trainer.logged_metrics.items():
            old_min_value = self.best_logged_metrics.get((metric_name, "min"), np.inf)
            old_max_value = self.best_logged_metrics.get((metric_name, "max"), -np.inf)

            self.best_logged_metrics[(metric_name, "min")] = min(old_min_value, self.trainer.logged_metrics[metric_name])
            self.best_logged_metrics[(metric_name, "max")] = max(old_max_value, self.trainer.logged_metrics[metric_name])
