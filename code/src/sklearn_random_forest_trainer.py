import bz2
import logging
import pathlib
import pickle

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torchmetrics import classification as metrics

from src.custom_metrics.min_accuracy import MinAccuracy
from src.metrics import MetricType


class SKLearnRandomForestTrainer:
    def __init__(self, train_batcher, val_batcher, save_dir_path, model):
        self.model = model
        self.save_dir_path = pathlib.Path(save_dir_path)
        self.logger = logging.getLogger("SKLearnRandomForestTrainer")
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self._load_train_data()
        self._load_validation_data()

    def _load_train_data(self):
        self.X_train = self.train_batcher.packages.numpy()
        self.y_train = self.train_batcher.frame_targets[self.train_batcher.frame_ids]

    def _load_validation_data(self):
        self.X_val = self.val_batcher.packages.numpy()
        self.y_val = self.val_batcher.frame_targets[self.val_batcher.frame_ids]

    def fit(self):
        self.model.fit(self.X_train, self.y_train)
        self.save_model()

    def compute_train_scores(self):
        return self._compute_scores(self.X_train, self.y_train)

    def compute_validation_scores(self):
        return self._compute_scores(self.X_val, self.y_val)

    def _compute_scores(self, X, y):
        num_classes = y.max() + 1
        metric_objs = {
            MetricType.ACCURACY: metrics.Accuracy(num_classes=num_classes, average="macro"),
            MetricType.F1: metrics.F1(num_classes=num_classes, average="macro"),
            MetricType.COHEN_KAPPA: metrics.CohenKappa(num_classes=num_classes),
            MetricType.MATTHEWS: metrics.MatthewsCorrcoef(num_classes=num_classes),
            MetricType.MIN_ACCURACY: MinAccuracy(num_classes=num_classes),
        }

        idxs = np.arange(len(X))
        np.random.shuffle(idxs)

        for frame_idxs in BatchSampler(idxs, batch_size=10000, drop_last=False):
            X_ = X[frame_idxs]
            y_ = torch.from_numpy(y[frame_idxs])
            h = torch.from_numpy(self.model.predict(X_)).int()

            for metric_name, metric_obj in metric_objs.items():
                metric_obj.update(h, y_)

        computed_metrics = {metric_name: float(metric_obj.compute().numpy()) for metric_name, metric_obj in metric_objs.items()}

        return computed_metrics

    @property
    def save_path(self):
        return self.save_dir_path.joinpath("model.pkl")

    def save_model(self):
        # Random Forest models can get very big, up to 40 Gb, so we compress them here
        with bz2.BZ2File(f"{self.save_path}.bz2", "w") as f:
            pickle.dump(self.model, f)
        self.logger.info(f"model saved in {self.save_path}.bz2")
