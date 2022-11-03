import os
import pathlib
from typing import Optional, Union, Type

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.datamodules.datasets.bin_dataset import BinDataset
from src.datamodules.datasets.window_dataset import WindowDataset

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 6))


class BaseDatamodule(LightningDataModule):
    dataset_klass: Union[Type[BinDataset], Type[WindowDataset]]
    train_dataset: Union[BinDataset, WindowDataset] = None
    val_dataset: Union[BinDataset, WindowDataset] = None
    test_dataset: Union[BinDataset, WindowDataset] = None

    default_dataloader_settings = dict(
            num_workers=NUM_WORKERS,
            pin_memory=True,
    )

    def __init__(self, data_path: str, batch_size: int, data_stats_path, data_hyperparameters, data_loader_settings=None, dataset_kwargs=None):
        super().__init__()
        if dataset_kwargs:
            self.dataset_kwargs = dataset_kwargs
        else:
            self.dataset_kwargs = {}
        self.data_stats_path = pathlib.Path(data_stats_path)

        if not data_loader_settings:
            data_loader_settings = {}

        self.data_loader_settings = {**self.default_dataloader_settings, **data_loader_settings}

        self.batch_size = batch_size
        self.data_path = pathlib.Path(data_path)
        self.data_hyperparameters = data_hyperparameters

    def setup(self, stage: Optional[str] = "train"):
        if stage == "test":
            self.test_dataset = self.dataset_klass(self.data_path,
                                                   "test",
                                                   enforce_data_stats=self.data_stats_path,
                                                   data_hyperparameters=self.data_hyperparameters,
                                                   **self.dataset_kwargs)
        elif stage == "validate":
            self.val_dataset = self.dataset_klass(self.data_path,
                                                  "validation",
                                                  enforce_data_stats=self.data_stats_path,
                                                  data_hyperparameters=self.data_hyperparameters,
                                                  **self.dataset_kwargs)
        else:
            if not self.train_dataset:
                self.train_dataset = self.dataset_klass(self.data_path, "train",
                                                        data_hyperparameters=self.data_hyperparameters,
                                                        **self.dataset_kwargs)
                self.train_dataset.safe_settings(self.data_stats_path)

            if not self.val_dataset:
                self.val_dataset = self.dataset_klass(self.data_path, "validation",
                                                      enforce_data_stats=self.train_dataset.data_stats,
                                                      data_hyperparameters=self.data_hyperparameters,
                                                      **self.dataset_kwargs)

    def shuffle_train(self):
        self.train_dataset.reshuffle()

    def train_dataloader(self):
        self.data_loader_settings["batch_size"] = self.batch_size
        return DataLoader(self.train_dataset, shuffle=True, **self.data_loader_settings)

    def val_dataloader(self):
        self.data_loader_settings["batch_size"] = self.batch_size
        return DataLoader(self.val_dataset, **self.data_loader_settings)

    def test_dataloader(self):
        self.data_loader_settings["batch_size"] = self.batch_size
        return DataLoader(self.test_dataset, **self.data_loader_settings)

    def safe_train_data_stats_settings(self):
        self.train_dataset.safe_settings(self.data_stats_path)


class BinDatamodule(BaseDatamodule):
    dataset_klass = BinDataset


class WindowDatamodule(BaseDatamodule):
    dataset_klass = WindowDataset
