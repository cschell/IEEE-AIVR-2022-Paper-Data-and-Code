import os

from src.data_selection import DataSelection

import numpy as np
import pandas as pd

import pytest

from src.datamodules.datasets.bin_dataset import BinDataset
from src.hyperparameters.data_hyperparameters import BinDataHyperparameters


@pytest.fixture()
def dummy_hdf5_file():
    hdf5_path = "./_test_data.hdf5"
    build_dummy_hdf5_file(hdf5_path)
    yield hdf5_path
    os.remove(hdf5_path)


def build_dummy_hdf5_file(output_path):
    num_takes = 10
    num_subjects = 3

    df_list = []
    for take_id in range(num_takes):
        take_length = np.random.randint(311, 1023)
        frame_idx = np.arange(take_length)
        subject_id = np.random.randint(0, num_subjects)
        df = pd.DataFrame({"frame_idx": frame_idx})
        df["subject_id"] = subject_id
        df["take_id"] = take_id

        # feature columns are set to the subject_id so that the correct
        # association of sample values and targets (x and y) can be verified
        df[BinDataset.feature_columns] = subject_id

        df_list.append(df)

    final_df = pd.concat(df_list, ignore_index=True)
    final_df.to_hdf(output_path, key="dummy", mode="w", index=False)


def test_correct_x_and_y_assignment_in_batches(dummy_hdf5_file):
    hdf5_file_path = dummy_hdf5_file
    hdf5_key = "dummy"
    data_hyperparameters = BinDataHyperparameters(frames_per_bin=30, data_selection=DataSelection.SCENE_RELATIVE)

    dataset = BinDataset(hdf5_file_path, hdf5_key, data_hyperparameters)

    for batch in dataset:
        x, y = batch["data"], batch["targets"]
        unscaled_x = ((x * dataset.data_stats["stds"]) + dataset.data_stats["means"]).numpy().round(5)

        # the test data are built so that each feature has the corresponding subject id
        # as value, so here the value of the features (x) should equal the sample target (y)
        assert (unscaled_x[0] == dataset.labels[y]).all()
