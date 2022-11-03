from typing import Iterable

import numpy as np
import pandas as pd

from src.data_selection import DataSelection
from src.datamodules.datasets.helpers import compute_change_idxs


class BinMaker:
    def __init__(self, frames: pd.DataFrame, take_ids: Iterable[int], data_selection: DataSelection):
        self.data_selection = data_selection
        self.take_ids = take_ids
        self.frame_change_idxs = compute_change_idxs(take_ids)
        self.frames = frames

        self.num_samples, self.num_features = frames.shape

    def make(self, frames_per_package: int):
        assert frames_per_package > 0

        # disregard first <data_category_padding> number of frames for velocity or acceleration data,
        # since these are NaN
        if self.data_selection == DataSelection.SCENE_RELATIVE or \
                self.data_selection == DataSelection.BODY_RELATIVE:
            data_selection_padding = 0
        elif self.data_selection == DataSelection.BODY_RELATIVE_VELOCITY:
            data_selection_padding = 1
        elif self.data_selection == DataSelection.ACCELERATION:
            data_selection_padding = 2
        else:
            raise Exception("unknown data category, don't know what to do")

        invalid_lookback_frames = np.concatenate([self.frame_change_idxs + i for i in range(frames_per_package + (data_selection_padding - 1))])
        invalid_frame_idxs = invalid_lookback_frames
        invalid_frame_idxs.sort()

        rolled_frames = self.frames.rolling(window=frames_per_package)

        means = rolled_frames.mean().add_suffix("_mean")
        stds = rolled_frames.std().add_suffix("_stds")
        medians = rolled_frames.median().add_suffix("_medians")
        mins = rolled_frames.min().add_suffix("_mins")
        maxs = rolled_frames.max().add_suffix("_maxs")

        packages = pd.concat(
                [means, stds, medians, mins, maxs],
                axis=1)

        frame_ids = np.arange(len(self.frames))
        mask = ~np.isin(frame_ids, invalid_frame_idxs)

        packages = packages[mask]
        frame_ids = frame_ids[mask]
        assert not np.isnan(packages.values).any()

        return packages, frame_ids
