from typing import Union

import pandas as pd
import numpy as np


class WindowMaker():
    def __init__(self, num_frames, window_size, frame_step_size, change_idxs, is_acceleration=False):
        self.is_acceleration = is_acceleration
        self.num_frames = num_frames
        self.window_size = window_size
        self.frame_step_size = frame_step_size
        self.change_idxs = change_idxs
        self.windowed_ids = self._compute_windows()

    def shuffle_windows(self):
        np.random.shuffle(self.windowed_ids)

    @property
    def window_span(self):
        return (self.window_size - 1) * self.frame_step_size

    @property
    def num_windows(self):
        return len(self.windowed_ids)

    @property
    def num_samples(self):
        return self.num_windows

    def _compute_windows(self):
        frame_ids = np.arange(self.num_frames, dtype="uint32")

        num_shifts = self.window_size * self.frame_step_size

        max_num_windows = self.num_frames - self.window_span
        all_windows_np = np.zeros((max_num_windows, self.window_size), dtype="uint32")
        for offset in range(num_shifts):
            downscaled_frame_ids = frame_ids[offset:][::self.frame_step_size]

            offcut = -(len(downscaled_frame_ids) % self.window_size)

            if offcut == 0:
                offcut = None

            result = downscaled_frame_ids[:offcut].reshape(-1, self.window_size)
            all_windows_np[result[:, 0]] = result

        exclude_frame_ids = np.concatenate(
                [(self.change_idxs - i)[(self.change_idxs - i) >= 0] for i in range(1 - self.frame_step_size * (1 + self.is_acceleration), self.window_span + 1)]
        )

        cleaned_all_windows_np = np.delete(all_windows_np, exclude_frame_ids, axis=0)

        return cleaned_all_windows_np

    def to_windows(self, unwindowed_data: Union[pd.DataFrame, np.ndarray], window_idxs: Union[np.ndarray, list]) -> np.ndarray:
        if type(unwindowed_data) != np.ndarray:
            unwindowed_data: np.ndarray = unwindowed_data.values

        return unwindowed_data[self.windowed_ids[window_idxs].flatten()].reshape(-1, self.window_size, unwindowed_data.shape[-1])

    def first_window_frame_ids(self, window_idxs: Union[np.ndarray, list]):

        return self.windowed_ids[window_idxs, 0]

    @staticmethod
    def compute_change_idxs(values):
        return np.where(np.diff(values, prepend=np.nan))[0]
