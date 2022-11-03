from dataclasses import dataclass
from typing import Optional

from src.data_selection import DataSelection
from src.talking_with_hands.metadata import Metadata


@dataclass
class DataHyperparameters:
    data_selection: DataSelection

    def __post_init__(self) -> None:
        if isinstance(self.data_selection, str):
            self.data_selection = DataSelection(self.data_selection)


@dataclass
class WindowDataHyperparameters(DataHyperparameters):
    fps: int
    window_size: int  # in frames
    displace_positional_data: Optional[bool] = False
    max_number_of_frames_per_subject: Optional[int] = None

    @property
    def frame_step_size(self):
        return Metadata.fps // self.fps

    @property
    def window_length_in_s(self):
        return self.window_size / self.fps

    @property
    def token(self):
        return f"{self.data_selection}_{self.fps}fps_{self.window_size}window-size"


@dataclass
class BinDataHyperparameters(DataHyperparameters):
    frames_per_bin: int
    max_number_of_frames_per_subject: Optional[int] = None
    displace_positional_data: Optional[bool] = False

    @property
    def token(self):
        return f"{self.data_selection}_{self.frames_per_bin}frames-per-bin"
