import torch

from src.datamodules.datasets.base_dataset import BaseDataset
from src.datamodules.datasets.bin_maker import BinMaker


class BinDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.return_frame_ids = False

        self.frames_per_package = self.data_hyperparameters.frames_per_bin
        self.frame_step_size = 1

        self._load_and_set_data()
        self._compute_and_set_packages()

    def _compute_and_set_packages(self):
        sfpm = BinMaker(frames=self.frames, take_ids=self.take_id, data_selection=self.data_hyperparameters.data_selection)
        packages, frame_ids = sfpm.make(frames_per_package=self.frames_per_package)

        self._own_means = packages.mean().values
        self._own_stds = packages.std().values
        self._set_or_load_data_stats()

        assert len(packages) == len(frame_ids)
        assert max(frame_ids) == len(self.frames) - 1, f"{max(frame_ids)} == {len(self.frames) - 1}"

        self.frame_ids = frame_ids
        self.packages = self._prepare_data(packages.values)
        self.package_targets = torch.from_numpy(self.frame_targets[frame_ids]).long()

    @property
    def num_features(self):
        return self.packages.shape[1]

    @property
    def num_samples(self):
        return self.packages.shape[0]

    def __getitem__(self, sample_idx):
        data = self.packages[sample_idx]
        targets = self.package_targets[sample_idx]

        if len(data) == 0:
            raise IndexError

        batch = {
            "data": data,
            "targets": targets
        }

        if self.return_frame_ids:
            batch["frame_ids"] = self.frame_ids[sample_idx].astype("int32").reshape(1)

        return batch
