from typing import Any, Optional, Callable

import torch
import torchmetrics

class MinAccuracy(torchmetrics.Accuracy):
    def __init__(self, threshold: float = 0.5,
        num_classes: Optional[int] = None,
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        subset_accuracy: bool = False,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,):
        average = "none"

        super().__init__(threshold, num_classes, average, mdmc_average, ignore_index, top_k, multiclass, subset_accuracy, compute_on_step, dist_sync_on_step, process_group,
                         dist_sync_fn)

    def compute(self) -> torch.Tensor:
        accuracies = super().compute()

        return accuracies.min()
