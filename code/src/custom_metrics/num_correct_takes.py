import torch
import torchmetrics


class NumCorrectTakes(torchmetrics.Metric):
    def __init__(self, num_classes, dist_sync_on_step=False, *args):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.num_classes = num_classes
        self.add_state("summed_predictions", default=torch.zeros((num_classes, num_classes)))

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        for subject_idx in range(self.num_classes):
            self.summed_predictions[subject_idx] += preds[target == subject_idx].sum(axis=0)

    def compute(self) -> torch.Tensor:
        return (self.summed_predictions.argmax(axis=1).cpu() == torch.arange(self.num_classes).cpu()).sum()
