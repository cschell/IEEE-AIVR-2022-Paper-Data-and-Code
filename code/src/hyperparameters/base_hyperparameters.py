import torch
from torch import nn


class BaseHyperparameters:
    @property
    def normalization_method(self):
        if self.normalization_method_name == "batch":
            return nn.BatchNorm2d
        elif self.normalization_method_name == "instance":
            return nn.InstanceNorm2d
        elif self.normalization_method_name == "none":
            return nn.Identity
        else:
            raise Exception("no valid normalization method provided")

    def __post_init__(self):
        if hasattr(self, "kernel_sizes") and hasattr(self, "layer_sizes"):
            assert len(self.kernel_sizes) == len(
                    self.layer_sizes), "numbers of configured layers and kernel sizes don't match; len(kernel_sizes) is %s and len(layer_sizes) is %s" % (
                len(self.kernel_sizes), len(self.layer_sizes))

    @property
    def optimizer(self):
        return getattr(torch.optim, self.optimizer_name)
