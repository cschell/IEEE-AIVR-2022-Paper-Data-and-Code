import enum

import torchmetrics

from src.custom_metrics.min_accuracy import MinAccuracy
from src.custom_metrics.num_correct_takes import NumCorrectTakes


class MetricType(enum.Enum):
    ACCURACY = "accuracy"
    MIN_ACCURACY = "min_accuracy"
    F1 = "f1"
    PRECISION = "precision"
    RECALL = "recall"
    COHEN_KAPPA = "cohen_kappa"
    MATTHEWS = "matthews_corrcoef"
    LOSS = "loss"
    NUM_CORRECT_TAKES = "num_correct_takes"


class DatasetPurpose(enum.Enum):
    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"


def get_metric_name(metric: MetricType, purpose: DatasetPurpose):
    return "/".join([metric.value, purpose.value])


def initialize_metric(metric_name: str, num_out_classes: int = None):
    klass_name = "".join([s.capitalize() for s in metric_name.split("_")])

    if metric_name == MetricType.MIN_ACCURACY.value:
        return MinAccuracy(num_classes=num_out_classes)
    elif metric_name == MetricType.NUM_CORRECT_TAKES.value:
        return NumCorrectTakes(num_classes=num_out_classes)
    elif metric_name == MetricType.F1.value:
        return torchmetrics.classification.f_beta.F1(num_classes=num_out_classes)
    elif metric_name in [MetricType.COHEN_KAPPA.value, MetricType.MATTHEWS.value]:
        return getattr(getattr(torchmetrics.classification, metric_name), klass_name)(num_classes=num_out_classes)
    else:
        return getattr(torchmetrics, klass_name)(num_classes=num_out_classes, average="macro")
