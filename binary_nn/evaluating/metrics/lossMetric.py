from typing import Any

import torchmetrics
from torchmetrics import Metric


class LossMetric(Metric):
    def __init__(self, loss_fn):
        super().__init__()
        self.loss_fn = loss_fn
        self.mean_metric = torchmetrics.MeanMetric()

    def update(self, *args, **kwargs) -> None:
        return self.mean_metric(self.loss_fn(*args, **kwargs))

    def compute(self) -> Any:
        return self.mean_metric.compute()
