from typing import Any

import torch
from torch import nn

import torchmetrics
from torchmetrics import Metric

from binary_nn.models.common.binary.modules import Sign, SignUnsat


class Saturation(Metric):
    is_differentiable = False
    full_state_update = True

    def __init__(self, model: nn.Module, threshold=1.0):
        super().__init__()
        self.module_list = []

        self.threshold = threshold

        to_hook = (nn.Sigmoid, nn.SiLU, nn.ReLU, nn.ELU, nn.GELU, nn.CELU, nn.Hardswish, nn.Softplus, Sign, SignUnsat)

        handles = []
        for mod in model.modules():
            if isinstance(mod, to_hook):
                handle = mod.register_forward_hook(self.saturation_hook)
                handles.append(handle)

        self.add_state("sum", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.sum_buffer = 0
        self.total_buffer = 0

    def saturation_hook(self, mod, input, output):
        input = input[0] if isinstance(input, tuple) else input
        self.sum_buffer += (torch.abs(input) >= self.threshold).sum()
        self.total_buffer += input.numel()

    def reset_buffers(self):
        self.sum_buffer = 0
        self.total_buffer = 0

    def update(self, *args, **kwargs) -> None:
        self.sum += self.sum_buffer
        self.total += self.total_buffer

    def compute(self) -> Any:
        return self.sum / self.total
