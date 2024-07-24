from collections import OrderedDict
from typing import Any

import torch
from torch import nn

import torchmetrics
from torch.nn.utils.parametrize import ParametrizationList
from torchmetrics import Metric

from binary_nn.models.common.binary.modules import Sign, SignUnsat, BiRealAct


def recurse_modules_except_parametrizations(mod: nn.Module):
    for m in mod.children():
        if not isinstance(m, ParametrizationList):
            yield m
            yield from recurse_modules_except_parametrizations(m)


class Saturation():
    def __init__(self, model: nn.Module, threshold=1.0):
        super().__init__()
        self.module_list = []

        self.threshold = threshold

        to_hook = (nn.Sigmoid, nn.SiLU, nn.ReLU, nn.ELU, nn.GELU, nn.CELU, nn.Hardswish, nn.Softplus, Sign, SignUnsat,
                   BiRealAct)

        handles = []
        for mod in recurse_modules_except_parametrizations(model):
            if isinstance(mod, to_hook):
                handle = mod.register_forward_hook(self.saturation_hook)
                handles.append(handle)

        self.sum_buffer = OrderedDict()
        self.total_buffer = OrderedDict()
        self.device = None

    def saturation_hook(self, mod, input, output):
        input = input[0] if isinstance(input, tuple) else input
        self.device = input.device

        self.sum_buffer[mod] = (torch.abs(input) >= self.threshold).sum().item()
        self.total_buffer[mod] = input.numel()

        # self.sum_buffer += (torch.abs(input) >= self.threshold).sum().item()
        # self.total_buffer += input.numel().item()

    def reset_buffers(self):
        self.sum_buffer.clear()
        self.total_buffer.clear()

    def __call__(self):
        return torch.tensor([s/t for s, t in zip(self.sum_buffer.values(), self.total_buffer.values())])
