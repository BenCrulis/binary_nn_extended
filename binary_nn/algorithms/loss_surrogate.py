import gc
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Optimizer

from torchmetrics.functional import accuracy

from binary_nn.models.common.binary.modules import Sign, SignUnsat
from utils.configuration import ConfigurableMixin


class LossSurrogate(ConfigurableMixin):
    def __init__(self, last_module, modules_to_hook=(nn.Linear, nn.Conv1d, nn.Conv2d)):
        super().__init__()
        self.last_module = last_module
        self.modules_to_hook = modules_to_hook

    def config(self):
        return {}

    def __call__(self, model: nn.Module, x, y, opt: Optimizer, loss_fn):

        modules = []
        for mod in model.modules():
            if isinstance(mod, self.modules_to_hook) and mod is not self.last_module:
                modules.append(mod)

        return None



