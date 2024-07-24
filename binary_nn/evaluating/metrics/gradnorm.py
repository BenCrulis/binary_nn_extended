from typing import Any

import torch
from torch import nn

import torchmetrics
from torch.nn.utils.parametrize import ParametrizationList
from torchmetrics import Metric


def computes_grad_norm(mod: nn.Module):
    p = [p.grad.flatten() for p in mod.parameters() if p.grad is not None]
    p = torch.cat(p)
    return torch.linalg.vector_norm(p)


class GradNorm(Metric):
    is_differentiable = False
    full_state_update = True

    def __init__(self, model: nn.Module, sample):
        super().__init__()
        self.module_list = []

        def hook(mod, input, output):
            self.module_list.append(mod)

        handles = []
        for mod in model.modules():
            handle = mod.register_forward_hook(hook)
            handles.append(handle)

        model.eval()
        model(sample)
        for handle in handles:
            handle.remove()

        self.module_list = [m for m in self.module_list if len(list(m.parameters(recurse=False))) > 0 and not isinstance(m, ParametrizationList)]
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, *args, **kwargs) -> None:
        norms = torch.stack([computes_grad_norm(m) for m in self.module_list])
        self.sum = self.sum + norms
        self.total += 1

    def compute(self) -> Any:
        return self.sum / self.total
