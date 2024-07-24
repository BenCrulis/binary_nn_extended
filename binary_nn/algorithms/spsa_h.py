import gc
from copy import deepcopy

import torch
from torch import nn
from torch.nn.utils.parametrize import ParametrizationList
from torch.optim import Optimizer

from torchmetrics.functional import accuracy

from binary_nn.models.common.binary.modules import Sign, SignUnsat
from utils.configuration import ConfigurableMixin


class ExpendableTensor(torch.Tensor):
    def __add__(self, other):
        if self.shape[0] == other.shape[0] * 2:
            return torch.add(self, other.repeat((2, *[1 for i in other.shape[1:]])))
        elif self.shape[0] * 2 == other.shape[0]:
            return torch.add(self.repeat((2, *[1 for i in other.shape[1:]])), other)
        return torch.add(self, other)


class SPSAH(ConfigurableMixin):
    def __init__(self, c=1e-3, distribution="gaussian", modules_to_hook=(nn.Linear, nn.Conv1d, nn.Conv2d)):
        super().__init__()
        self.c = c
        self.modules_to_hook = modules_to_hook
        self.distribution = distribution

    def config(self):
        return {"SPSA-h distribution": self.distribution}

    def _generate_perturbation(self, shape, device=None):
        if self.distribution == "rademacher":
            return torch.distributions.Bernoulli(probs=torch.tensor([0.5], device=device)).sample(shape).squeeze(-1) * 2 - 1
        elif self.distribution == "gaussian":
            return torch.randn(shape, device=device)

    @torch.no_grad()
    def __call__(self, model: nn.Module, x, y, opt: Optimizer, loss_fn):

        # compute statistics for batchnorm and other similar modules
        model.train()
        y_pred = model(x)
        l = loss_fn(y_pred, y)

        model.eval()

        candidate_modules = []

        for module in model.modules():
            if len(list(module.parameters(recurse=False))) > 0 and not isinstance(module, ParametrizationList):
                candidate_modules.append(module)

        i = torch.randint(0, len(candidate_modules), (1,)).item()
        mod = candidate_modules[i]
        c = None

        perturbation = None
        saved_out = None
        handle = None

        def forward_hook(mod, input, output):
            nonlocal c
            nonlocal perturbation
            nonlocal saved_out
            nonlocal handle
            handle.remove()
            input = input[0] if isinstance(input, tuple) else input
            input = input.detach()
            with torch.enable_grad():
                out = mod(input)
                saved_out = out
            handle = mod.register_forward_hook(forward_hook)
            c = self.c / out.numel()
            pert = self._generate_perturbation(out.shape, device=out.device)
            perturbation = pert
            out_pos = out + pert*c
            out_neg = out - pert*c
            new_out = torch.cat([out_pos, out_neg], dim=0)
            return ExpendableTensor(new_out)

        handle = mod.register_forward_hook(forward_hook)
        with torch.no_grad():
            y_pred_pert = model(ExpendableTensor(x))

        l_plus = loss_fn(y_pred_pert[:len(x)], y)
        l_minus = loss_fn(y_pred_pert[len(x):], y)

        # compute the final gradient
        gain = (l_plus - l_minus) / (2.0 * c)

        opt.zero_grad()
        saved_out.backward(perturbation*gain)
        opt.step()
        handle.remove()
        return l, y_pred



