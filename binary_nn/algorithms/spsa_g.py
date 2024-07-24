import math

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


class SPSAG(ConfigurableMixin):
    def __init__(self, n_classes, c=1e-3, perturbation="hebbian", modules_to_hook=(nn.Linear, nn.Conv1d, nn.Conv2d)):
        super().__init__()
        self.n_classes = n_classes
        self.c = c
        self.modules_to_hook = modules_to_hook
        self.perturbation = perturbation

    def config(self):
        return {"spsa-g-pert": self.perturbation}

    def _drtp_get_mat(self, mod, x, y):
        if hasattr(mod, "drtp_backward"):
            return mod.drtp_backward
        if isinstance(mod, nn.Conv1d):
            pass

    def _generate_perturbation(self, mod, x, y):
        device = x.device
        if self.perturbation == "drtp":
            eye = torch.eye(self.n_classes, device=device)
            signal = eye[y]
            signal -= signal.mean(1)[:, None]
            bw = self._drtp_get_mat(mod, x, y)
            pert = signal @ bw
        elif self.perturbation == "hebbian":
            mean = x.mean([x for x in range(1, len(x.shape))])
            cx = x - mean.view((-1, *[1 for x in x.shape[1:]])).expand(x.shape)
            pert = cx
        else:
            raise ValueError(f"unknown perturbation type: {self.perturbation}")
        pert /= torch.linalg.vector_norm(pert, keepdim=True)
        return pert

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

        perturbation = None
        saved_out = None
        handle = None

        def forward_hook(mod, input, output):
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
            c = self.c
            pert = self._generate_perturbation(mod, out, y)
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
        gain = (l_plus - l_minus) / (2.0 * self.c)

        opt.zero_grad()
        saved_out.backward(perturbation*gain)
        opt.step()
        handle.remove()
        return l, y_pred



