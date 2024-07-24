import gc
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Optimizer

from torchmetrics.functional import accuracy

from binary_nn.models.common.binary.modules import Sign, SignUnsat
from utils.configuration import ConfigurableMixin


class SPSA(ConfigurableMixin):
    def __init__(self, c=1e-3, modules_to_hook=(nn.Linear, nn.Conv1d, nn.Conv2d)):
        super().__init__()
        self.c = c
        self.modules_to_hook = modules_to_hook

    def _generate_perturbation(self, shape, device=None):
        return torch.distributions.Bernoulli(probs=torch.tensor([0.5], device=device)).sample(shape).squeeze(-1) * 2 - 1

    @torch.no_grad()
    def __call__(self, model, x, y, opt: Optimizer, loss_fn):

        # compute statistics for batchnorm and other similar modules
        model.train()
        y_pred = model(x)
        l = loss_fn(y_pred, y)

        model.eval()

        candidate_modules = []

        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                candidate_modules.append(module)

        i = torch.randint(0, len(candidate_modules), (1,)).item()
        mod = candidate_modules[i]

        old_state = deepcopy(mod.state_dict())

        # compute the perturbation vector and w_minus
        perturbation = []
        for p in mod.parameters():
            pert = self._generate_perturbation(p.shape, device=p.device)
            perturbation.append(pert)
            p.data -= pert*self.c

        y_minus = model(x)
        l_minus = loss_fn(y_minus, y)

        # compute w_plus
        mod.load_state_dict(old_state)
        for p, pert in zip(mod.parameters(), perturbation):
            p.data += pert*self.c

        y_plus = model(x)
        l_plus = loss_fn(y_plus, y)

        # compute the final gradient
        gain = (l_plus - l_minus) / (2.0 * self.c)

        opt.zero_grad()
        mod.load_state_dict(old_state)
        for p, pert in zip(mod.parameters(), perturbation):
            p.grad = pert*gain
        opt.step()

        return l, y_pred



