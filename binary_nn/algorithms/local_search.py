import gc

import torch
from torch import nn
from torch.optim import Optimizer

from utils.configuration import ConfigurableMixin


class LS(ConfigurableMixin):
    def __init__(self, inversion_prob=0.1, modules_to_hook=(nn.Linear, nn.Conv1d, nn.Conv2d)):
        super().__init__()
        self.prob = inversion_prob
        self.modules_to_hook = modules_to_hook

    @torch.no_grad()
    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        model.eval()

        candidate_modules = []

        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                candidate_modules.append(module)

        i = torch.randint(0, len(candidate_modules), (1,)).item()
        mod = candidate_modules[i]

        old_state = mod.state_dict()

        y_pred1 = model(x)
        l1 = loss_fn(y_pred1, y)

        for p in mod.parameters():
            mask = torch.rand(p.shape) < self.prob
            p.data[mask] *= -1

        y_pred2 = model(x)
        l2 = loss_fn(y_pred2, y)

        l = l2
        y_pred = y_pred2

        if l1 <= l2:
            l = l1
            y_pred = y_pred1
            mod.load_state_dict(old_state)

        model.train()
        model(x)

        return l, y_pred



