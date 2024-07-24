import torch
from mlp_mixer_pytorch.mlp_mixer_pytorch import PreNormResidual
from torch import nn
from torch.optim import Optimizer
from torchvision.models.mobilenetv2 import InvertedResidual

from binary_nn.models.common.binary.modules import Sign
from utils.configuration import ConfigurableMixin

from binary_nn.utils import hsic


class HSIC(ConfigurableMixin):
    def __init__(self, num_classes, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d), gamma=2.0, mode="biased"):
        super().__init__()
        self.n_classes = num_classes
        self.modules_to_hook = modules_to_hook
        self.z_kernel = hsic.CosineSimilarityKernel()
        self.y_kernel = hsic.CosineSimilarityKernel()
        self.gamma = gamma
        self.mode = mode
        self.modules_to_detach = (PreNormResidual, InvertedResidual)

    def config(self):
        return {"hsic-gamma": self.gamma, "hsic-mode": self.mode}

    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        model.train()

        device = x.device
        eye = torch.eye(self.n_classes, device=device)
        signal = eye[y]
        signal -= signal.mean(1)[:, None]

        def signal_hook(mod, tensor):
            tensor = tensor[0] if isinstance(tensor, tuple) else tensor
            if not tensor.requires_grad:
                return
            l = hsic.estimate_hsic_zy_objective(tensor.view((len(tensor), -1)), signal, self.z_kernel, self.y_kernel, self.gamma)
            l.backward()
            return tensor.detach()

        handles = []
        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                handle = module.register_forward_pre_hook(signal_hook)
                handles.append(handle)
            if isinstance(module, self.modules_to_detach):
                handle = module.register_forward_pre_hook(signal_hook)
                handles.append(handle)

        opt.zero_grad()
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        l.backward()
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred