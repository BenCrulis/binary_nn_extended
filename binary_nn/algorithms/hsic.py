import torch
from mlp_mixer_pytorch.mlp_mixer_pytorch import PreNormResidual
from torch import nn
from torch.optim import Optimizer
from torchvision.models.mobilenetv2 import InvertedResidual

from utils.configuration import ConfigurableMixin

from binary_nn.utils import hsic


class HSIC(ConfigurableMixin):
    def __init__(self, num_classes, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d), gamma=2.0):
        super().__init__()
        self.n_classes = num_classes
        self.modules_to_hook = modules_to_hook
        self.z_kernel = hsic.CosineSimilarityKernel()
        self.y_kernel = hsic.CosineSimilarityKernel()
        self.gamma = gamma
        self.mode = "biased"
        self.modules_to_detach = (PreNormResidual, InvertedResidual)

    def config(self):
        return {"HSIC gamma": self.gamma, "HSIC mode": self.mode}

    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        model.train()

        device = x.device
        eye = torch.eye(self.n_classes, device=device)
        signal = eye[y]
        signal -= signal.mean(1)[:, None]

        def detach_output_hook(mod, input, output):
            output = output[0] if isinstance(output, tuple) else output
            return output.detach()

        def signal_hook(mod, input):
            input = input[0] if isinstance(input, tuple) else input
            if not input.requires_grad:
                return
            l = hsic.estimate_hsic_zy_objective(input.view((len(input), -1)), signal, self.z_kernel, self.y_kernel, self.gamma)
            l.backward()
            return input.detach()

        handles = []
        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                handle = module.register_forward_pre_hook(signal_hook)
                handles.append(handle)
            if isinstance(module, self.modules_to_detach):
                handle = module.register_forward_hook(detach_output_hook)
                handles.append(handle)

        opt.zero_grad()
        y_pred = model(x)
        l = loss_fn(y_pred, y)
        l.backward()
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred