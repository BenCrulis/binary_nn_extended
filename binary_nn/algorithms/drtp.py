import torch
from mlp_mixer_pytorch.mlp_mixer_pytorch import PreNormResidual
from torch import nn
from torch.optim import Optimizer
from torchvision.models.mobilenetv2 import InvertedResidual

from utils.configuration import ConfigurableMixin


class DRTP(ConfigurableMixin):
    def __init__(self, output_layer, num_classes, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d)):
        super().__init__()
        self.n_classes = num_classes
        self.output_layer = output_layer
        self.modules_to_hook = modules_to_hook
        self.error = None

    def config(self):
        return {"DRTP version": "slow"}

    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        s = self

        model.train()

        handles = []
        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                class Hook():
                    def forward_hook(self, mod, input, output):
                        self.input = input[0] if isinstance(input, tuple) else input

                    @torch.no_grad()
                    def backward_hook(self, m: nn.Module, grad_input, grad_output):
                        if isinstance(grad_input, tuple) and len(grad_input) > 1:
                            raise ValueError("multi input module is not supported")
                        grad_input = grad_input[0] if isinstance(grad_input, tuple) else grad_input
                        if grad_input is None:
                            return None
                        grad_output = grad_output[0] if isinstance(grad_output, tuple) else grad_output
                        device = grad_output.device
                        backward_mat = m.drtp_backward if hasattr(m, "drtp_backward") else None
                        error = s.error
                        n_outputs = error.shape[-1]
                        if backward_mat is None:  # if the backward shortcut doesn't exist for this layer, create it
                            channel_dim = grad_input.shape[2] if len(grad_input.shape) == 3 else grad_input.shape[1]
                            backward_mat = torch.empty((n_outputs, channel_dim), device=device)
                            nn.init.normal_(backward_mat)
                            m.register_buffer("drtp_backward", backward_mat, persistent=True)
                        layer_error = error @ backward_mat
                        if len(grad_input.shape) == 3:
                            # the channel dimension is the third one for MLPMixer (even with Conv1D layers)
                            layer_error = layer_error[:, None, :].expand(grad_input.shape)
                        elif len(grad_input.shape) == 4:
                            layer_error = layer_error[:, :, None, None].expand(grad_input.shape)
                        assert layer_error.shape == grad_input.shape
                        return layer_error,
                hook = Hook()
                handle = module.register_forward_hook(hook.forward_hook)
                handles.append(handle)
                handle = module.register_full_backward_hook(hook.backward_hook)
                handles.append(handle)

        device = x.device
        eye = torch.eye(self.n_classes, device=device)
        signal = eye[y]
        signal -= signal.mean(1)[:, None]
        self.error = signal

        y_pred = model(x)
        l = loss_fn(y_pred, y)
        opt.zero_grad()
        l.backward()
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred


class DRTPFast(ConfigurableMixin):
    def __init__(self, output_layer, num_classes, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d)):
        super().__init__()
        self.n_classes = num_classes
        self.output_layer = output_layer
        self.modules_to_hook = modules_to_hook
        self.modules_to_detach = (PreNormResidual, InvertedResidual)

    def config(self):
        return {"DRTP version": "efficient"}

    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        model.train()

        device = x.device
        eye = torch.eye(self.n_classes, device=device)
        signal = eye[y]
        signal -= signal.mean(1)[:, None]

        def detach_hook(mod, input):
            input = input[0] if isinstance(input, tuple) else input
            return input.detach()

        def signal_hook(mod, input):
            input = input[0] if isinstance(input, tuple) else input
            device = input.device
            if not input.requires_grad:
                return
            backward_mat = mod.drtp_backward if hasattr(mod, "drtp_backward") else None
            error = signal
            n_outputs = error.shape[-1]
            if backward_mat is None:  # if the backward shortcut doesn't exist for this layer, create it
                channel_dim = input.shape[2] if len(input.shape) == 3 else input.shape[1]
                backward_mat = torch.empty((n_outputs, channel_dim), device=device)
                nn.init.normal_(backward_mat)
                mod.register_buffer("drtp_backward", backward_mat, persistent=True)
            layer_error = error @ backward_mat
            if len(input.shape) == 3:
                # the channel dimension is the third one for MLPMixer (even with Conv1D layers)
                layer_error = layer_error[:, None, :].expand(input.shape)
            elif len(input.shape) == 4:
                layer_error = layer_error[:, :, None, None].expand(input.shape)
            assert layer_error.shape == input.shape
            input.backward(layer_error)
            return input.detach()

        handles = []
        for module in model.modules():
            if isinstance(module, self.modules_to_hook):
                handle = module.register_forward_pre_hook(signal_hook)
                handles.append(handle)
            if isinstance(module, self.modules_to_detach):
                handle = module.register_forward_pre_hook(detach_hook)
                handles.append(handle)

        y_pred = model(x)
        l = loss_fn(y_pred, y)
        opt.zero_grad()
        l.backward()
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred