import torch
from torch import nn
from torch.optim import Optimizer

from utils.configuration import ConfigurableMixin


class DFA(ConfigurableMixin):
    def __init__(self, output_layer, modules_to_hook=(nn.Linear, nn.Conv2d)):
        super().__init__()
        self.output_layer = output_layer
        self.modules_to_hook = modules_to_hook
        self.error = None

    def __call__(self, model, x, y, opt: Optimizer, loss_fn):
        s = self

        model.train()

        handles = []
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
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
                        backward_mat = m.dfa_backward if hasattr(m, "dfa_backward") else None
                        error = s.error
                        n_outputs = error.shape[-1]
                        if backward_mat is None:  # if the backward shortcut doesn't exist for this layer, create it
                            backward_mat = torch.empty((n_outputs, grad_input.shape[1]), device=device)
                            nn.init.normal_(backward_mat)
                            m.register_buffer("dfa_backward", backward_mat, persistent=True)
                        layer_error = error @ backward_mat
                        if layer_error.shape != grad_input.shape:
                            layer_error = layer_error[..., None, None].repeat((1, 1, *grad_input.shape[-2:]))
                        return layer_error,
                hook = Hook()
                handle = module.register_forward_hook(hook.forward_hook)
                handles.append(handle)
                handle = module.register_full_backward_hook(hook.backward_hook)
                handles.append(handle)

        y_pred = model(x)

        output = y_pred.detach().clone()
        output.requires_grad = True
        l = loss_fn(output, y)
        l.backward()  # a second .backward() will be called automatically on strategy.loss

        self.error = output.grad

        opt.zero_grad()
        y_pred.backward(self.error)
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred



