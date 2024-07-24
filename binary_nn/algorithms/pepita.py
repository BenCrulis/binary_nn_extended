import torch
from mlp_mixer_pytorch.mlp_mixer_pytorch import PreNormResidual
from torch import nn
from torch.optim import Optimizer

from torch.nn.grad import conv1d_weight, conv2d_weight, conv3d_weight

from torchvision.models.mobilenetv2 import InvertedResidual

from binary_nn.models.common.binary.modules import Sign
from utils.configuration import ConfigurableMixin

from binary_nn.utils import hsic


def parse_module_path(path, model):
    if isinstance(path, int):
        if path < 0:
            return str(len(model) + path)
        else:
            return str(path)
    return path


WEIGHT_MODS = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)


class Pepita(ConfigurableMixin):
    def __init__(self, num_classes, output_layer, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d), gain=1.0):
        super().__init__()
        self.n_classes = num_classes
        self.output_layer = output_layer
        self.gain = gain
        self.modules_to_hook = modules_to_hook
        self.modules_to_detach = (PreNormResidual, InvertedResidual)
        self.bw = None
        self.input_shape = None

    def config(self):
        return {"pepita-gain": self.gain}

    def _init_bw(self, error_shape, device=None):
        self.bw = torch.empty(error_shape, device=device)
        nn.init.kaiming_normal_(self.bw, a=self.gain, mode="fan_in", nonlinearity="linear")
        pass

    def __call__(self, model: nn.Module, x, y, opt: Optimizer, loss_fn):
        self.input_shape = x.shape[1:]
        model.eval()

        device = x.device
        eye = torch.eye(self.n_classes, device=device)
        signal = eye[y]
        signal -= signal.mean(1)[:, None]

        handles = []

        parametrization_mods = []
        for m in model.modules():
            if hasattr(m, "parametrizations"):
                for pml in m.parametrizations.values():
                    parametrization_mods.extend(pml)

        blocks = []
        current_block = None

        def structure_hook(mod, input):
            torch.set_grad_enabled(False)
            nonlocal current_block
            if current_block is None:
                if isinstance(mod, WEIGHT_MODS):
                    current_block = [mod]
                    return
            elif len(current_block) == 1:
                if isinstance(mod, WEIGHT_MODS):
                    current_block = [current_block[0], current_block[0]]
                    blocks.append(current_block)
                    current_block = [mod]
                    return
                elif isinstance(mod, (Sign, nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.Softmax, nn.Softplus, nn.SELU,
                                    nn.ELU, nn.CELU, nn.GELU)):
                    current_block.append(mod)
                    blocks.append(current_block)
                    current_block = None

        def activate_grad_hook(mod, input):
            torch.set_grad_enabled(True)

        output_layer = model.get_submodule(parse_module_path(self.output_layer, model))

        for m in model.modules():
            if m not in parametrization_mods and m is not output_layer:
                handle = m.register_forward_pre_hook(structure_hook)
                handles.append(handle)

        handles.append(output_layer.register_forward_pre_hook(activate_grad_hook))

        # first pass over the data and compute the output error
        opt.zero_grad()
        y_pred = model(x)
        y_pred.retain_grad()
        l = loss_fn(y_pred, y)
        l.backward()
        error: torch.Tensor = y_pred.grad

        if self.bw is None:
            self._init_bw((error.flatten(1).shape[1], x.flatten(1).shape[1]), device=y_pred.device)

        input_error = error.flatten(1) @ self.bw
        perturbed_input = x + input_error.view(x.shape)
        new_input = torch.cat([x, perturbed_input])  #  to avoid storing the activations, we recompute them with the error

        # clears up hooks that discover the model structure
        for handle in handles:
            handle.remove()
        handles.clear()

        class Hook():
            def hook_weights(self, mod, input, output):
                input = input[0] if isinstance(input, tuple) else input
                self.input = input
                self.mod = mod

            def hook_act(self, mod, input, output):
                output = output[0] if isinstance(output, tuple) else output
                half = len(self.input) // 2
                h = output[:half]
                h_err = output[half:]
                mod_input = self.input[:half]
                self.input = None
                err = h - h_err
                k = 1.0
                if isinstance(self.mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    if isinstance(self.mod, nn.Conv1d):
                        bw_fn = conv1d_weight
                        k = output.shape[-1]
                    elif isinstance(self.mod, nn.Conv2d):
                        bw_fn = conv2d_weight
                        k = output.shape[-1] * output.shape[-2]
                    elif isinstance(self.mod, nn.Conv3d):
                        bw_fn = conv3d_weight
                    w_grad = bw_fn(mod_input, self.mod.weight.shape, err, self.mod.stride, self.mod.padding, self.mod.dilation, self.mod.groups)
                else:
                    if len(err.shape) == 3:
                        k = err.shape[-1]
                    w_grad = err.view((-1, err.shape[-1])).T @ mod_input.view((-1, mod_input.shape[-1]))
                w_grad = w_grad / k
                w = self.mod.parametrizations["weight"].original if hasattr(self.mod, "parametrizations") else self.mod.weight
                w.grad = w_grad
                pass

        for weight_mod, act in blocks:
            hook = Hook()
            handle = weight_mod.register_forward_hook(hook.hook_weights)
            handles.append(handle)
            handle = act.register_forward_hook(hook.hook_act)
            handles.append(handle)
        model.train()
        with torch.no_grad():
            _ = model(new_input)
        opt.step()

        for handle in handles:
            handle.remove()

        return l, y_pred