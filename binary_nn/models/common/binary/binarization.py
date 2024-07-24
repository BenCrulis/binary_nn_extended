import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from binary_nn.models.common.binary.modules import Sign, SignUnsat
from binary_nn.models.common.utils import map_layers_generic


def apply_binarization_parametrization(model: nn.Module, spared=None, apply_to_bias=False):
    if spared is None:
        spared = ()
    elif isinstance(spared, str):
        spared = [spared]
    spared = [model.get_submodule(x) if isinstance(x, str) else
              model[x] if isinstance(x, int) else x
              for x in spared]

    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d)) and mod not in spared:
            register_parametrization(mod, "weight", Sign())
            if apply_to_bias:
                register_parametrization(mod, "bias", Sign())


def clip_weights_for_binary_layers(model: nn.Module, apply_to_bias=False):
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            if hasattr(mod, "parametrizations") \
                    and any(isinstance(x, (Sign, SignUnsat)) for x in mod.parametrizations["weight"]):
                w = mod.parametrizations.weight.original
                w.data.clip_(-1.0, 1.0)
                if apply_to_bias and mod.bias is not None:
                    b = mod.parametrizations.bias.original
                    b.data.clip_(-1.0, 1.0)


def replace_activations_to_sign(model: nn.Module, unsaturating=False):
    for name, child in model.named_children():
        if isinstance(child, (nn.ReLU, nn.Tanh)):
            model.register_module(name, SignUnsat() if unsaturating else Sign())
        else:
            replace_activations_to_sign(child, unsaturating)
