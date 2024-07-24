import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from binary_nn.models.common.binary.modules import Sign


def apply_binarization_parametrization(model: nn.Module, spared=None, apply_to_bias=False):
    if spared is None:
        spared = ()
    elif isinstance(spared, str):
        spared = [spared]
    spared = [model.get_submodule(x) if isinstance(x, str) else x for x in spared]

    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d)) and mod not in spared:
            register_parametrization(mod, "weight", Sign())
            if apply_to_bias:
                register_parametrization(mod, "bias", Sign())


def clip_weights_for_binary_layers(model: nn.Module, apply_to_bias=False):
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            if hasattr(mod, "parametrizations") and any(isinstance(x, Sign) for x in mod.parametrizations):
                w = mod.parametrizations.weight.original
                w.data.clip_(-1.0, 1.0)
                if apply_to_bias and mod.bias is not None:
                    b = mod.parametrizations.bias.original
                    b.data.clip_(-1.0, 1.0)
