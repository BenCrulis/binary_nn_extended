import math

import torch
from torch import nn
from torch.nn.init import calculate_gain
from torch.optim import Optimizer
from torch.nn.utils.parametrize import register_parametrization

from binary_nn.models.common.binary.modules import GetQuantnet_binary
from utils.configuration import ConfigurableMixin


def _calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = shape[1]
    num_output_fmaps = shape[0]
    receptive_field_size = 1
    if len(shape) > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(shape, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == 'fan_in' else fan_out


def compute_bound(shape):
    fan = _calculate_correct_fan(shape, mode="fan_in")
    # gain = calculate_gain(nonlinearity="Tanh", a=math.sqrt(5))
    gain = math.sqrt(5)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return bound


class SeededRandomWeights(nn.Module):
    def __init__(self, shape, binarize=False, seed=None, device=None, bound=None):
        super().__init__()
        self.shape = shape
        self.device = device
        self.binarize = binarize
        if seed is None:
            seed = torch.randint(2**31, size=(1,)).item()
        self.seed = seed
        self.bound = compute_bound(self.shape) if bound is None else bound

    def get_float_weights(self):
        state = torch.get_rng_state()
        torch.manual_seed(self.seed)
        new_w = torch.rand(size=self.shape, device=self.device) * (2.0 * self.bound) - self.bound
        torch.set_rng_state(state)
        return new_w

    def forward(self, w):
        new_w = self.get_float_weights()
        if self.binarize:
            new_w = (new_w >= 0.0) * 2.0 - 1.0
        return new_w


class MaskedWeights(nn.Module):
    def __init__(self, shape, prune_rate, device=None, constant_init=None):
        super().__init__()
        self.shape = shape
        self.device = device
        self.scores = nn.Parameter(torch.ones(shape), requires_grad=True)
        self.prune_rate = prune_rate

        if constant_init is None:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        else:
            self.scores.data *= constant_init

    @property
    def clamped_scores(self):
        return torch.abs(self.scores)

    def forward(self, w):
        quantnet = GetQuantnet_binary.apply(self.clamped_scores, w, self.prune_rate)
        return quantnet * torch.sign(w)


def set_seeded_random_weights(module: nn.Module, binarize=False):
    parametrizations = [None, None]
    if hasattr(module, "weight"):
        if module.weight is not None:
            w = module.weight
            parametrization = SeededRandomWeights(module.weight.shape, binarize=binarize, device=w.device)
            register_parametrization(module, "weight", parametrization)
            parametrizations[0] = parametrization
            module.parametrizations["weight"].original = None
    if hasattr(module, "bias"):
        if module.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in)
            parametrization = SeededRandomWeights(module.bias.shape, binarize=False, device=w.device,
                                                  bound=bound)
            register_parametrization(module, "bias", parametrization)
            parametrizations[1] = parametrization
            module.parametrizations["bias"].original = None
    return parametrizations


def set_scores(module: nn.Module):
    if hasattr(module, "weight"):
        if module.weight is not None and not hasattr(module, "weight_scores"):
            module.scores = nn.Parameter(torch.Tensor(module.weight.shape))
            nn.init.kaiming_uniform_(module.scores, a=math.sqrt(5))


class Biprop(ConfigurableMixin):
    def __init__(self, model, pruning_rate, modules_to_hook=(nn.Linear, nn.Conv2d, nn.Conv1d)):
        super().__init__()
        self.modules_to_hook = modules_to_hook
        self.pruning_rate = pruning_rate
        self._init_model(model)

    def config(self):
        return {"Biprop pruning rate": self.pruning_rate,}

    def _init_model(self, model: nn.Module):
        for mod in model.modules():
            if isinstance(mod, self.modules_to_hook):
                if mod.bias is not None:
                    mod.bias.requires_grad = False
                set_seeded_random_weights(mod, binarize=False)
                register_parametrization(mod, "weight", MaskedWeights(mod.weight.shape, 0.2))

