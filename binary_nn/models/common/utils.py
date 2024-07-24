import torch
from torch import nn


def count_parameters(model):
    n_params = 0
    for p in model.parameters():
        n_params += p.numel()
    return n_params


def count_neurons(model):
    n_params = 0
    for mod in model.modules():
        n = 0
        if isinstance(mod, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            n += mod.weight.shape[0]
        n_params += n
    return n_params


def recurse_modules(model: nn.Module, excepted=None):
    excepted = () if excepted is None else excepted
    if not isinstance(model, excepted):
        for mod in model.children():
            yield mod
            yield from recurse_modules(mod, excepted)
    return ()


def get_modules_in_order(model: nn.Module, sample_in):
    layers = []

    def hook(mod, _in, _out):
        layers.append(mod)

    handles = []
    for mod in model.modules():
        handles.append(mod.register_forward_hook(hook))

    with torch.no_grad():
        model(sample_in)

    for handle in handles:
        handle.remove()

    return layers


def get_layers_in_order(model: nn.Module, sample_in):
    return [x for x in get_modules_in_order(model, sample_in) if not isinstance(x, (nn.Sequential, nn.ModuleList, nn.ModuleDict))
            and x is not model]


def get_module_attributes(module: nn.Module):
    if not isinstance(module, nn.Module):
        return []
    l = []
    mem = []
    if hasattr(module, "__iter__"):
        for i, mod in enumerate(module):
            l.append((module, i, mod))
            mem.append(mod)
            l.extend(get_module_attributes(mod))

    for attr in dir(module):
        val = getattr(module, attr)
        if isinstance(val, nn.Module):
            if val not in mem:
                l.append((module, attr, val))
                mem.append(val)
                l.extend(get_module_attributes(val))
    return l


def map_layers_generic(module, mapping):
    module_attributes = get_module_attributes(module)

    for mod, attr, child in module_attributes:
        new_module = mapping(attr, child)
        if new_module is not None:
            if isinstance(attr, str):
                setattr(mod, attr, new_module)
            elif isinstance(attr, int):
                mod[attr] = new_module
        if isinstance(child, nn.Module):
            map_layers_generic(child, mapping)


def execute_model_from_layer(model: nn.Module, layer, input, sample):
    def inject_input(mod, _, output):
        output = output[0] if isinstance(output, tuple) else output
        final_shape = input.shape + (1,) * (len(output.shape) - 2)
        new_output = torch.broadcast_to(input.view(final_shape), (input.shape[0], *output.shape[1:]))
        return new_output

    if isinstance(layer, str):
        layer = model.get_submodule(layer)

    with torch.inference_mode():
        handle = layer.register_forward_hook(inject_input)
        out = model(sample)
        handle.remove()
        return out
