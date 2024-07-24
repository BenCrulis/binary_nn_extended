from mlp_mixer_pytorch.mlp_mixer_pytorch import PreNormResidual
from torch import nn
from torchvision.models.mobilenetv2 import InvertedResidual


class PreNorm(nn.Module):
    def __init__(self, norm, fn):
        super().__init__()
        self.fn = fn
        self.norm = norm

    def forward(self, x):
        return self.fn(self.norm(x))


def remove_skips(model: nn.Module):
    """
    Warning: this is only supported for MLPMixer, MobileNetV2 and models that don't have skip connexions
    :param model: model to modify
    :return:
    """

    for name, m in model.named_children():
        if isinstance(m, PreNormResidual):
            new_m = PreNorm(m.norm, m.fn)
            model.add_module(name, new_m)
        elif isinstance(m, InvertedResidual):
            m.use_res_connect = False
        else:
            remove_skips(m)
