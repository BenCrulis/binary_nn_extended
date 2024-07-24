import torch
from torch import nn

from binary_nn.models.common.binary.utils import sign, signUnsat


class Sign(nn.Module):
    def forward(self, x):
        return sign(x)


class SignUnsat(nn.Module):
    def forward(self, x):
        return signUnsat(x)

