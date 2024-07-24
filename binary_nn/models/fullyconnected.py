import torch
from torch import nn


class FullyConnected(nn.Module):
    def __init__(self, n_classes, color_input=False, with_batchnorm=False, bias=False, act=nn.Tanh):
        super().__init__()
        self.n_classes = n_classes
        self.color_input = color_input
        n_input = 28*28*3 if color_input else 28*28

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_input, 700, bias=bias),
            *((nn.BatchNorm1d(700),) if with_batchnorm else ()),
            act(),
            nn.Linear(700, 500, bias=bias),
            *((nn.BatchNorm1d(700),) if with_batchnorm else ()),
            act(),
            nn.Linear(500, 300, bias=bias),
            *((nn.BatchNorm1d(700),) if with_batchnorm else ()),
            act(),
            nn.Linear(300, 200, bias=bias),
            *((nn.BatchNorm1d(700),) if with_batchnorm else ()),
            act()
        )

        self.output = nn.Linear(200, n_classes, bias=bias)

    def forward(self, x):
        return self.output(self.layers(x))


