import torch
from torch import nn
from torch import autograd

from binary_nn.models.common.binary.utils import sign, signUnsat


class Sign(nn.Module):
    def forward(self, x):
        return sign(x)


class SignUnsat(nn.Module):
    def forward(self, x):
        return signUnsat(x)


class F_BinAct(torch.autograd.Function):
    """
    Taken from https://github.com/chrundle/biprop/blob/main/models/resnet_BinAct_v2.py
    """
    @staticmethod
    def forward(ctx, inp):
        # Save input for backward
        ctx.save_for_backward(inp)
        # Unscaled sign function
        return torch.sign(inp)

    @staticmethod
    def backward(ctx, grad_out):
        # Get input from saved ctx
        inp, = ctx.saved_tensors
        # Clone grad_out
        grad_input = grad_out.clone()
        # Gradient approximation from quadratic spline
        inp = torch.clamp(inp, min=-1.0, max=1.0)
        inp = 2 * (1 - torch.abs(inp))
        # Return gradient
        return grad_input * inp


class BiRealAct(nn.Module):
    """
    Taken from https://github.com/chrundle/biprop/blob/main/models/resnet_BinAct_v2.py
    """
    def __init__(self):
        super(BiRealAct, self).__init__()

    def forward(self, input):
        return F_BinAct.apply(input)


class GetQuantnet_binary(autograd.Function):
    """
    Adapted from https://github.com/chrundle/biprop/blob/main/utils/conv_type.py
    """

    @staticmethod
    def forward(ctx, scores, weights, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory. switched 0 and 1
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # Perform binary quantization of weights
        abs_wgt = torch.abs(weights.clone()) # Absolute value of original weights
        print(f"scores device: {scores.device}, weights device: {weights.device}", flush=True)
        q_weight = abs_wgt * out # Remove pruned weights
        num_unpruned = int(k * scores.numel()) # Number of unpruned weights
        alpha = torch.sum(q_weight) / num_unpruned # Compute alpha = || q_weight ||_1 / (number of unpruned weights)

        # Save absolute value of weights for backward
        ctx.save_for_backward(abs_wgt)

        # Return pruning mask with gain term alpha for binary weights
        return alpha * out

    @staticmethod
    def backward(ctx, g):
        # Get absolute value of weights from saved ctx
        abs_wgt, = ctx.saved_tensors
        # send the gradient g times abs_wgt on the backward pass
        return g * abs_wgt, None, None
