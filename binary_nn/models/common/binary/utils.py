from typing import Any
import torch


class SignSatFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x) -> Any:
        ctx.save_for_backward(x)
        return torch.where(x > 0.0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        inp = ctx.saved_tensors[0]
        return grad_output * (torch.abs(inp) <= 1.0)
sign = SignSatFun.apply



class SignUnSatFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x) -> Any:
        ctx.save_for_backward(x)
        return torch.where(x > 0.0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        return grad_output
signUnsat = SignUnSatFun.apply


class StepSatFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x) -> Any:
        ctx.save_for_backward(x)
        return torch.where(x > 0.0, torch.ones_like(x), torch.zeros_like(x))

    @staticmethod
    def backward(ctx: Any, grad_output: Any) -> Any:
        inp = ctx.saved_tensors
        return grad_output * (torch.abs(inp) <= 1.0)
step = StepSatFun.apply

