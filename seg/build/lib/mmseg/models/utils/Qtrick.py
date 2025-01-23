import torch
import torch.nn as nn
class Quant(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, i, min_value=0, max_value=4):
        ctx.min = min_value
        ctx.max = max_value
        ctx.save_for_backward(i)
        return torch.round(torch.clamp(i, min=min_value, max=max_value))

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        i, = ctx.saved_tensors
        grad_input[i < ctx.min] = 0
        grad_input[i > ctx.max] = 0
        return grad_input, None, None


class MultiSpike_norm(nn.Module):
    def __init__(
            self,
            Norm=4,
    ):
        super().__init__()
        self.spike = Quant()
        self.T = Norm
        # if Vth_learnable == False:
        #     self.Vth = Vth
        # else:
        #     self.register_parameter("Vth", nn.Parameter(torch.tensor([1.0])))

    def __repr__(self):
        return f"MultiSpike_norm(Norm={self.T})"

    def forward(self, x):  # B C H W
        return self.spike.apply(x) / self.T