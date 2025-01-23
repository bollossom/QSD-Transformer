"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
import math
from torch.nn.parameter import Parameter
from ._quan_base_plus import _Conv2dQ, Qmodes, _LinearQ,_Conv1dQ
# , _LinearQ_v2,_Conv1dQ

# __all__ = ['Conv2dLSQ', 'LinearLSQ', 'ActLSQ']


# class FunLSQ(torch.autograd.Function):
#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def forward(ctx, weight, alpha, g, Qn, Qp):
#         assert alpha > 0, 'alpha = {}'.format(alpha)
#         ctx.save_for_backward(weight, alpha)
#         ctx.other = g, Qn, Qp
#         q_w = (weight / alpha).round().clamp(Qn, Qp)
#         w_q = q_w * alpha
#         return w_q

#     @staticmethod
#     @torch.cuda.amp.custom_fwd
#     def backward(ctx, grad_weight):
#         weight, alpha = ctx.saved_tensors
#         g, Qn, Qp = ctx.other
#         q_w = weight / alpha
#         indicate_small = (q_w < Qn).float()
#         indicate_big = (q_w > Qp).float()
#         # indicate_middle = torch.ones(indicate_small.shape).to(indicate_small.device) - indicate_small - indicate_big
#         indicate_middle = 1.0 - indicate_small - indicate_big  # Thanks to @haolibai
#         grad_alpha = ((indicate_small * Qn + indicate_big * Qp + indicate_middle * (
#                 -q_w + q_w.round())) * grad_weight * g).sum().unsqueeze(dim=0)
#         grad_weight = indicate_middle * grad_weight
#         # The following operation can make sure that alpha is always greater than zero in any case and can also
#         # suppress the update speed of alpha. (Personal understanding)
#         # grad_alpha.clamp_(-alpha.item(), alpha.item())  # FYI
#         return grad_weight, grad_alpha, None, None, None


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, mode=Qmodes.kernel_wise, **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        # self.act = ActLSQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.

        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)
        # print(alpha.shape)
        # print(self.weight.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha

        # x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv2d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

def count_unique(arr):
    unique, counts = torch.unique(arr, return_counts=True)
    print("不同的值:", unique)
    print("对应的个数:", counts)
class Conv1dLSQ(_Conv1dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=4, mode=Qmodes.kernel_wise, **kwargs):
        super(Conv1dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits=nbits_w, mode=mode)
        # self.act = ActLSQ(in_features=in_channels, nbits_a=nbits_w)

    def forward(self, x):
        if self.alpha is None:
            return F.conv1d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        # w_reshape = self.weight.reshape([self.weight.shape[0], -1]).transpose(0, 1)
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() * 2)
            self.init_state.fill_(1)
        """  
        Implementation according to paper. 
        Feels wrong ...
        When we initialize the alpha as a big number (e.g., self.weight.abs().max() * 2), 
        the clamp function can be skipped.
        Then we get w_q = w / alpha * alpha = w, and $\frac{\partial w_q}{\partial \alpha} = 0$
        As a result, I don't think the pseudo-code in the paper echoes the formula.

        Please see jupyter/STE_LSQ.ipynb fo detailed comparison.
        """
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1: 31GB GPU memory (AlexNet w4a4 bs 2048) 17min/epoch
        alpha = grad_scale(self.alpha, g)

        # print(alpha.shape)
        alpha = alpha.unsqueeze(1).unsqueeze(2)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        # x = self.act(x)
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2: 25GB GPU memory (AlexNet w4a4 bs 2048) 32min/epoch
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        # wq = y.transpose(0, 1).reshape(self.weight.shape).detach() + self.weight - self.weight.detach()
        return F.conv1d(x, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
# class LinearLSQ_v2(_LinearQ_v2):
#     def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
#         super(LinearLSQ_v2, self).__init__(in_features=in_features,
#                                            out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
#         self.act = ActLSQ(in_features=in_features, nbits_a=nbits_w)

#     def forward(self, x):
#         if self.alpha is None:
#             return F.linear(x, self.weight, self.bias)
#         Qn = -2 ** (self.nbits - 1)
#         Qp = 2 ** (self.nbits - 1) - 1
#         if self.training and self.init_state == 0:
#             self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#             self.beta.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#             # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
#             self.init_state.fill_(1)
#         g = 1.0 / math.sqrt(self.weight.numel() * Qp)

#         # Method1:
#         alpha = grad_scale(self.alpha, g)
#         beta = grad_scale(self.beta, g)
#         alpha = alpha.unsqueeze(1)
#         beta = beta.unsqueeze(0)
#         scale = alpha @ beta  # ).transpose(0,1)
#         # print(scale.shape)
#         w_q = round_pass((self.weight / scale).clamp(Qn, Qp)) * scale
#         x = self.act(x)
#         # w = self.weight / alpha
#         # w = w.clamp(Qn, Qp)
#         # q_w = round_pass(w)
#         # w_q = q_w * alpha

#         # Method2:
#         # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
#         return F.linear(x, w_q, self.bias)


class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w, mode=Qmodes.kernel_wise)
        # self.act = ActLSQ(in_features=in_features, nbits_a=nbits_w)

    def qw(self, weight):
        Qn = -2 ** (self.nbits - 1)
        Qp = 2 ** (self.nbits - 1) - 1
        if self.training and self.init_state == 0:
            self.alpha.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
            # self.alpha.data.copy_(self.weight.abs().max() / 2 ** (self.nbits - 1))
            self.init_state.fill_(1)
        g = 1.0 / math.sqrt(self.weight.numel() * Qp)

        # Method1:
        alpha = grad_scale(self.alpha, g)
        alpha = alpha.unsqueeze(1)
        w_q = round_pass((self.weight / alpha).clamp(Qn, Qp)) * alpha
        return w_q

    def forward(self, x):
        if self.alpha is None:
            return F.linear(x, self.weight, self.bias)

        w_q = self.qw(self.weight)

        # x = self.act(x)
        # w = self.weight / alpha
        # w = w.clamp(Qn, Qp)
        # q_w = round_pass(w)
        # w_q = q_w * alpha

        # Method2:
        # w_q = FunLSQ.apply(self.weight, self.alpha, g, Qn, Qp)
        return F.linear(x, w_q, self.bias)


# class LinearMCN(Linear):
#     def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
#         super(LinearMCN, self).__init__(in_features=in_features,
#                                         out_features=out_features, bias=bias)
#         self.act = ActLSQ(in_features=in_features, nbits_a=nbits_w)
#         self.MCF_Function = MCF_Function.apply
#         self.nbits = nbits_w
#         self.generate_MFilters()
#         self.out_features = out_features

#     # self.in_features = in_features

#     def generate_MFilters(self):
#         self.MFilters = Parameter(torch.randn(self.out_features, 1))

#     # self.beta = Parameter(torch.randn(1, self.in_features))

#     def forward(self, x):
#         if self.MFilters is None:
#             return F.linear(x, self.weight, self.bias)
#         Qn = -2 ** (self.nbits - 1)
#         Qp = 2 ** (self.nbits - 1) - 1
#         bin = 0.02
#         self.MFilters.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))
#         # self.beta.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp))

#         x = self.act(x)
#         # print(self.weight.shape)
#         # alpha = (self.beta @ self.MFilters).transpose(0,1)
#         weight_bin = (self.weight / self.MFilters).clamp(Qn, Qp).round()  # * bin
#         w_q = self.MCF_Function(self.weight, self.MFilters, weight_bin)

#         return F.linear(x, w_q, self.bias)


# class MCF_Function(torch.autograd.Function):

#     @staticmethod
#     def forward(ctx, weight, MScale, weight_bin):
#         weight = weight_bin * MScale

#         ctx.save_for_backward(weight, MScale, weight_bin)
#         return weight

#     @staticmethod
#     def backward(ctx, gradOutput):
#         weight, MScale, weight_bin = ctx.saved_tensors

#         nChannel = MScale.size()[0]
#         nOutputPlane = gradOutput.size()[0]
#         nInputPlane = gradOutput.size()[1]
#         para_loss = 0.0001

#         target1 = para_loss * (weight - weight_bin * MScale)
#         gradWeight = target1 + (gradOutput * MScale)

#         target2 = (weight - weight_bin * MScale) * weight_bin
#         grad_h2_sum = torch.sum(gradOutput * weight, keepdim=True, dim=1)

#         grad_target2 = torch.sum(target2, keepdim=True, dim=1)

#         gradMScale = grad_h2_sum - para_loss * grad_target2
#         gradweight_bin = gradOutput * 0
#         return gradWeight, gradMScale, gradweight_bin


# class ActLSQ(_ActQ):
#     def __init__(self, in_features, nbits_a=4, mode=Qmodes.kernel_wise, **kwargs):
#         super(ActLSQ, self).__init__(in_features=in_features, nbits=nbits_a, mode=mode)
#         # print(self.alpha.shape, self.zero_point.shape)

#     def forward(self, x):
#         return x
        # if self.alpha is None:
        #     return x
        #
        # if self.training and self.init_state == 0:
        #     # The init alpha for activation is very very important as the experimental results shows.
        #     # Please select a init_rate for activation.
        #     # self.alpha.data.copy_(x.max() / 2 ** (self.nbits - 1) * self.init_rate)
        #     if x.min() < -1e-5:
        #         self.signed.data.fill_(1)
        #     # print(self.signed)
        #     if self.signed == 1:
        #         Qn = -2 ** (self.nbits - 1)
        #         Qp = 2 ** (self.nbits - 1) - 1
        #     else:
        #         Qn = 0
        #         Qp = 2 ** self.nbits - 1
        #     self.alpha.data.copy_(2 * x.abs().mean() / math.sqrt(Qp))
        #     self.zero_point.data.copy_(
        #         self.zero_point.data * 0.9 + 0.1 * (torch.min(x.detach()) - self.alpha.data * Qn))
        #     self.init_state.fill_(1)
        #
        # # print(self.signed)
        #
        # if self.signed == 1:
        #     Qn = -2 ** (self.nbits - 1)
        #     Qp = 2 ** (self.nbits - 1) - 1
        # else:
        #     Qn = 0
        #     Qp = 2 ** self.nbits - 1
        #
        # g = 1.0 / math.sqrt(x.numel() * Qp)
        #
        # # Method1:
        # zero_point = (self.zero_point.round() - self.zero_point).detach() + self.zero_point
        # alpha = grad_scale(self.alpha, g)
        # zero_point = grad_scale(zero_point, g)
        # # x = round_pass((x / alpha).clamp(Qn, Qp)) * alpha
        # if len(x.shape) == 2:
        #     alpha = alpha.unsqueeze(0)
        #     zero_point = zero_point.unsqueeze(0)
        # elif len(x.shape) == 3:
        #     if x.shape[0] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(1).unsqueeze(2)
        #         zero_point = zero_point.unsqueeze(1).unsqueeze(2)
        #     elif x.shape[1] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(0).unsqueeze(2)
        #         zero_point = zero_point.unsqueeze(0).unsqueeze(2)
        #     elif x.shape[2] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(0).unsqueeze(0)
        #         zero_point = zero_point.unsqueeze(0).unsqueeze(0)
        # elif len(x.shape) == 4:
        #     # A, B, C, D = x.shape
        #     if x.shape[0] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #         zero_point = zero_point.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        #     elif x.shape[1] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #         zero_point = zero_point.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #     elif x.shape[2] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        #         zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        #     elif x.shape[3] == alpha.shape[0]:
        #         alpha = alpha.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #         zero_point = zero_point.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        #
        # # print(alpha.shape, zero_point.shape)
        # # print(x.shape)
        # x = round_pass((x / alpha + zero_point).clamp(Qn, Qp))
        # x = (x - zero_point) * alpha
        # # x = x.clamp(Qn, Qp)
        # # q_x = round_pass(x)
        # # x_q = q_x * alpha
        #
        # # Method2:
        # # x_q = FunLSQ.apply(x, self.alpha, g, Qn, Qp)
        # return x