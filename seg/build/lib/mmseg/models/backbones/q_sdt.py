import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from torch.autograd import Function
import torch
import torch.nn as nn
# from mmdet.registry import MODELS
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader
from mmseg.registry import MODELS

import torch
import torchinfo
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from functools import partial

class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)


relu4 = ReLUX(thre=4)

import torch


class multispike(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, input, lens):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(4*inputs, self.lens)/4


class Multispike_att(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(4*inputs, self.lens)/2


# from visualizer import get_local
import torch
# import torchinfo
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
import torch.nn.functional as F
from functools import partial


from .quan_w import Conv2dLSQ,Conv1dLSQ

class BNAndPadLayer(nn.Module):
    def __init__(
            self,
            pad_pixels,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
    ):
        super(BNAndPadLayer, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features, eps, momentum, affine, track_running_stats
        )
        self.pad_pixels = pad_pixels

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            if self.bn.affine:
                pad_values = (
                        self.bn.bias.detach()
                        - self.bn.running_mean
                        * self.bn.weight.detach()
                        / torch.sqrt(self.bn.running_var + self.bn.eps)
                )
            else:
                pad_values = -self.bn.running_mean / torch.sqrt(
                    self.bn.running_var + self.bn.eps
                )
            output = F.pad(output, [self.pad_pixels] * 4)
            pad_values = pad_values.view(1, -1, 1, 1)
            output[:, :, 0: self.pad_pixels, :] = pad_values
            output[:, :, -self.pad_pixels:, :] = pad_values
            output[:, :, :, 0: self.pad_pixels] = pad_values
            output[:, :, :, -self.pad_pixels:] = pad_values
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def running_mean(self):
        return self.bn.running_mean

    @property
    def running_var(self):
        return self.bn.running_var

    @property
    def eps(self):
        return self.bn.eps


class RepConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        bias=False,
    ):
        super().__init__()
        # hidden_channel = in_channel
        conv1x1 = Conv2dLSQ(in_channel, in_channel, 1, 1, 0, bias=False, groups=1)
        bn = BNAndPadLayer(pad_pixels=1, num_features=in_channel)
        conv3x3 = nn.Sequential(
            Conv2dLSQ(in_channel, in_channel, 3, 1, 0, groups=in_channel, bias=False),
            Conv2dLSQ(in_channel, out_channel, 1, 1, 0, groups=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.body = nn.Sequential(conv1x1, bn, conv3x3)

    def forward(self, x):
        return self.body(x)

import numpy as np

class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(
        self,
        dim,
        expansion_ratio=2,
        act2_layer=nn.Identity,
        bias=False,
        kernel_size=7,
        padding=3,
    ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.lif1 = Multispike()
        self.pwconv1 = Conv2dLSQ(dim, med_channels, kernel_size=1, stride=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(med_channels)
        self.lif2 = Multispike()
        self.dwconv = Conv2dLSQ(
            med_channels,
            med_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=med_channels,
            bias=bias,
        )  # depthwise conv
        self.pwconv2 = Conv2dLSQ(med_channels, dim, kernel_size=1, stride=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.lif1(x)
        x = self.bn1(self.pwconv1(x.flatten(0, 1))).reshape(T, B, -1, H, W)
        x = self.lif2(x)
        x = self.dwconv(x.flatten(0, 1))
        x = self.bn2(self.pwconv2(x)).reshape(T, B, -1, H, W)
        return x


class MS_ConvBlock(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio=4.0,
    ):
        super().__init__()

        self.Conv = SepConv(dim=dim)
        # self.Conv = MHMC(dim=dim)

        self.lif1 = Multispike()
        self.conv1 = Conv2dLSQ(
            dim, dim * mlp_ratio, 3, 1, 1, bias=False
        )
        # self.conv1 = RepConv(dim, dim*mlp_ratio)
        self.bn1 = nn.BatchNorm2d(dim * mlp_ratio)  # 这里可以进行改进
        self.lif2 = Multispike()
        self.conv2 = Conv2dLSQ(
            dim * mlp_ratio, dim, 3, 1, 1, bias=False
        )
        # self.conv2 = RepConv(dim*mlp_ratio, dim)
        self.bn2 = nn.BatchNorm2d(dim)  # 这里可以进行改进

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.Conv(x) + x
        x_feat = x
        x = self.bn1(self.conv1(self.lif1(x).flatten(0, 1))).reshape(T, B, 4 * C, H, W)
        x = self.bn2(self.conv2(self.lif2(x).flatten(0, 1))).reshape(T, B, C, H, W)
        x = x_feat + x

        return x


class MS_MLP(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, drop=0.0, layer=0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = linear_unit(in_features, hidden_features)
        self.fc1_conv = Conv1dLSQ(in_features, hidden_features, 1, 1)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = Multispike()

        # self.fc2 = linear_unit(hidden_features, out_features)
        self.fc2_conv = Conv1dLSQ(
            hidden_features, out_features, 1, 1
        )
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = Multispike()
        # self.drop = nn.Dropout(0.1)

        self.c_hidden = hidden_features
        self.c_output = out_features

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        x = x.flatten(3)
        x = self.fc1_lif(x)
        x = self.fc1_conv(x.flatten(0, 1))
        x = self.fc1_bn(x).reshape(T, B, self.c_hidden, N).contiguous()

        x = self.fc2_lif(x)
        x = self.fc2_conv(x.flatten(0, 1))
        x = self.fc2_bn(x).reshape(T, B, C, H, W).contiguous()

        return x


class MS_Attention_RepConv_qkv_id(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            sr_ratio=1,
            Partial_channel=False,
    ):
        super().__init__()
        assert (
                dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.25
        self.Partial_channel = Partial_channel
        self.head_lif = Multispike()
        if self.Partial_channel:
            self.ratio = 0.25
        else:
            self.ratio = 1.0
        self.q_conv = nn.Sequential(RepConv(int(dim* self.ratio), int(dim* self.ratio), bias=False), nn.BatchNorm2d(int(dim* self.ratio)))

        self.k_conv = nn.Sequential(RepConv(int(dim* self.ratio), int(dim* self.ratio), bias=False), nn.BatchNorm2d(int(dim* self.ratio)))

        self.v_conv = nn.Sequential(RepConv(int(dim* self.ratio), int(dim* self.ratio), bias=False), nn.BatchNorm2d(int(dim* self.ratio)))

        self.q_lif = Multispike()

        self.k_lif = Multispike()

        self.v_lif = Multispike()

        self.attn_lif = Multispike_att()

        self.proj_conv = nn.Sequential(
            RepConv(int(dim* self.ratio),int(dim* self.ratio), bias=False), nn.BatchNorm2d(int(dim* self.ratio))
        )

    def forward(self, x):
        T, B, C, H, W = x.shape
        N = H * W
        if self.Partial_channel:
            num_samples = int(C*self.ratio)
            # print(C,num_samples)
            idx = torch.randperm(C)[:num_samples]
            x_2 = x[:,:,idx,:,:]
            remain_idx = [i for i in range(C) if i not in idx]
            remain_x = x[:,:,remain_idx,:,:]
            T, B, C, H, W = x_2.shape
            x = x_2
        x = self.head_lif(x)

        q = self.q_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        k = self.k_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)
        v = self.v_conv(x.flatten(0, 1)).reshape(T, B, C, H, W)

        q = self.q_lif(q).flatten(3)
        q = (
            q.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        k = self.k_lif(k).flatten(3)
        k = (
            k.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        v = self.v_lif(v).flatten(3)
        v = (
            v.transpose(-1, -2)
                .reshape(T, B, N, self.num_heads, C // self.num_heads)
                .permute(0, 1, 3, 2, 4)
                .contiguous()
        )

        x = k.transpose(-2, -1) @ v
        x = (q @ x) * self.scale

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x).reshape(T, B, C, H, W)
        x = x.reshape(T, B, C, H, W)
        x = x.flatten(0, 1)
        x = self.proj_conv(x).reshape(T, B, C, H, W)
        if self.Partial_channel:
            x = torch.cat((x, remain_x), dim=2)
        return x


class MS_Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            sr_ratio=1,
            init_values=1e-6,
    ):
        super().__init__()

        self.attn = MS_Attention_RepConv_qkv_id(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,

        )
        self.layer_scale1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.layer_scale2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MS_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(x) * self.layer_scale1.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = x + self.mlp(x) * self.layer_scale2.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return x


class MS_DownSampling(nn.Module):
    def __init__(
            self,
            in_channels=2,
            embed_dims=256,
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=True,
    ):
        super().__init__()

        self.encode_conv = Conv2dLSQ(
            in_channels,
            embed_dims,
            kernel_size,
            stride,
            padding,
        )

        self.encode_bn = nn.BatchNorm2d(embed_dims)
        if not first_layer:
            self.encode_lif = Multispike()

    def forward(self, x):
        T, B, _, _, _ = x.shape

        if hasattr(self, "encode_lif"):
            x = self.encode_lif(x)
        x = self.encode_conv(x.flatten(0, 1))
        _, _, H, W = x.shape
        x = self.encode_bn(x).reshape(T, B, -1, H, W).contiguous()

        return x


@MODELS.register_module()
class Spiking_vit_MetaFormer(BaseModule):
    def __init__(
            self,
            img_size_h=128,
            img_size_w=128,
            patch_size=16,
            in_channels=2,
            num_classes=1000,
            embed_dim=[64, 128, 256],
            num_heads=[1, 2, 4],
            mlp_ratios=[4, 4, 4],
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            norm_layer=nn.LayerNorm,
            depths=[6, 8, 6],
            sr_ratios=[8, 4, 2],
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.depths = depths
        self.T = 1
        # embed_dim = [64, 128, 256, 512]

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depths)
        ]  # stochastic depth decay rule

        self.downsample1_1 = MS_DownSampling(
            in_channels=in_channels,
            embed_dims=embed_dim[0] // 2,
            kernel_size=7,
            stride=2,
            padding=3,
            first_layer=True,
        )

        self.ConvBlock1_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0] // 2, mlp_ratio=mlp_ratios)]
        )

        self.downsample1_2 = MS_DownSampling(
            in_channels=embed_dim[0] // 2,
            embed_dims=embed_dim[0],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock1_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[0], mlp_ratio=mlp_ratios)]
        )

        self.downsample2 = MS_DownSampling(
            in_channels=embed_dim[0],
            embed_dims=embed_dim[1],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.ConvBlock2_1 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.ConvBlock2_2 = nn.ModuleList(
            [MS_ConvBlock(dim=embed_dim[1], mlp_ratio=mlp_ratios)]
        )

        self.downsample3 = MS_DownSampling(
            in_channels=embed_dim[1],
            embed_dims=embed_dim[2],
            kernel_size=3,
            stride=2,
            padding=1,
            first_layer=False,
        )

        self.block3 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[2],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(int(depths * 0.75))
            ]
        )

        self.downsample4 = MS_DownSampling(
            in_channels=embed_dim[2],
            embed_dims=embed_dim[3],
            kernel_size=3,
            stride=1,
            padding=1,
            first_layer=False,
        )

        self.block4 = nn.ModuleList(
            [
                MS_Block(
                    dim=embed_dim[3],
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratios,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios,
                )
                for j in range(int(depths * 0.25))
            ]
        )

        # self.lif = Multispike()
        # self.head = LinearLSQ(embed_dim[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                before_state_dict =ckpt['state_dict']
                import collections
                _state_dict = collections.OrderedDict()
                for k,v in before_state_dict.items():
                    _state_dict[k[9:]] = v
            self.load_state_dict(_state_dict, False)

    def forward_features(self, x):
        x = self.downsample1_1(x)
        for blk in self.ConvBlock1_1:
            x = blk(x)
        x1 = x
        x = self.downsample1_2(x)
        for blk in self.ConvBlock1_2:
            x = blk(x)
        x2 = x
        x = self.downsample2(x)
        for blk in self.ConvBlock2_1:
            x = blk(x)
        for blk in self.ConvBlock2_2:
            x = blk(x)
        x3 = x
        x = self.downsample3(x)
        for blk in self.block3:
            x = blk(x)

        x = self.downsample4(x)
        for blk in self.block4:
            x = blk(x)
        x4 = x
        return [x1,x2,x3,x4]
        # return x  # T,B,C,N

    from mmdet.utils import AvoidCUDAOOM
    @AvoidCUDAOOM.retry_if_cuda_oom
    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        # print('q_sdt')
        return x


# def spikformer_8_256_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[48, 96, 192, 256],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=500,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# def spikformer_8_384_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[96, 192, 384, 480],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# def spikformer_8_512_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[128, 256, 512, 640],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=500,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# def spikformer_12_512_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[128, 256, 512, 640],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=500,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=12,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# def spikformer_8_768_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[192, 384, 768, 960],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=500,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# def spikformer_8_15M_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[64, 128, 256, 360],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model

# def spikformer_8_10M_CAFormer(**kwargs):
#     model = Spiking_vit_MetaFormer(
#         img_size_h=224,
#         img_size_w=224,
#         patch_size=16,
#         embed_dim=[64, 128, 256, 384],
#         num_heads=8,
#         mlp_ratios=4,
#         in_channels=3,
#         num_classes=1000,
#         qkv_bias=False,
#         norm_layer=partial(nn.LayerNorm, eps=1e-6),
#         depths=8,
#         sr_ratios=1,
#         **kwargs,
#     )
#     return model


# from timm.models import create_model

# if __name__ == "__main__":
#     #     import torchsummary
#     # state_dict = torch.load('/home/ligq/qxr/bsnn/output/spikformer_8_15M_CAFormer_fulldataset/checkpoint-0.pth', map_location=torch.device('cuda'))
#     model = spikformer_8_15M_CAFormer()
#     # msg = model.load_state_dict(state_dict["model"], strict=False)
#     # print(msg)
#     x = torch.randn(1, 3, 224, 224)
#     print(model(x).shape)
#     print("Parameter numbers: {}".format(
#         sum(p.numel() for p in model.parameters())))
#     # torchsummary.summary(model, (2, 3, 224, 224))


