# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial
import torch.nn.functional as F
import torch
import torch.nn as nn
# from util.logging import master_print as print

from video_vit import Attention,Block,PatchEmbed
from model.pos_embed import get_2d_sincos_pos_embed

from timm.models.vision_transformer import DropPath, Mlp


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU())

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class BasicBlock_us(nn.Module):
    def __init__(self, inplanes, upsamp=1):
        super(BasicBlock_us, self).__init__()
        planes = int(inplanes / upsamp)  # assumes integer result, fix later
        self.conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsamp = upsamp
        self.couple = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, padding=1, stride=upsamp, output_padding=1)
        self.bnc = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = self.couple(x)
        residual = self.bnc(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class VisionTransformer(nn.Module):
    """Vision Transformer with support for global average pooling"""

    def __init__(
            self,
            num_frames,
            t_patch_size,
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            no_qkv_bias=True,
            qk_scale=None,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.5,
            norm_layer=nn.LayerNorm,
            # dropout=0.5,  # 0.5
            sep_pos_embed=True,
            cls_embed=False,
            **kwargs,
    ):
        super().__init__()
        # print(locals())
        #
        self.sep_pos_embed = sep_pos_embed
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim, num_frames, t_patch_size
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.cls_embed = cls_embed

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim), requires_grad=True
            )  # fixed or not?
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                                cls_token=False)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop_path=dpr[i],
                    attn_func=partial(
                        Attention,
                        input_size=self.patch_embed.input_size,
                    ),
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        self.cls_seg = nn.Sequential(
            nn.Conv2d(256, 2, kernel_size=3, padding=1),
        )
        # self.sm = nn.Softmax(dim=1)
        # self.sm = nn.Sigmoid()
        self.sm = nn.LogSoftmax(dim=1)
        # torch.nn.init.normal_(self.head.weight, std=0.02)
        # self.linear = nn.Sequential(
        #     nn.Linear(768, 512),
        #     nn.GELU(),
        #     nn.Linear(512,192))
        # self.upernet = UPerNet(num_classes=2)
        self.decoder = FPNHEAD()
        # self.pos_drop = nn.Dropout(p=drop_rate)
        # 1024直接32*32
        # self.conv0 = nn.Sequential(
        #     nn.Conv2d(768, 512, 1, 1),
        #     nn.GroupNorm(32, 512),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(512, 256, 4, 4)  # 256,128,128
        # )
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(768, 512, 1, 1),
        #     nn.GroupNorm(32, 512),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(512, 512, 2, 2)  # 512,64,64
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(768, 1024, 1, 1),
        #     nn.GroupNorm(32, 1024),
        #     nn.GELU(),
        #       # 1024,32,32
        # )
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(768, 2048, 1, 1),
        #     nn.GroupNorm(64, 2048),
        #     nn.GELU(),
        #     nn.MaxPool2d(2, stride=2)  # 2048,16,16
        # )
        self.conv0 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 256, 8, 8),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(768, 512, 1, 1),
            nn.GroupNorm(32, 512),
            nn.GELU(),
            nn.ConvTranspose2d(512, 512, 4, 4),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(768, 1024, 1, 1),
            nn.GroupNorm(32, 1024),
            nn.GELU(),
            nn.ConvTranspose2d(1024, 1024, 2, 2),  # 2048, 16, 16
            # nn.Dropout(0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(768, 2048, 1, 1),
            nn.GroupNorm(32, 2048),
            nn.GELU(),
            # nn.Dropout(0.5)
            # 2048, 16, 16
        )
        self.fc = nn.Sequential(
            nn.Linear(4, 1))

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N = x.shape[0]
        N, T, H, W, p, u, t, h, w = (N, 12, 128, 128, 8, 3, 4, 16, 16)

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            "cls_token",
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_class",
        }

    def forward(self, x1, x2):
        # embed patches
        # x = torch.cat([x1, x2], dim=1)
        # x = x1 - x2
        # x = x[:, :-1, :, :]  # 切片处理数据维度
        x = torch.unsqueeze(x1, dim=1)
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: temporal; L: spatial
        # x = x.squeeze()
        x = x.view([N, T * L, C])

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed

        # reshape to [N, T, L, C] or [N, T*L, C]
        requires_t_shape = (
                len(self.blocks) > 0  # support empty decoder
                and hasattr(self.blocks[0].attn, "requires_t_shape")
                and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])
        # q = 0
        for blk in self.blocks:
            x = blk(x)
        # B = x.shape[0]
        # x = self.linear(x)

        B = x.shape[0]
        # x = x.view([N, T*L, C])
        # xx1 = self.unpatchify(x)
        # xx1 = x.reshape(B, 32, 32, 768).permute(0, 3, 1, 2).contiguous()
        # xx1 = x.view([N, T, L, C])
        xx1 = x
        # m = torch.split(xx1, 256, dim=1)
        # xx1 = torch.stack(m, dim=1)
        # image2
        x = x2
        # x = x[:, :-1, :, :]  # 切片处理数据维度
        x = torch.unsqueeze(x, dim=1)
        x = self.patch_embed(x)
        N, T, L, C = x.shape  # T: temporal; L: spatial

        x = x.view([N, T * L, C])
        # x = x.squeeze()

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            pos_embed = self.pos_embed[:, :, :]
        x = x + pos_embed

        # reshape to [N, T, L, C] or [N, T*L, C]
        requires_t_shape = (
                len(self.blocks) > 0  # support empty decoder
                and hasattr(self.blocks[0].attn, "requires_t_shape")
                and self.blocks[0].attn.requires_t_shape
        )
        if requires_t_shape:
            x = x.view([N, T, L, C])
        # q = 0
        for blk in self.blocks:
            x = blk(x)
        # x = self.linear(x)

        B = x.shape[0]
        # xx2 = self.unpatchify(x)
        # xx2 = x.reshape(B, 32, 32, 768).permute(0, 3, 1, 2).contiguous()
        # x = x.view([N, T*L, C])
        # xx2 = x.view([N, T, L, C])
        # xx2 = x.view([N, T, L, C])
        xx2 = x
        # xx1 = xx1[:, 1:, :]
        # xx2 = xx2[:, 1:, :]
        xx1 = xx1.view([N, T, L, C])
        xx2 = xx2.view([N, T, L, C])
        # m = torch.split(xx2, 256, dim=1)
        # xx2 = torch.stack(m, dim=1)

        # xx1 = xx1.squeeze()
        # xx2 = xx2.squeeze()
        # xx1 = x1.view([N, T, L, C])
        # xx2 = x2.view([N, T, L, C])
        # xx1 = xx1.permute(0,2,1,3)
        # xx2 = xx2.permute(0, 2, 1, 3)
        xx1 = xx1.permute(0, 2, 3, 1)
        xx2 = xx2.permute(0, 2, 3, 1)
        if xx1.shape[3] == 4:
            xx1 = self.fc(xx1)
            xx2 = self.fc(xx2)

        # xx1 = xx1[:,:,:,:].mean(dim=1,keepdim=False)
        # xx1 = xx1[:,:,:,:].sum(dim=1)
        # val1, idx   = torch.max(xx1, dim=1)
        # xx1 = val1
        # val2, idx = torch.max(xx2, dim=1)
        # xx2 = val2
        # xx2 = xx2[:, :, :, :].mean(dim=1)
        # xx2 = xx2[:, :, :, :].sum(dim=1)
        x = xx1 - xx2  # B, 256, 768
        # x = xx1
        x = x.reshape(B, 16, 16, 768).permute(0, 3, 1, 2).contiguous()
        # x = x.squeeze()

        m = {}

        m[0] = self.conv0(x)  # 256,128,128

        m[1] = self.conv1(x)  # 512,64,64

        m[2] = self.conv2(x)  # 1024,32,32

        m[3] = self.conv3(x)  # 2048,16,16

        m = list(m.values())
        x = self.decoder(m)
        x = self.cls_seg(x)
        x = self.sm(x)

        return x


class BasicBlock(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):

        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation,
                               dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out



class PPM(nn.ModuleList):
    def __init__(self, pool_sizes, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_sizes = pool_sizes
        self.in_channels = in_channels
        self.out_channels = out_channels
        for pool_size in pool_sizes:
            self.append(
                nn.Sequential(
                    nn.AdaptiveMaxPool2d(pool_size),
                    nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1),
                )
            )

    def forward(self, x):
        out_puts = []
        for ppm in self:
            ppm_out = nn.functional.interpolate(ppm(x), size=(x.size(2), x.size(3)), mode='bilinear',
                                                align_corners=True)
            out_puts.append(ppm_out)
        return out_puts


class PPMHEAD(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=[1, 2, 3, 6], num_classes=31):
        super(PPMHEAD, self).__init__()
        self.pool_sizes = pool_sizes
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.psp_modules = PPM(self.pool_sizes, self.in_channels, self.out_channels)
        self.final = nn.Sequential(
            nn.Conv2d(self.in_channels + len(self.pool_sizes) * self.out_channels, self.out_channels, kernel_size=1),
            # nn.BatchNorm2d(self.out_channels),
            nn.GroupNorm(16, self.out_channels),
            nn.GELU(),
            # nn.Dropout(0.5)
        )

    def forward(self, x):
        out = self.psp_modules(x)
        out.append(x)
        out = torch.cat(out, 1)
        out = self.final(out)
        return out


class FPNHEAD(nn.Module):
    def __init__(self, channels=2048, out_channels=256):
        super(FPNHEAD, self).__init__()
        self.PPMHead = PPMHEAD(in_channels=channels, out_channels=out_channels)

        self.Conv_fuse1 = nn.Sequential(
            nn.Conv2d(channels // 2, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse1_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2 = nn.Sequential(
            nn.Conv2d(channels // 4, out_channels, 1),
            nn.GroupNorm(16, out_channels),
            # nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse2_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.Conv_fuse3 = nn.Sequential(
            nn.Conv2d(channels // 8, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.Conv_fuse3_ = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.fuse_all = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(16, out_channels),
            nn.GELU(),
            nn.Dropout(0.5)
        )

        self.conv_x1 = nn.Conv2d(out_channels, out_channels, 1)

    def forward(self, input_fpn):
        # b, 512, 7, 7
        x1 = self.PPMHead(input_fpn[-1])

        x = nn.functional.interpolate(x1, size=(x1.size(2) * 2, x1.size(3) * 2), mode='bilinear', align_corners=True)
        x = self.conv_x1(x) + self.Conv_fuse1(input_fpn[-2])
        x2 = self.Conv_fuse1_(x)

        x = nn.functional.interpolate(x2, size=(x2.size(2) * 2, x2.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse2(input_fpn[-3])
        x3 = self.Conv_fuse2_(x)

        x = nn.functional.interpolate(x3, size=(x3.size(2) * 2, x3.size(3) * 2), mode='bilinear', align_corners=True)
        x = x + self.Conv_fuse3(input_fpn[-4])
        x4 = self.Conv_fuse3_(x)

        x1 = F.interpolate(x1, x4.size()[-2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x4.size()[-2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, x4.size()[-2:], mode='bilinear', align_corners=True)

        x = self.fuse_all(torch.cat([x1, x2, x3, x4], 1))

        return x



def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        sep_pos_embed=True,
        no_qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16_12(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=12,
        no_qkv_bias=False,
        sep_pos_embed=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model

def vit_base_patch16_nosep(**kwargs):
    model = VisionTransformer(
        img_size=128,
        in_chans=1,
        patch_size=8,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        num_frames=12,
        t_patch_size=3,
        no_qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        sep_pos_embed=False,
        **kwargs,
    )
    return model



if __name__ == '__main__':
    # input1 = torch.rand(2, 12, 128, 128)
    # input2 = torch.rand(2, 12, 128, 128)
    #
    #
    # model = vit_base_patch16()
    # print(model)
    # output = model(input1, input2)
    # print((output.shape))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = vit_base_patch16()
    checkpoint = torch.load('/mnt/sda/LXY/code/SR/OSCD/spectralGPT+_54_28.pth',
                            map_location=device)  # /media/ps/sda1/liyuxuan/change_detection/best/linear_test/linear_test69_43.pth

    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    # model.load_state_dict(torch.load(weights_path, map_location=device)['model'])
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)

    input1 = torch.rand(2, 12, 128, 128).cuda()
    input2 = torch.rand(2, 12, 128, 128).cuda()
    output = model(input1, input2)
    print((output.shape))
    #



