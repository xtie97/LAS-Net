from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetrBasicBlock 
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import
from monai.networks.layers.factories import Conv, Norm
from monai.networks.blocks.dynunet_block import UnetResBlock, get_conv_layer

rearrange, _ = optional_import("einops", name="rearrange")

class UnetrUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample_kernel_size: int,
        norm_name: str,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )
        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels + out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

        # proj the baseline feature maps to the same dimension as the follow-up feature maps
        self.proj_skip = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            Norm[norm_name, spatial_dims](out_channels)
        )
        self.proj_out = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1),
            Norm[norm_name, spatial_dims](out_channels)
        )
        self.proj_act = nn.LeakyReLU(inplace=True) # negative slope = 0.01

        self.proj_mask = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](in_channels=out_channels, out_channels=1, kernel_size=1, stride=1),
            Norm[norm_name, spatial_dims](1),
        )
        # consider to concatenate mask of baseline and interim to further mask the skip connection
        self.spatial_cross = nn.Sequential(
            Conv[Conv.CONV, spatial_dims](in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=(7-1) // 2),
            Norm[norm_name, spatial_dims](1),
        )


    def forward(self, inp, skip, inp_ref, skip_ref):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out_ref = self.transp_conv(inp_ref)

        # attention layer 
        skip_mask = self.proj_mask(self.proj_act(self.proj_skip(skip) + self.proj_out(out)))
        skip_mask_ref = self.proj_mask(self.proj_act(self.proj_skip(skip_ref) + self.proj_out(out_ref)))

        # element-wise multiplication
        skip = skip * torch.sigmoid(self.spatial_cross(torch.cat((skip_mask, skip_mask_ref), dim=1)))
        skip_ref = skip_ref * torch.sigmoid(skip_mask_ref)

        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)

        out_ref = torch.cat((out_ref, skip_ref), dim=1)
        out_ref = self.conv_block(out_ref)

        return out, out_ref


def window_partition(x, window_size):
    """window partition operation based on Swin Transformer.

    Args:
        x: input tensor with shape [B, H, W, C].
        window_size: tuple, window size for partition.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on Swin Transformer.

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on Swin Transformer.

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        else:
            raise NotImplementedError("Only 3D window attention is supported")

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, x_ref=None, mask=None):
        b, n, c = x.shape
        if x_ref is None:
            q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            q, _, _ = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
            _, k, v = self.qkv(x_ref).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.clone()[:n, :n].reshape(-1)].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is None:
            attn = self.softmax(attn)
        else: 
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        
        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        is_cross: bool = False,
    ) -> None:

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.is_cross = is_cross
        self.norm1 = norm_layer(dim)
        if self.is_cross:
            self.norm2 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")

    def forward_part1(self, x, x_ref=None, mask_matrix=None): 
        x_shape = x.size()
        x = self.norm1(x) # dim=-1
        if self.is_cross and x_ref is None:
            raise ValueError("This is for cross attention, but x_ref is None")

        if x_ref is not None:
            x_ref = self.norm2(x_ref)
        
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]

            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            if x_ref is not None:
                x_ref = F.pad(x_ref, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))

            _, dp, hp, wp, _ = x.shape
            dims = [b, dp, hp, wp]
        else:
            raise ValueError("input images must be 3D")

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
                if x_ref is not None:
                    shifted_x_ref = torch.roll(x_ref, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            else:
                raise ValueError("input images must be 3D")
            attn_mask = mask_matrix
        else:
            shifted_x = x
            if x_ref is not None:
                shifted_x_ref = x_ref
            attn_mask = None
        
        # after padding: 
        x_windows = window_partition(shifted_x, window_size)
        if x_ref is not None:
            x_ref_windows = window_partition(shifted_x_ref, window_size)
            attn_windows = self.attn(x_windows, x_ref_windows, mask=attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=attn_mask)
        #print(attn_windows.shape) # [1000, 7*7*7, 24]
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        #print(attn_windows.shape) # [1000, 7, 7, 7, 24]
        shifted_x = window_reverse(attn_windows, window_size, dims)
        #print(shifted_x.shape)    # [1, 70, 70, 70, 24]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            else:
                raise ValueError("input images must be 3D")
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        else:
            raise ValueError("input images must be 3D")
    
        return x
    

    def forward_part2(self, x):
        return self.drop_path(self.mlp(self.norm3(x)))

    def forward(self, x, x_ref=None, mask_matrix=None):
        shortcut = x
        if x_ref is not None:
            x = self.forward_part1(x, x_ref, mask_matrix=mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix=mask_matrix)

        x = shortcut + self.drop_path(x)
        x = x + self.forward_part2(x)
            
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on Swin Transformer.
    
    """

    def __init__(self, dim: int, norm_layer: type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""
    def forward(self, x):
        x = rearrange(x, "b c d h w -> b d h w c")
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        x = rearrange(x, "b d h w c -> b c d h w")
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """
     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class EncoderLayer(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        is_shift: bool = True, 
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for _ in window_size)

        # non-shift self-window-attention
        self.block1 = SwinTransformerBlock(dim=dim,
                                           num_heads=num_heads,
                                           window_size=self.window_size,
                                           shift_size=self.no_shift,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias,
                                           drop=drop,
                                           attn_drop=attn_drop,
                                           drop_path=drop_path,
                                           norm_layer=norm_layer)

        self.is_shift = is_shift
        # shifted self-window-attention
        if self.is_shift:                              
            self.block2 = SwinTransformerBlock(dim=dim,
                                            num_heads=num_heads,
                                            window_size=self.window_size,
                                            shift_size=self.shift_size,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop,
                                            attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            norm_layer=norm_layer)
        # additional non-shift self-window-attention
        else:
            self.block2 = SwinTransformerBlock(dim=dim,
                                            num_heads=num_heads,
                                            window_size=self.window_size,
                                            shift_size=self.no_shift,
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop,
                                            attn_drop=attn_drop,
                                            drop_path=drop_path,
                                            norm_layer=norm_layer)

        # window cross-attention
        self.block_cross =  SwinTransformerBlock(dim=dim,
                                                num_heads=num_heads,
                                                window_size=self.window_size,
                                                shift_size=self.no_shift,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=qkv_bias,
                                                drop=drop,
                                                attn_drop=attn_drop,
                                                drop_path=drop_path,
                                                norm_layer=norm_layer,
                                                is_cross=True)
            

    def forward(self, x, x_ref):
        x_shape = x.size()
            
        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]

        x = rearrange(x, "b c d h w -> b d h w c")
        x_ref = rearrange(x_ref, "b c d h w -> b d h w c")
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        
        # self-window-attention
        x = self.block1(x, mask_matrix=attn_mask)
        x_ref = self.block1(x_ref, mask_matrix=attn_mask)

        x = self.block2(x, mask_matrix=attn_mask)
        x_ref = self.block2(x_ref, mask_matrix=attn_mask)
        
        # window cross-attention
        x = self.block_cross(x, x_ref, mask_matrix=attn_mask)

        x = x.view(b, d, h, w, -1)
        x_ref = x_ref.view(b, d, h, w, -1)
        
        x = rearrange(x, "b d h w c -> b c d h w")
        x_ref = rearrange(x_ref, "b d h w c -> b c d h w")

        return x, x_ref


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_chans,
        embed_dim,
        patch_size: int = 2,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Conv[Conv.CONV, spatial_dims](
            in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=patch_size, padding=1)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x_shape = x.size()
   
        x = self.proj(x)
        if self.norm is not None:
            x_shape = x.size()
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            if len(x_shape) == 5:
                d, wh, ww = x_shape[2], x_shape[3], x_shape[4]
                x = x.transpose(1, 2).view(-1, self.embed_dim, d, wh, ww)
            
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        window_size: Sequence[int] = (7, 7, 7),
        conv_features: int = 32, 
        embed_dim: int = 32, 
        num_heads: Sequence[int] = [2, 4, 8, 16],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        super().__init__()

        # top conv layer 
        self.encoder_lv1 = UnetrBasicBlock(spatial_dims=spatial_dims, 
                                           in_channels=in_chans, 
                                           out_channels=conv_features, 
                                           kernel_size=3, 
                                           stride=1, 
                                           norm_name='instance', 
                                           res_block=True) 
       
        self.patch_embed = PatchEmbed(in_chans=conv_features, embed_dim=embed_dim) # 32*32*32
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(num_heads))]
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample

        self.encoder_lv2 = EncoderLayer(dim=int(embed_dim*1),
                                            num_heads=num_heads[0],
                                            window_size=window_size,
                                            drop_path=dpr[0],
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            norm_layer=norm_layer)
        self.encoder_lv2_down = down_sample_mod(dim=int(embed_dim*1), norm_layer=norm_layer, spatial_dims=len(window_size))
            
        self.encoder_lv3 = EncoderLayer(dim=int(embed_dim*2),
                                            num_heads=num_heads[1],
                                            window_size=window_size,
                                            drop_path=dpr[1],
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            norm_layer=norm_layer)
        self.encoder_lv3_down = down_sample_mod(dim=int(embed_dim*2), norm_layer=norm_layer, spatial_dims=len(window_size))

        self.encoder_lv4 = EncoderLayer(dim=int(embed_dim*4),
                                            num_heads=num_heads[2],
                                            window_size=window_size,
                                            drop_path=dpr[2],
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            norm_layer=norm_layer)
        self.encoder_lv4_down = down_sample_mod(dim=int(embed_dim*4), norm_layer=norm_layer, spatial_dims=len(window_size))

        self.encoder_lv5 = EncoderLayer(dim=int(embed_dim*8),
                                            num_heads=num_heads[3],
                                            window_size=window_size,
                                            drop_path=dpr[3],
                                            mlp_ratio=mlp_ratio,
                                            qkv_bias=qkv_bias,
                                            drop=drop_rate,
                                            attn_drop=attn_drop_rate,
                                            norm_layer=norm_layer,
                                            is_shift=False)
            

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            else:
                raise ValueError("Input tensor must be 5D.")
        return x

    def forward(self, x_in, normalize=True):
        x = x_in[:, :2, ...]
        x_ref = x_in[:, 2:, ...]
        
        x0 = self.encoder_lv1(x) # CNN output 128*128*128*32 
        x0_ref = self.encoder_lv1(x_ref) # CNN output 128*128*128*32

        x1 = self.patch_embed(x0) # 64*64*64*32
        x1_ref = self.patch_embed(x0_ref) # 64*64*64*32
        x1 = self.pos_drop(x1) 
        x1_ref = self.pos_drop(x1_ref)
        
        x1, x1_ref = self.encoder_lv2(x1.contiguous(), x1_ref.contiguous()) # 64*64*64*32
        x1_out = self.proj_out(x1, normalize)
        x1_ref_out = self.proj_out(x1_ref, normalize)
        
        x1 = self.encoder_lv2_down(x1.contiguous()) # 32*32*32*64
        x1_ref = self.encoder_lv2_down(x1_ref.contiguous()) # 32*32*32*64
        x2, x2_ref = self.encoder_lv3(x1.contiguous(), x1_ref.contiguous())
        x2_out = self.proj_out(x2, normalize)
        x2_ref_out = self.proj_out(x2_ref, normalize)

        x2 = self.encoder_lv3_down(x2.contiguous())
        x2_ref = self.encoder_lv3_down(x2_ref.contiguous())
        x3, x3_ref = self.encoder_lv4(x2.contiguous(), x2_ref.contiguous())
        x3_out = self.proj_out(x3, normalize)
        x3_ref_out = self.proj_out(x3_ref, normalize)

        x3 = self.encoder_lv4_down(x3.contiguous())
        x3_ref = self.encoder_lv4_down(x3_ref.contiguous())
        x4, x4_ref = self.encoder_lv5(x3.contiguous(), x3_ref.contiguous())
        x4_out = self.proj_out(x4, normalize)
        x4_ref_out = self.proj_out(x4_ref, normalize)

        return [x0,     x1_out,     x2_out,     x3_out,     x4_out, \
                x0_ref, x1_ref_out, x2_ref_out, x3_ref_out, x4_ref_out]
