from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock 
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, optional_import
from modules import SwinTransformer, UnetrUpBlock

rearrange, _ = optional_import("einops", name="rearrange")


class LASNet(nn.Module):
    """
    Longitudinally-aware segmentation network (LAS-Net) based on the implemtantion of Swin UNETR in MONAI
    <https://docs.monai.io/en/stable/_modules/monai/networks/nets/swin_unetr.html> 
    """

    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        feature_size: int = 32,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 3,
        downsample="merging",
    ) -> None:
        
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        if img_size[0] % 7 == 0: #112
            window_size = ensure_tuple_rep(7, spatial_dims) # 7*7*7
        elif img_size[0] % 8 == 0: #128
            window_size = ensure_tuple_rep(8, spatial_dims) # 8*8*8

        if not (spatial_dims == 3):
            raise ValueError("spatial dimension should be 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            window_size = window_size,
            conv_features = feature_size, 
            embed_dim = feature_size, 
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            downsample=downsample,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.bottleneck = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=8*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels) 

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)

        enc0, enc0_ref = hidden_states_out[0], hidden_states_out[5]

        enc1, enc1_ref = self.encoder2(hidden_states_out[1]), self.encoder2(hidden_states_out[6])
        enc2, enc2_ref = self.encoder3(hidden_states_out[2]), self.encoder3(hidden_states_out[7])
        enc3, enc3_ref = self.encoder4(hidden_states_out[3]), self.encoder4(hidden_states_out[8])
        enc4, enc4_ref = self.bottleneck(hidden_states_out[4]), self.bottleneck(hidden_states_out[9])
        
        
        dec3, dec3_ref = self.decoder4(enc4, enc3, enc4_ref, enc3_ref)
        dec2, dec2_ref = self.decoder3(dec3, enc2, dec3_ref, enc2_ref)
        dec1, dec1_ref = self.decoder2(dec2, enc1, dec2_ref, enc1_ref)
        dec0, dec0_ref = self.decoder1(dec1, enc0, dec1_ref, enc0_ref)
        
        out = self.out(dec0)
        out_ref = self.out(dec0_ref)

        return out, out_ref

