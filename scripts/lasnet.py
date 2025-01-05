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
        num_heads: Sequence[int] = [2, 4, 8, 16],
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 3,
        downsample="merging",
        deep_supr_num: int = 1,
        use_checkpoint: bool = False,
        use_v2: bool = False,
    ) -> None:
        
        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims) # 7*7*7
        patch_size = ensure_tuple_rep(2, spatial_dims)
        
        if not (spatial_dims == 3):
            raise ValueError("spatial dimension should be 3.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        self.normalize = normalize
        self.in_chans = in_channels
        self.deep_supr_num = deep_supr_num

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim = feature_size, 
            window_size = window_size,
            patch_size = patch_size,
            num_heads = num_heads,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            use_v2=use_v2,
        )

        self.encoder0 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
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

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=8*feature_size,
            out_channels=4*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=4*feature_size,
            out_channels=2*feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=2*feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder0 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        feature_size_list = [feature_size, feature_size, 2*feature_size, 4*feature_size, 8*feature_size]
        self.out = nn.ModuleList(
            [UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size_list[i], out_channels=out_channels) for i in range(deep_supr_num)]
        )
        
    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # decode the hidden states
        x = x_in[:, :self.in_chans, ...] # interim PET/CT images
        x_ref = x_in[:, self.in_chans:, ...] # baseline PET/CT images
        enc0, enc0_ref = self.encoder0(x), self.encoder0(x_ref)
        enc1, enc1_ref = self.encoder1(hidden_states_out[0]), self.encoder1(hidden_states_out[4]) 
        enc2, enc2_ref = self.encoder2(hidden_states_out[1]), self.encoder2(hidden_states_out[5])
        enc3, enc3_ref = self.encoder3(hidden_states_out[2]), self.encoder3(hidden_states_out[6])
        dec4, dec4_ref  = self.bottleneck(hidden_states_out[3]), self.bottleneck(hidden_states_out[7])
        
        decs: list[torch.Tensor] = []
        dec_refs: list[torch.Tensor] = []

        dec3, dec3_ref = self.decoder3(dec4, enc3, dec4_ref, enc3_ref)
        decs.append(dec3)
        dec_refs.append(dec3_ref)
        dec2, dec2_ref = self.decoder2(dec3, enc2, dec3_ref, enc2_ref)
        decs.append(dec2)
        dec_refs.append(dec2_ref)
        dec1, dec1_ref = self.decoder1(dec2, enc1, dec2_ref, enc1_ref)
        decs.append(dec1)
        dec_refs.append(dec1_ref)
        dec0, dec0_ref = self.decoder0(dec1, enc0, dec1_ref, enc0_ref)
        decs.append(dec0)
        dec_refs.append(dec0_ref)

        decs.reverse() # the first element is the output of the last layer
        dec_refs.reverse() 
        
        outs: list[torch.Tensor] = []
        out_refs: list[torch.Tensor] = []

        for i in range(self.deep_supr_num):
            outs.append(self.out[i](decs[i])) # the first element is the output of the last layer
            out_refs.append(self.out[i](dec_refs[i]))

        if not self.training or len(outs) == 1:
            return outs[0], out_refs[0]
        
        return outs, out_refs
