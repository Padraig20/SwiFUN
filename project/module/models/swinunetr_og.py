# https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/swin_unetr.py

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.utils import optional_import
from monai.utils.deprecate_utils import deprecated_arg

from .SwiFT.swin4d_transformer_ver7 import SwinTransformer4D

rearrange, _ = optional_import("einops", name="rearrange")

__all__ = [
    "SwinUNETR",
    "AvgMaxPool3D",
]

import torch
import torch.nn as nn

class AvgMaxPool3D(nn.Module):
    def __init__(self, num_channels):
        """
        Initializes the AvgMaxPool3D module.
        
        Args:
        num_channels (int): The number of channels in the input feature map.
        """
        super(AvgMaxPool3D, self).__init__()
        self.conv = nn.Conv3d(2 * num_channels, num_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the AvgMaxPool3D layer.
        
        Args:
        x (torch.Tensor): The input tensor with shape (B, C, H, D, W, T).
        
        Returns:
        torch.Tensor: Output tensor with shape (B, C, H, D, W).
        """
        B, C, H, D, W, T = x.size()
        
        x = x.view(B, C, H * D * W, T)
        
        # average and max pool along temporal dim
        avg_pooled = torch.mean(x, dim=-1, keepdim=True)  # (B, C, H*D*W, 1)
        max_pooled = torch.max(x, dim=-1, keepdim=True)[0]  # (B, C, H*D*W, 1)
        
        # concat along channel dim
        combined = torch.cat((avg_pooled, max_pooled), dim=1)  # (B, 2C, H*D*W, 1)
        
        combined = combined.view(B, 2 * C, H, D, W)
        
        # 1x1x1 conv to reduce channel dim
        combined = self.conv(combined)  # (B, C, H, D, W)
        
        return combined


class SwinUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    #patch_size: Final[int] = 2

    @deprecated_arg(
        name="img_size",
        since="1.3",
        removed="1.5",
        msg_suffix="The img_size argument is not required anymore and "
        "checks on the input size are run during forward().",
    )
    def __init__(
        self,
        img_size: Sequence[int] | int,
        in_channels: int,
        out_channels: int,
        patch_size: int = (6, 6, 6, 1),
        window_size: int = (4, 4, 4, 4),
        first_window_size: int = (4, 4, 4, 4),
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 36,
        norm_name: tuple | str = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        spatial_dims: int = 4,
        c_multiplier: int = 2,
        last_layer_full_MSA: bool = True,
        to_float: bool = True,
    ) -> None:
        """
        Args:
            img_size: spatial dimension of input image.
                This argument is only used for checking that the input image size is divisible by the patch size.
                The tensor passed to forward() can have a dynamic shape as long as its spatial dimensions are divisible by 2**5.
                It will be removed in an upcoming version.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            spatial_dims: number of spatial dims.
        """

        super().__init__()

        self.patch_size = patch_size

        if spatial_dims not in (2, 3, 4):
            raise ValueError("spatial dimension should be 2, 3 or 4.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        self.normalize = normalize
        
        # sample inoput shape
        in_channels = 1
        #feature_size = 36
        #window_size = (4, 4, 4, 4)
        #patch_size = (6, 6, 6, 1)
        #depths = (2, 2, 6, 2)
        #num_heads = (3, 6, 12, 24)
        #c_multiplier = 2
        
        self.swinViT = SwinTransformer4D(
            in_chans=in_channels, # in_channels is always 1 for our task
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            c_multiplier=c_multiplier,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            first_window_size=first_window_size,
            img_size=img_size,
            last_layer_full_MSA=last_layer_full_MSA,
            to_float=to_float,
        )
                
        self.temporal_squeeze_init = AvgMaxPool3D(num_channels=1)
        self.temporal_squeeze0 = AvgMaxPool3D(num_channels=feature_size)
        self.temporal_squeeze1 = AvgMaxPool3D(num_channels=2*feature_size)
        self.temporal_squeeze2 = AvgMaxPool3D(num_channels=4*feature_size)
        self.temporal_squeeze3 = AvgMaxPool3D(num_channels=8*feature_size)
        self.temporal_squeeze4 = AvgMaxPool3D(num_channels=8*feature_size)

        spatial_dims = 3 # unet dims are 3 afer temporal squeeze

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
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
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        
        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=8 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=1, #no upsampling needed here
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
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
            upsample_kernel_size=6,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def load_from(self, weights):
        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    @torch.jit.unused
    def _check_input_size(self, spatial_shape):
        img_size = np.array(spatial_shape)
        remainder = (img_size % np.power(self.patch_size, 5)) > 0
        if remainder.any():
            wrong_dims = (np.where(remainder)[0] + 2).tolist()
            raise ValueError(
                f"spatial dimensions {wrong_dims} of input image (spatial shape: {spatial_shape})"
                f" must be divisible by {self.patch_size}**5."
            )

    def forward(self, x_in, group_in=None):
        #print(x_in.shape) # 
        
        hidden_states_out = self.swinViT(x_in) # (b, c, h, w, d, t) = [16, 288, 2, 2, 2, 20] length 5
        # 0 = (b, c, h, w, d, t) = [16, 216, 96, 96, 96, 20]
        # 1 = (b, c, h, w, d, t) = [16, 1, 16, 16, 16, 20]
        # 2 = (b, c, h, w, d, t) = [16, 2, 8, 8, 8, 20]
        # 3 = (b, c, h, w, d, t) = [16, 4, 4, 4, 4, 20]
        # 4 = (b, c, h, w, d, t) = [16, 8, 2, 2, 2, 20]
        
        # temporal squeeze for each element in hidden_states_out
        hidden_states_out = [self.temporal_squeeze0(hidden_states_out[0]),
                             self.temporal_squeeze1(hidden_states_out[1]),
                             self.temporal_squeeze2(hidden_states_out[2]),
                             self.temporal_squeeze3(hidden_states_out[3]),
                             self.temporal_squeeze4(hidden_states_out[4])]
        # 0 = (b, c, h, w, d, t) = [16, 216, 96, 96, 96] [16, 36, 16, 16, 16, 20]  [16, 16, 16, 16, 36]
        # 1 = (b, c, h, w, d, t) = [16, 1, 16, 16, 16]   [16, 72, 8, 8, 8, 20]     [16, 8, 8, 8, 72]
        # 2 = (b, c, h, w, d, t) = [16, 2, 8, 8, 8]      [16, 144, 4, 4, 4, 20]    [16, 4, 4, 4, 144]
        # 3 = (b, c, h, w, d, t) = [16, 4, 4, 4, 4]      [16, 288, 2, 2, 2, 20]    [16, 2, 2, 2, 288]
        # 4 = (b, c, h, w, d, t) = [16, 8, 2, 2, 2]      [16, 288, 2, 2, 2, 20]    [16, 2, 2, 2, 288]
        
        # (b, c, h, w, d, t) = [16, 1, 96, 96, 96, 20] -> temporal squeeze -> (16, 1, 96, 96, 96)
        x_in = self.temporal_squeeze_init(x_in) # (b, h, w, d, c) = [16, 96, 96, 96, 1]
        enc0 = self.encoder1(x_in)
        
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        
        dec3 = self.decoder5(dec4, hidden_states_out[3]) # [16, 288, 2, 2, 2] - [16, 288, 2, 2, 2]
        dec2 = self.decoder4(dec3, enc3) # [16, 288, 2, 2, 2] - [16, 144, 4, 4, 4]
        dec1 = self.decoder3(dec2, enc2) # [16, 144, 4, 4, 4] - [16, 72, 8, 8, 8]
        dec0 = self.decoder2(dec1, enc1) # [16, 72, 8, 8, 8] - [16, 36, 16, 16, 16]
        out = self.decoder1(dec0, enc0) # [16, 36, 16, 16, 16] - [16, 36, 96, 96, 96]
        
        logits = self.out(out) # [16, 1, 96, 96, 96]

        return logits