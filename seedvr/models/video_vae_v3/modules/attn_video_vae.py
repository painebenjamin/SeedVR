# Copyright (c) 2023 HuggingFace Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.


from contextlib import nullcontext
from typing import Any, Literal

import diffusers
import torch
import torch.distributed as dist
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import Attention, SpatialNorm
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.downsampling import Downsample2D
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange, repeat
from flashpack.integrations.diffusers import FlashPackDiffusersModelMixin
from seedvr.common.distributed.advanced import get_sequence_parallel_world_size
from seedvr.common.logger import get_logger
from seedvr.common.utils import (
    maybe_use_tqdm,
    safe_interpolate_operation,
    safe_pad_operation,
    sliding_2d_windows,
)
from seedvr.models.utils import PretrainedMixin
from seedvr.models.video_vae_v3.modules.causal_inflation_lib import (
    InflatedCausalConv3d,
    causal_norm_wrapper,
    init_causal_conv3d,
    remove_head,
)
from seedvr.models.video_vae_v3.modules.global_config import set_norm_limit
from seedvr.models.video_vae_v3.modules.types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
    _receptive_field_t,
)
from tqdm import tqdm

logger = get_logger(__name__)  # pylint: disable=invalid-name
DEFAULT_LATENT_TILE_SIZE = (48, 48)
DEFAULT_LATENT_TILE_STRIDE = (32, 32)
DEFAULT_PIXEL_TILE_SIZE = (384, 384)
DEFAULT_PIXEL_TILE_STRIDE = (256, 256)


class Upsample3D(Upsample2D):
    """A 3D upsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv if self.name == "conv" else self.Conv2d_0

        assert (
            type(conv) is not nn.ConvTranspose2d
        ), f"Unsupported conv type: {type(conv)}"
        # Note: lora_layer is not passed into constructor in the original implementation.
        # So we make a simplification.
        conv = init_causal_conv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        assert not self.interpolate
        # [Override] MAGViT v2 implementation
        if not self.interpolate:
            upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
            self.upscale_conv = nn.Conv3d(
                self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
            )
            identity = (
                torch.eye(self.channels)
                .repeat(upscale_ratio, 1)
                .reshape_as(self.upscale_conv.weight)
            )
            self.upscale_conv.weight.data.copy_(identity)
            nn.init.zeros_(self.upscale_conv.bias)

        if self.name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: int | None = None,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split(
                    [split_size, hidden_states.size(2) - split_size], dim=2
                )
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            hidden_states[i] = self.upscale_conv(hidden_states[i])
            hidden_states[i] = rearrange(
                hidden_states[i],
                "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                x=self.spatial_ratio,
                y=self.spatial_ratio,
                z=self.temporal_ratio,
            )

        # [Overridden] For causal temporal conv
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states, memory_state=memory_state)
            else:
                hidden_states = self.Conv2d_0(hidden_states, memory_state=memory_state)

        if not self.slicing:
            return hidden_states
        else:
            return torch.cat(hidden_states, dim=2)


class Downsample3D(Downsample2D):
    """A 3D downsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if type(conv) in [nn.Conv2d, LoRACompatibleConv]:
            # Note: lora_layer is not passed into constructor in the original implementation.
            # So we make a simplification.
            conv = init_causal_conv3d(
                self.channels,
                self.out_channels,
                kernel_size=(
                    self.temporal_kernel,
                    self.spatial_kernel,
                    self.spatial_kernel,
                ),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                padding=(
                    1 if self.temporal_down else 0,
                    self.padding if self.spatial_down else 0,
                    self.padding if self.spatial_down else 0,
                ),
                inflation_mode=inflation_mode,
            )
        elif type(conv) is nn.AvgPool2d:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(
                    self.temporal_ratio,
                    self.spatial_ratio,
                    self.spatial_ratio,
                ),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv and self.padding == 0 and self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = safe_pad_operation(
                hidden_states, pad, mode="constant", value=0
            )

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = init_causal_conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = init_causal_conv3d(
            self.out_channels,
            self.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        if self.up:
            assert type(self.upsample) is Upsample3D
            self.upsample = Upsample3D(
                self.in_channels,
                use_conv=False,
                inflation_mode=inflation_mode,
                slicing=slicing,
            )
        elif self.down:
            assert type(self.downsample) is Downsample3D
            self.downsample = Downsample3D(
                self.in_channels,
                use_conv=False,
                padding=1,
                name="op",
                inflation_mode=inflation_mode,
            )

        if self.use_in_shortcut:
            self.conv_shortcut = init_causal_conv3d(
                self.in_channels,
                self.conv_shortcut.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )

    def forward(
        self,
        input_tensor,
        temb,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ):
        hidden_states = input_tensor

        hidden_states = causal_norm_wrapper(self.norm1, hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes.
            # see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample(input_tensor, memory_state=memory_state)
            hidden_states = self.upsample(hidden_states, memory_state=memory_state)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, memory_state=memory_state)
            hidden_states = self.downsample(hidden_states, memory_state=memory_state)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor


class DownEncoderBlock3D(DownEncoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
            downsample_padding=downsample_padding,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # [Override] Replace module.
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UpDecoderBlock3D(UpDecoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: int | None = None,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
            temb_channels=temb_channels,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                    slicing=slicing,
                )
            )

            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_upsample:
            # [Override] Replace module & use learnable upsample
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        interpolate=False,
                        inflation_mode=inflation_mode,
                        slicing=slicing,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor | None = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            hidden_states = resnet(hidden_states, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            # [Override] Replace module.
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups
                            if resnet_time_scale_shift == "default"
                            else None
                        ),
                        spatial_norm_dim=(
                            temb_channels
                            if resnet_time_scale_shift == "spatial"
                            else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(
        self, hidden_states, temb=None, memory_state: MemoryState = MemoryState.DISABLED
    ):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0](hidden_states, temb, memory_state=memory_state)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet(hidden_states, temb, memory_state=memory_state)

        return hidden_states


class Encoder3D(nn.Module):
    r"""
    [Override] override most logics to support extra condition input and causal conv

    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock3D",)`):
            The types of down blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_down_block`
            for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        # [Override] add extra_cond_dim, temporal down num
        temporal_down_num: int = 2,
        extra_cond_dim: int = None,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.extra_cond_dim = extra_cond_dim

        self.conv_extra_cond = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            down_block_type = down_block_type.replace("2D", "3D")
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # [Override] to support temporal down block design
            is_temporal_down_block = (
                i >= len(block_out_channels) - self.temporal_down_num - 1
            )
            # Note: take the last ones

            assert (
                down_block_type == "DownEncoderBlock3D"
            ), f"Unsupported down block type: {down_block_type}"

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                # Note: Don't know why set it as 0
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

            def zero_module(module):
                # Zero out the parameters of a module and return it.
                for p in module.parameters():
                    p.detach().zero_()
                return module

            self.conv_extra_cond.append(
                zero_module(
                    nn.Conv3d(
                        extra_cond_dim,
                        output_channel,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    )
                )
                if self.extra_cond_dim is not None and self.extra_cond_dim > 0
                else None
            )

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = init_causal_conv3d(
            block_out_channels[-1],
            conv_out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.gradient_checkpointing = gradient_checkpoint

    def forward(
        self,
        sample: torch.FloatTensor,
        extra_cond=None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block),
                    sample,
                    memory_state,
                    use_reentrant=False,
                )
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(
                        extra_block(extra_cond), size=sample.shape[2:]
                    )

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

            # sample = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.mid_block), sample, use_reentrant=False
            # )

        else:
            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = down_block(sample, memory_state=memory_state)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(
                        extra_block(extra_cond), size=sample.shape[2:]
                    )

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock3D",)`):
            The types of up blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        # [Override] add temporal up block
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            up_block_type = up_block_type.replace("2D", "3D")
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            assert (
                up_block_type == "UpDecoderBlock3D"
            ), f"Unsupported up block type: {up_block_type}"
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = init_causal_conv3d(
            block_out_channels[0],
            out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.gradient_checkpointing = gradient_checkpoint

    # Note: Just copy from Decoder.
    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: torch.FloatTensor | None = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                sample = self.mid_block(
                    sample, latent_embeds, memory_state=memory_state
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        memory_state,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = self.mid_block(
                    sample, latent_embeds, memory_state=memory_state
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        memory_state,
                    )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds, memory_state=memory_state)

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class VideoAutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str] = ("DownEncoderBlock3D",) * 4,
        up_block_types: tuple[str] = ("UpDecoderBlock3D",) * 4,
        block_out_channels: tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.9152,
        force_upcast: float = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "pad",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 4,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        spatial_downsample_factor: int = 8,
        temporal_downsample_factor: int = 4,
        slicing: dict[str, Any] = {"split_size": 4, "memory_device": "same"},
        memory_limit: dict[str, Any] = {"conv_max_mem": 0.5, "norm_max_mem": 0.5},
        extra_cond_dim: int | None = None,
        attention: bool = True,
        grouping: bool = False,
    ) -> None:
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = slicing_sample_min_size // (
            2**temporal_scale_num
        )
        self.grouping = grouping

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # [Override] make sure it can be normally initialized
            down_block_types=tuple(
                [
                    down_block_type.replace("3D", "2D")
                    for down_block_type in down_block_types
                ]
            ),
            up_block_types=tuple(
                [up_block_type.replace("3D", "2D") for up_block_type in up_block_types]
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
        )

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
            mid_block_add_attention=attention,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
            mid_block_add_attention=attention,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False
        if slicing is not None:
            self.set_causal_slicing(**slicing)
        if memory_limit is not None:
            self.set_memory_limit(**memory_limit)

    @apply_forward_hook
    def encode(
        self, x: torch.FloatTensor, return_dict: bool = True
    ) -> AutoencoderKLOutput:
        h = self.slicing_encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | torch.Tensor:
        decoded = self.slicing_decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _encode(
        self, x: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> torch.Tensor:
        h = self.encoder(x, memory_state=memory_state)
        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h
        return output

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED
    ) -> torch.Tensor:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z, memory_state=memory_state)
        output = self.decoder(z, memory_state=memory_state)
        return output

    def slicing_encode(self, x: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if (
            self.use_slicing
            and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size
        ):
            x_slices = x[:, :, 1:].split(
                split_size=self.slicing_sample_min_size * sp_size, dim=2
            )
            progress_bar = tqdm(total=len(x_slices), desc="Encoding")
            encoded_slices = [
                self.tiled_encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    device=self.device,
                    memory_state=MemoryState.INITIALIZING,
                    use_tqdm=False,
                )
            ]
            progress_bar.update(1)
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self.tiled_encode(
                        x_slices[x_idx],
                        device=self.device,
                        memory_state=MemoryState.ACTIVE,
                        use_tqdm=False,
                    )
                )
                progress_bar.update(1)
            progress_bar.close()
            return torch.cat(encoded_slices, dim=2)
        else:
            return self.tiled_encode(x, device=self.device, use_tqdm=True)

    def slicing_decode(self, z: torch.Tensor) -> torch.Tensor:
        sp_size = get_sequence_parallel_world_size()
        if (
            self.use_slicing
            and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size
        ):
            z_slices = z[:, :, 1:].split(
                split_size=self.slicing_latent_min_size * sp_size, dim=2
            )
            progress_bar = None
            if len(z_slices) > 1:
                progress_bar = tqdm(total=len(z_slices), desc="Decoding")

            decoded_slices = [
                self.tiled_decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    device=self.device,
                    use_tqdm=False,
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            if progress_bar is not None:
                progress_bar.update(1)
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self.tiled_decode(
                        z_slices[z_idx],
                        device=self.device,
                        use_tqdm=False,
                        memory_state=MemoryState.ACTIVE,
                    )
                )
                if progress_bar is not None:
                    progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()
            return torch.cat(decoded_slices, dim=2)
        else:
            return self.tiled_decode(
                z,
                device=self.device,
                use_tqdm=True,
                memory_state=MemoryState.DISABLED,
            )

    @torch.no_grad()
    def tiled_decode(
        self,
        hidden_states: torch.Tensor,
        device: torch.device,
        tile_size: tuple[int, int] = DEFAULT_LATENT_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_LATENT_TILE_STRIDE,
        loop: bool = False,
        use_tqdm: bool = True,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.Tensor | None:
        """
        :param hidden_states: hidden states tensor [B, C, T, H, W]
        :param device: device
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: output tensor [B, C, T, H, W]
        """
        hidden_states = hidden_states.to(dtype=self.dtype)
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        size_h = min(size_h, H)
        size_w = min(size_w, W)
        stride_h = min(stride_h, size_h)
        stride_w = min(stride_w, size_w)

        is_distributed = False
        rank = 0
        world_size = 1

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            is_distributed = world_size > 1

        # Split tasks
        tasks = sliding_2d_windows(
            height=H,
            width=W,
            tile_size=(size_w, size_h),
            tile_stride=(stride_w, stride_h),
        )

        # Handle distributed processing
        if is_distributed:
            # Distribute tasks among ranks
            if len(tasks) < world_size:
                # If fewer tasks than ranks, distribute one task per rank until we run out
                if rank < len(tasks):
                    local_tasks = [tasks[rank]]
                else:
                    local_tasks = []
            else:
                # Distribute tasks evenly
                tasks_per_rank = len(tasks) // world_size
                remainder = len(tasks) % world_size

                start_idx = rank * tasks_per_rank + min(rank, remainder)
                end_idx = start_idx + tasks_per_rank + (1 if rank < remainder else 0)
                local_tasks = tasks[start_idx:end_idx]
        else:
            local_tasks = tasks

        # Use GPU for distributed processing, CPU for non-distributed
        if is_distributed:
            data_device = device
            computation_device = device
        else:
            data_device = torch.device("cpu")
            computation_device = device

        out_T = T * 4 - 3
        weight = torch.zeros(
            (
                1,
                1,
                out_T,
                H * self.spatial_downsample_factor,
                W * self.spatial_downsample_factor,
            ),
            dtype=hidden_states.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (
                1,
                3,
                out_T,
                H * self.spatial_downsample_factor,
                W * self.spatial_downsample_factor,
            ),
            dtype=hidden_states.dtype,
            device=data_device,
        )

        # Process local tasks
        for h, h_, w, w_ in maybe_use_tqdm(
            local_tasks, desc="Decoding", use_tqdm=use_tqdm and rank == 0
        ):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_].to(
                computation_device
            )
            hidden_states_batch = self._decode(
                hidden_states_batch, memory_state=memory_state
            ).to(data_device)
            hidden_states_batch.clamp_(-1, 1)
            hidden_states_batch = hidden_states_batch[:, :, :out_T]

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.spatial_downsample_factor,
                    (size_w - stride_w) * self.spatial_downsample_factor,
                ),
            ).to(dtype=hidden_states.dtype, device=data_device)

            target_h = h * self.spatial_downsample_factor
            target_w = w * self.spatial_downsample_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += (
                hidden_states_batch * mask
            )
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask

        # Handle distributed assembly
        if is_distributed:
            value_reduce = dist.reduce(
                values, dst=0, op=dist.ReduceOp.SUM, async_op=True
            )
            weight_reduce = dist.reduce(
                weight, dst=0, op=dist.ReduceOp.SUM, async_op=True
            )

            # Wait for the reduction to complete
            value_reduce.wait()
            weight_reduce.wait()
            torch.cuda.synchronize() if device.type == "cuda" else None

            if rank == 0:
                # Distributed case, only return values on rank 0
                values = values / weight
                values = values.float().clamp_(-1, 1)
                return values.to("cpu")
            return None
        else:
            # Non-distributed case
            values = values / weight
            values = values.float().clamp_(-1, 1)
            return values.to("cpu")

    @torch.no_grad()
    def tiled_encode(
        self,
        video: torch.Tensor,
        device: torch.device | None = None,
        tile_size: tuple[int, int] = DEFAULT_PIXEL_TILE_SIZE,
        tile_stride: tuple[int, int] = DEFAULT_PIXEL_TILE_STRIDE,
        use_tqdm: bool = True,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.Tensor:
        """
        :param video: video tensor [B, C, T, H, W]
        :param device: device
        :param tile_size: tile size
        :param tile_stride: tile stride
        :return: output tensor [B, C, T, H, W]
        """
        if device is None:
            device = self.device

        video = video.to(dtype=self.dtype)
        _, _, T, H, W = video.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        is_distributed = False
        rank = 0
        world_size = 1

        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            is_distributed = world_size > 1

        # Split tasks
        tasks = sliding_2d_windows(
            height=H,
            width=W,
            tile_size=(size_w, size_h),
            tile_stride=(stride_w, stride_h),
        )

        # Handle distributed processing
        if is_distributed:
            # Distribute tasks among ranks
            if len(tasks) < world_size:
                # If fewer tasks than ranks, distribute one task per rank until we run out
                if rank < len(tasks):
                    local_tasks = [tasks[rank]]
                else:
                    local_tasks = []
            else:
                # Distribute tasks evenly
                tasks_per_rank = len(tasks) // world_size
                remainder = len(tasks) % world_size

                start_idx = rank * tasks_per_rank + min(rank, remainder)
                end_idx = start_idx + tasks_per_rank + (1 if rank < remainder else 0)
                local_tasks = tasks[start_idx:end_idx]
        else:
            local_tasks = tasks

        # Use GPU for distributed processing, CPU for non-distributed
        if is_distributed:
            data_device = device
            computation_device = device
        else:
            data_device = torch.device("cpu")
            computation_device = device

        out_T = (T + 3) // 4
        weight = torch.zeros(
            (
                1,
                1,
                out_T,
                H // self.spatial_downsample_factor,
                W // self.spatial_downsample_factor,
            ),
            dtype=video.dtype,
            device=data_device,
        )
        values = torch.zeros(
            (
                1,
                self.config.latent_channels * 2,
                out_T,
                H // self.spatial_downsample_factor,
                W // self.spatial_downsample_factor,
            ),
            dtype=video.dtype,
            device=data_device,
        )

        task_num = 0
        num_tasks = len(local_tasks)
        for h, h_, w, w_ in maybe_use_tqdm(
            local_tasks, desc="Encoding", use_tqdm=use_tqdm and rank == 0
        ):
            hidden_states_batch = video[:, :, :, h:h_, w:w_].to(computation_device)
            hidden_states_batch = self._encode(
                hidden_states_batch, memory_state=memory_state
            ).to(data_device)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) // self.spatial_downsample_factor,
                    (size_w - stride_w) // self.spatial_downsample_factor,
                ),
            ).to(dtype=video.dtype, device=data_device)

            target_h = h // self.spatial_downsample_factor
            target_w = w // self.spatial_downsample_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += (
                hidden_states_batch * mask
            )
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask

            task_num += 1

        # Handle distributed assembly
        if is_distributed:
            value_reduce = dist.all_reduce(values, op=dist.ReduceOp.SUM, async_op=True)
            weight_reduce = dist.all_reduce(weight, op=dist.ReduceOp.SUM, async_op=True)

            # Wait for the reduction to complete
            value_reduce.wait()
            weight_reduce.wait()
            torch.cuda.synchronize() if device.type == "cuda" else None

            # All ranks return the same values
            values = values / weight
            return values
        else:
            # Non-distributed case
            values = values / weight
            return values.to(device)

    def build_1d_mask(
        self, length: int, left_bound: bool, right_bound: bool, border_width: int
    ) -> torch.Tensor:
        """
        Builds a 1D mask.

        :param length: length
        :param left_bound: left bound
        :param right_bound: right bound
        :param border_width: border width
        :return: mask
        """
        x = torch.ones((length,))

        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + 1) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + 1) / border_width, dims=(0,)
            )

        return x

    def build_mask(
        self,
        data: torch.Tensor,
        is_bound: tuple[bool, bool, bool, bool],
        border_width: tuple[int, int],
    ) -> torch.Tensor:
        """
        :param data: data tensor [B, C, T, H, W]
        :param is_bound: bound toggle for each side
        :param border_width: border width
        :return: mask tensor [1, 1, 1, H, W]
        """
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(H, is_bound[0], is_bound[1], border_width[0])
        w = self.build_1d_mask(W, is_bound[2], is_bound[3], border_width[1])

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    def forward(
        self,
        x: torch.FloatTensor,
        mode: Literal["encode", "decode", "all"] = "all",
        **kwargs,
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict)


class VideoAutoencoderKLWrapper(
    PretrainedMixin, FlashPackDiffusersModelMixin, VideoAutoencoderKL, ConfigMixin
):
    @register_to_config
    def __init__(
        self,
        freeze_encoder: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: tuple[str] = ("DownEncoderBlock3D",) * 4,
        up_block_types: tuple[str] = ("UpDecoderBlock3D",) * 4,
        block_out_channels: tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        latent_channels: int = 16,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.9152,
        force_upcast: float = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "pad",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 4,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        spatial_downsample_factor: int = 8,
        temporal_downsample_factor: int = 4,
        slicing: dict[str, Any] = {"split_size": 4, "memory_device": "same"},
        memory_limit: dict[str, Any] = {"conv_max_mem": 0.5, "norm_max_mem": 0.5},
        extra_cond_dim: int | None = None,
        attention: bool = True,
        grouping: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            temporal_scale_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
            slicing_sample_min_size=slicing_sample_min_size,
            use_quant_conv=use_quant_conv,
            use_post_quant_conv=use_post_quant_conv,
            spatial_downsample_factor=spatial_downsample_factor,
            temporal_downsample_factor=temporal_downsample_factor,
            slicing=slicing,
            memory_limit=memory_limit,
            extra_cond_dim=extra_cond_dim,
            attention=attention,
            grouping=grouping,
        )
        self.freeze_encoder = freeze_encoder

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def encode(self, x: torch.FloatTensor) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x).latent_dist
        z = p.sample().squeeze(2)
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.FloatTensor) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z).sample.squeeze(2)
        return CausalDecoderOutput(x)

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1, f"Unsupported input shape: {x.shape}"
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: int | None,
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            split_size = split_size * 4
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = split_size // self.temporal_downsample_factor
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: float | None, norm_max_mem: float | None):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(
                    conv_max_mem if conv_max_mem is not None else float("inf")
                )
