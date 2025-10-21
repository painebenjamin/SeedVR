# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

from typing import List, Optional, Tuple, Union, Any, Callable
import torch
import os
import random
from einops import rearrange
from omegaconf import DictConfig, ListConfig
from torch import Tensor

from huggingface_hub import snapshot_download
from flashpack.integrations.diffusers import FlashPackDiffusionPipeline
from diffusers.configuration_utils import register_to_config

from seedvr.common.config import create_object
from seedvr.common.decorators import log_on_entry
from seedvr.common.diffusion import (
    classifier_free_guidance_dispatcher,
    create_sampler_from_config,
    create_sampling_timesteps_from_config,
    create_schedule_from_config,
)
from seedvr.common.distributed import (
    get_device,
    get_global_rank,
)
from seedvr.common.utils import filter_kwargs_for_method
from seedvr.models.dit.na import flatten, unflatten, pack, unpack
from seedvr.models.dit.nadit import NaDiT
from seedvr.models.video_vae_v3.modules.attn_video_vae import VideoAutoencoderKLWrapper
from seedvr.models.embeds import PrecomputedEmbeddings
from seedvr.common.diffusion.schedules.base import Schedule
from seedvr.common.diffusion.timesteps.base import SamplingTimesteps
from seedvr.common.diffusion.samplers.base import Sampler
from seedvr.common.diffusion.types import PredictionType
from seedvr.common.distributed.meta_init_utils import (
    meta_non_persistent_buffer_init_fn,
)
from seedvr.data.image.transforms.area_resize import area_resize
from seedvr.data.image.transforms.divisible_crop import DivisibleCrop
from seedvr.data.video.transforms.rearrange import Rearrange
# from common.fs import download

from seedvr.projects.video_diffusion_sr.color_fix import (
    get_wavelet_kernel,
    wavelet_reconstruction,
)


class SeedVRPipeline(FlashPackDiffusionPipeline):
    @register_to_config
    def __init__(
        self,
        dit: NaDiT,
        vae: VideoAutoencoderKLWrapper,
        sampler: Sampler,
        embeds: Optional[PrecomputedEmbeddings] = None,
        transform_timesteps: bool = True,
    ) -> None:
        super().__init__()
        embeds = embeds or PrecomputedEmbeddings.default(device=get_device(), dtype=dit.dtype)
        self.register_modules(
            dit=dit,
            vae=vae,
            sampler=sampler,
            embeds=embeds,
        )
        self.wavelet_kernel = get_wavelet_kernel(
            self.dit.dtype,
            in_device=self.dit.device
        )
        self.transform_timesteps = transform_timesteps

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    @classmethod
    def from_original_pretrained(
        cls,
        pretrained_model_name_or_path: str = "ByteDance-Seed/SeedVR2-7B",
        dit_filename: str = "seedvr2_ema_7b.pth",
        vae_filename: str = "ema_vae.pth",
        device: str | torch.device | int | None = None,
        schedule_type: str = "lerp",
        schedule_t: float = 1000.0,
        sampler_type: str = "euler",
        sampler_prediction_type: str = "v_lerp",
        timesteps_sampling_type: str = "uniform_trailing",
        timesteps_sampling_steps: int = 1,
        **kwargs: Any,
    ) -> "SeedVRPipeline":
        """
        Load the pipeline from a pretrained model repository.
        """
        if not os.path.isdir(pretrained_model_name_or_path):
            download_kwargs = filter_kwargs_for_method(snapshot_download, kwargs)
            download_kwargs["allow_patterns"] = [dit_filename, vae_filename]
            pretrained_model_name_or_path = snapshot_download(pretrained_model_name_or_path, **download_kwargs)

        dit_path = os.path.join(pretrained_model_name_or_path, dit_filename)
        vae_path = os.path.join(pretrained_model_name_or_path, vae_filename)

        dit = NaDiT.from_single_file(dit_path, device=device)
        vae = VideoAutoencoderKLWrapper.from_single_file(vae_path, device=device)
        schedule = create_schedule_from_config(
            config=DictConfig({"type": schedule_type, "T": schedule_t}),
            device=device,
        )
        sampling_timesteps = create_sampling_timesteps_from_config(
            config=DictConfig({"type": timesteps_sampling_type, "steps": timesteps_sampling_steps}),
            schedule=schedule,
            device=device,
        )
        sampler = create_sampler_from_config(
            config=DictConfig({"type": sampler_type, "prediction_type": sampler_prediction_type}),
            schedule=schedule,
            timesteps=sampling_timesteps,
        )
        return cls(dit=dit, vae=vae, sampler=sampler)

    @torch.no_grad()
    def vae_encode(self, samples: List[Tensor], use_sample: bool = True) -> List[Tensor]:
        latents = []
        if len(samples) > 0:
            device = get_device()
            dtype = self.vae.dtype
            scale = self.vae.config.scaling_factor
            shift = getattr(self.vae.config, "shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group samples of the same shape to batches if enabled.
            if self.vae.grouping:
                batches, indices = pack(samples)
            else:
                batches = [sample.unsqueeze(0) for sample in samples]

            # Vae process by each group.
            for sample in batches:
                sample = sample.to(device, dtype)
                print(f"{sample.shape=}")
                if hasattr(self.vae, "preprocess"):
                    sample = self.vae.preprocess(sample)
                if use_sample:
                    latent = self.vae.encode(sample).latent
                else:
                    # Deterministic vae encode, only used for i2v inference (optionally)
                    latent = self.vae.encode(sample).posterior.mode().squeeze(2)
                latent = latent.unsqueeze(2) if latent.ndim == 4 else latent
                latent = rearrange(latent, "b c ... -> b ... c")
                latent = (latent - shift) * scale
                latents.append(latent)

            # Ungroup back to individual latent with the original order.
            if self.vae.grouping:
                latents = unpack(latents, indices)
            else:
                latents = [latent.squeeze(0) for latent in latents]

        return latents

    @torch.no_grad()
    def vae_decode(self, latents: List[Tensor]) -> List[Tensor]:
        samples = []
        if len(latents) > 0:
            device = get_device()
            dtype = self.vae.dtype
            scale = self.vae.config.scaling_factor
            shift = getattr(self.vae.config, "shifting_factor", 0.0)

            if isinstance(scale, ListConfig):
                scale = torch.tensor(scale, device=device, dtype=dtype)
            if isinstance(shift, ListConfig):
                shift = torch.tensor(shift, device=device, dtype=dtype)

            # Group latents of the same shape to batches if enabled.
            if self.vae.grouping:
                latents, indices = pack(latents)
            else:
                latents = [latent.unsqueeze(0) for latent in latents]

            # Vae process by each group.
            for latent in latents:
                latent = latent.to(device, dtype)
                latent = latent / scale + shift
                latent = rearrange(latent, "b ... c -> b c ...")
                latent = latent.squeeze(2)
                sample = self.vae.decode(latent).sample
                if hasattr(self.vae, "postprocess"):
                    sample = self.vae.postprocess(sample)
                samples.append(sample)

            # Ungroup back to individual sample with the original order.
            if self.vae.grouping:
                samples = unpack(samples, indices)
            else:
                samples = [sample.squeeze(0) for sample in samples]

        return samples


    @torch.no_grad()
    def inference(
        self,
        noises: List[Tensor],
        conditions: List[Tensor],
        cfg_scale: float,
        cfg_rescale: float = 0.0,
        dit_offload: bool = False,
    ) -> List[Tensor]:
        assert len(noises) == len(conditions)
        batch_size = len(noises)

        # Return if empty.
        if batch_size == 0:
            return []

        (text_pos_embeds, text_pos_shapes), (text_neg_embeds, text_neg_shapes) = self.embeds.get()
        latents, latents_shapes = flatten(noises)
        latents_cond, _ = flatten(conditions)

        # Enter eval mode.
        was_training = self.dit.training
        self.dit.eval()

        # Sampling.
        latents = self.sampler.sample(
            x=latents,
            f=lambda args: classifier_free_guidance_dispatcher(
                pos=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_pos_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_pos_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                neg=lambda: self.dit(
                    vid=torch.cat([args.x_t, latents_cond], dim=-1),
                    txt=text_neg_embeds,
                    vid_shape=latents_shapes,
                    txt_shape=text_neg_shapes,
                    timestep=args.t.repeat(batch_size),
                ).vid_sample,
                scale=(
                    cfg_scale
                    if (args.i + 1) / len(self.sampler.timesteps)
                    <= 1.0
                    else 1.0
                ),
                rescale=cfg_rescale,
            ),
        )

        # Exit eval mode.
        self.dit.train(was_training)

        # Unflatten.
        latents = unflatten(latents, latents_shapes)

        if dit_offload:
            self.dit.to("cpu")

        # Vae decode.
        self.vae.to(get_device())
        samples = self.vae_decode(latents)

        if dit_offload:
            self.dit.to(get_device())
        return samples

    def get_linear_shift_function(self, x1: float, y1: float, x2: float, y2: float) -> Callable[[float], float]:
        """
        Get a linear shift function.
        """
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def timestep_transform(
        self,
        timesteps: Tensor,
        latents_shapes: Tensor,
    ) -> Tensor:
        # Skip if not needed.
        if not self.transform_timesteps:
            return timesteps

        # Compute resolution.
        frames = (latents_shapes[:, 0] - 1) * self.vae.temporal_downsample_factor + 1
        heights = latents_shapes[:, 1] * self.vae.spatial_downsample_factor
        widths = latents_shapes[:, 2] * self.vae.spatial_downsample_factor

        img_shift_fn = self.get_linear_shift_function(
            x1=256 * 256,
            y1=1.0,
            x2=1024 * 1024,
            y2=3.2,
        )
        vid_shift_fn = self.get_linear_shift_function(
            x1=256 * 256 * 37,
            y1=1.0,
            x2=1280 * 720 * 145,
            y2=5.0,
        )

        shift = torch.where(
            frames > 1,
            vid_shift_fn(heights * widths * frames),
            img_shift_fn(heights * widths),
        )

        # Shift timesteps.
        timesteps = timesteps / self.sampler.schedule.T
        timesteps = shift * timesteps / (1 + (shift - 1) * timesteps)
        timesteps = timesteps * self.sampler.schedule.T
        return timesteps

    def add_noise(
        self,
        x: torch.Tensor,
        aug_noise: torch.Tensor,
        cond_noise_scale: float = 0.25,
    ) -> torch.Tensor:
        """
        Add noise to the input.
        """
        t = torch.tensor([self.sampler.timesteps.T], device=x.device) * cond_noise_scale
        shape = torch.tensor(x.shape[1:], device=x.device).unsqueeze(0)
        t = self.timestep_transform(t, shape)
        x = self.sampler.schedule.forward(x, aug_noise, t)
        return x

    def random_seeded_like(
        self,
        tensor: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        return torch.randn(
            tensor.shape,
            device=tensor.device,
            dtype=tensor.dtype,
            generator=generator,
        )

    @torch.no_grad()
    def __call__(
        self,
        media: torch.Tensor,
        width: int,
        height: int,
        cfg_scale: float = 1.0,
        cfg_rescale: float = 1.0,
        seed: int | None = None,
        batch_size: int = 1,
        temporal_overlap: int = 0,
        cond_noise_scale: float = 0.25,
    ) -> torch.Tensor:
        """
        Generate a video from a media.
        """
        assert media.ndim == 4, "Media must be in CFHW format"
        num_frames = media.shape[0]
        overlap = 0 if num_frames == 1 else temporal_overlap
        step_size = batch_size - overlap
        if seed is None:
            seed = random.randint(0, 2**32-1)

        generator = torch.Generator(device=get_device()).manual_seed(seed)

        # TODO: Implement the video generation.
        latents = self.vae_encode([media])
        print(f"{latents[0].shape=} {len(latents)=} {latents[0].min()} {latents[0].max()}")
        noises = [self.random_seeded_like(latent, generator) for latent in latents]
        print(f"{noises[0].shape=} {len(noises)=} {noises[0].min()} {noises[0].max()}")
        aug_noises = [self.random_seeded_like(latent, generator) for latent in latents]
        print(f"{aug_noises[0].shape=} {len(aug_noises)=} {aug_noises[0].min()} {aug_noises[0].max()}")
        conditions = [
            self.get_condition(
                noise,
                self.add_noise(latent, aug_noise, cond_noise_scale),
                task="sr",
            )
            for noise, aug_noise, latent in zip(noises, aug_noises, latents)
        ]
        print(f"{conditions[0].shape=} {len(conditions)=} {conditions[0].min()} {conditions[0].max()}")
        samples = self.inference(
            noises,
            conditions,
            cfg_scale,
            cfg_rescale,
        )
        print(f"{samples[0]=}")
        """
        samples = [
            wavelet_reconstruction(
                samples[i], media[:, i].to(samples[i].device), self.wavelet_kernel
            )
            for i in range(len(samples))
        ]
        """
        return samples