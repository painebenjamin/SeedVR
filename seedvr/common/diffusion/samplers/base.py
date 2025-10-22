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

"""
Sampler base class.
"""

from abc import ABCMeta, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from omegaconf import DictConfig
from ..types import PredictionType, SamplingDirection


@dataclass
class SamplerModelArgs:
    x_t: torch.Tensor
    t: torch.Tensor
    i: int


class Sampler(ConfigMixin, SchedulerMixin, metaclass=ABCMeta):
    """
    Samplers are ODE/SDE solvers.
    """

    config_name = "scheduler_config.json"

    @register_to_config
    def __init__(
        self,
        schedule_type: Literal["lerp"] = "lerp",
        schedule_t: int | float = 1000.0,
        timesteps_type: Literal["uniform_trailing"] = "uniform_trailing",
        timesteps_steps: int = 1000,
        timesteps_shift: float = 1.0,
        prediction_type: PredictionType = PredictionType.v_lerp,
        return_endpoint: bool = True,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        from seedvr.common.diffusion.config import (
            create_sampling_timesteps_from_config,
            create_schedule_from_config,
        )

        self.schedule = create_schedule_from_config(
            config=DictConfig({"type": schedule_type, "T": schedule_t}),
            device=device,
        )
        self.timesteps = create_sampling_timesteps_from_config(
            config=DictConfig({"type": timesteps_type, "steps": timesteps_steps}),
            schedule=self.schedule,
            device=device,
        )
        self.prediction_type = prediction_type
        self.return_endpoint = return_endpoint

    @abstractmethod
    def sample(
        self,
        x: torch.Tensor,
        f: Callable[[SamplerModelArgs], torch.Tensor],
    ) -> torch.Tensor:
        """
        Generate a new sample given the the intial sample x and score function f.
        """

    def get_next_timestep(
        self,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the next sample timestep.
        Support multiple different timesteps t in a batch.
        If no more steps, return out of bound value -1 or T+1.
        """
        T = self.timesteps.T
        steps = len(self.timesteps)
        curr_idx = self.timesteps.index(t)
        next_idx = curr_idx + 1
        bound = -1 if self.timesteps.direction == SamplingDirection.backward else T + 1

        s = self.timesteps[next_idx.clamp_max(steps - 1)]
        s = s.where(next_idx < steps, bound)
        return s

    def get_endpoint(
        self,
        pred: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get to the endpoint of the probability flow.
        """
        x_0, x_T = self.schedule.convert_from_pred(pred, self.prediction_type, x_t, t)
        return x_0 if self.timesteps.direction == SamplingDirection.backward else x_T
