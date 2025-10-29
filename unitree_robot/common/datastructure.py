from enum import Enum
from pydantic import BaseModel, ConfigDict
from torch import Tensor, empty
from typing import Sequence

import numpy as np


class NETWORK_INTERFACE(Enum):
    LOCAL = "lo"
    LAPTOP_1 = "enp0s13f0u1"


class UnrollData(BaseModel):
    observation: Tensor
    logits: Tensor
    action: Tensor
    reward: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def initialize_empty(
        cls,
        unroll_length: int,
        observation_shape: Sequence[int],
        action_shape: Sequence[int],
        reward_shape: Sequence[int],
        device: str
    ):
        return UnrollData(
            observation=empty(size=[unroll_length, *observation_shape], device=device),
            logits=empty(size=[unroll_length, *action_shape], device=device), # TODO this could bei either `action_shape` or `2` (mean, var)
            action=empty(size=[unroll_length, *action_shape], device=device),
            reward=empty(size=[unroll_length, *reward_shape], device=device),
        )

