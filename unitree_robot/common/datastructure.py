from enum import Enum
from pydantic import BaseModel, ConfigDict
from torch import Tensor, empty, stack
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
        observation_size: int,
        action_size: int,
        reward_size: int,
        device: str
    ):
        return UnrollData(
            observation=empty(size=[unroll_length, observation_size], device=device),
            logits=empty(size=[unroll_length, action_size * 2], device=device),
            action=empty(size=[unroll_length, action_size], device=device),
            reward=empty(size=[unroll_length, reward_size], device=device),
        )

class MultiUnrollData(BaseModel):
    observation: Tensor
    logits: Tensor
    action: Tensor
    reward: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_multiple_unrolls(cls, unrolls: Sequence[UnrollData]):
        return MultiUnrollData(
            observation=stack([u.observation for u in unrolls], dim=0),
            logits=stack([u.logits for u in unrolls], dim=0),
            action=stack([u.action for u in unrolls], dim=0),
            reward=stack([u.reward for u in unrolls], dim=0),
        )

    def as_batched(self):
        self.observation.reshape([num_unrolls, num_minibatches, minibatch_size, -1], inplace=True)
        self.logits.reshape([num_unrolls, num_minibatches, -1], inplace=True)
        self.action.reshape([num_unrolls, num_minibatches, -1], inplace=True)
        self.reward.reshape([num_unrolls, num_minibatches, minibatch_size, -1], inplace=True)
