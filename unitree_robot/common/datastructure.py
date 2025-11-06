from enum import Enum
from pydantic import BaseModel, ConfigDict
from torch import Tensor, zeros, stack, randperm
from typing import Sequence

from torch.utils.data import Dataset

import numpy as np


class NETWORK_INTERFACE(Enum):
    LOCAL = "lo"
    LAPTOP_1 = "enp0s13f0u1"


class UnrollData(BaseModel):
    observation: Tensor
    logits: Tensor
    actions: Tensor
    rewards: Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def initialize_empty(
        cls,
        num_unrolls: int,
        unroll_length: int,
        observation_size: int,
        action_size: int,
    ):
        return UnrollData(
            observation=zeros(size=[num_unrolls, unroll_length, observation_size], device="cpu"),
            logits=zeros(size=[num_unrolls, unroll_length, action_size * 2], device="cpu"),
            actions=zeros(size=[num_unrolls, unroll_length, action_size], device="cpu"),
            rewards=zeros(size=[num_unrolls, unroll_length, 1], device="cpu"),
        )


# class MultiUnrollDataset(Dataset):
#     def __init__(
#         self,
#         unrolls: Sequence[UnrollData],
#         num_minibatches: int,
#         minibatch_size: int,
#         minibatched: bool = True,
#     ):
#         if not minibatched:
#             raise NotImplementedError
#
#         self.observations = stack([u.observation for u in unrolls], dim=0)
#         self.logits = stack([u.logits for u in unrolls], dim=0)
#         self.actions = stack([u.actions for u in unrolls], dim=0)
#         self.rewards = stack([u.rewards for u in unrolls], dim=0)
#
#         self.preprocess(
#             num_unrolls=len(unrolls),
#             num_minibatches=num_minibatches,
#             minibatch_size=minibatch_size,
#         )
#
#     def __len__(self):
#         return self.observations.shape[0]
#
#     def __getitem__(self, idx: int):
#         return {
#             "observations": self.observations[idx],
#             "logits": self.logits[idx],
#             "actions": self.actions[idx],
#             "rewards": self.rewards[idx],
#         }
#
#     def preprocess(self, num_unrolls: int, num_minibatches: int, minibatch_size: int):
#         observations = self.observations.view(
#             [num_unrolls, num_minibatches, minibatch_size, -1]
#         )
#         logits = self.logits.view([num_unrolls, num_minibatches, minibatch_size, -1])
#         actions = self.actions.view([num_unrolls, num_minibatches, minibatch_size, -1])
#         rewards = self.rewards.view([num_unrolls, num_minibatches, minibatch_size, -1])
#
#         ll = num_unrolls * num_minibatches
#         self.observations = observations.reshape([ll, minibatch_size, -1])
#         self.logits = logits.reshape([ll, minibatch_size, -1])
#         self.actions = actions.reshape([ll, minibatch_size, -1])
#         self.rewards = rewards.reshape([ll, minibatch_size, -1])
#
#     def validate(self):
#         assert ~self.actions.isnan().any(), "Action contains NaN values"
#         assert ~self.actions.isinf().any(), "Action contains infinite values"
#         assert ~self.rewards.isnan().any(), "Reward contains NaN values"
#         assert ~self.rewards.isinf().any(), "Reward contains infinite values"
#         assert ~self.logits.isnan().any(), "Logits contains NaN values"
#         assert ~self.logits.isinf().any(), "Logits contains infinite values"
#         assert ~self.observations.isnan().any(), "Observations contain NaN values"
#         assert ~self.observations.isinf().any(), "Observations contain infinite values"
