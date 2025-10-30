from enum import Enum
from pydantic import BaseModel, ConfigDict
from torch import Tensor, empty, stack, randperm
from typing import Sequence

from torch.utils.data import Dataset

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



class MultiUnrollDataset(Dataset):

    def __init__(
        self,
        unrolls: Sequence[UnrollData],
        num_minibatches: int,
        minibatch_size: int,
        minibatched: bool = True
    ):
        if not minibatched:
            raise NotImplementedError

        self.observations = stack([u.observation for u in unrolls], dim=0)
        self.logits = stack([u.logits for u in unrolls], dim=0)
        self.action = stack([u.action for u in unrolls], dim=0)
        self.reward = stack([u.reward for u in unrolls], dim=0)

        self.preprocess(
            num_unrolls=len(unrolls),
            num_minibatches=num_minibatches,
            minibatch_size=minibatch_size
        )

    def __len__(self):
        return self.observations.shape[0]

    def __getitem__(self, idx):
        return {
            "observation": self.observations[idx],
            "logits": self.logits[idx],
            "action": self.action[idx],
            "reward": self.reward[idx],
        }

    def preprocess(self, num_unrolls: int, num_minibatches: int, minibatch_size: int):

        o = self.observations.view([num_unrolls, num_minibatches, minibatch_size, -1])
        l = self.logits.view([num_unrolls, num_minibatches, minibatch_size, -1])
        a = self.action.view([num_unrolls, num_minibatches, minibatch_size, -1])
        r = self.reward.view([num_unrolls, num_minibatches, minibatch_size, -1])

        print(o.shape)
        print(l.shape)
        print(a.shape)
        print(r.shape)

        index = randperm(num_unrolls + num_minibatches)

        self.observations = o.reshape([num_unrolls * num_minibatches, minibatch_size, -1])[index, :]
        self.logits = l.reshape([num_unrolls * num_minibatches, minibatch_size, -1])[index, :]
        self.action = a.reshape([num_unrolls * num_minibatches, minibatch_size, -1])[index, :]
        self.reward = r.reshape([num_unrolls * num_minibatches, minibatch_size, -1])[index, :]
