from torch import nn, Tensor, zeros, stack, randperm
from typing import Sequence
from dataclasses import dataclass
from torch.utils.data import Dataset
import numpy as np

class UnrollData(nn.Module):

    def __init__(
        self,
        num_unrolls: int,
        unroll_length: int,
        observation_size: int,
        action_size: int,
    ):
        super().__init__()

        self.observations = nn.Parameter(zeros(size=[num_unrolls, unroll_length, observation_size]), requires_grad=False)
        self.logits = nn.Parameter(zeros(size=[num_unrolls, unroll_length, action_size * 2]), requires_grad=False)
        self.actions = nn.Parameter(zeros(size=[num_unrolls, unroll_length, action_size]), requires_grad=False)
        self.rewards = nn.Parameter(zeros(size=[num_unrolls, unroll_length, 1]), requires_grad=False)

    def update(
        self,
        unroll_step: int,
        observation: Tensor,
        logits: Tensor,
        action: Tensor,
        reward: Tensor,
    ):
        self.observations[:, unroll_step:unroll_step+1, :] = observation
        self.logits[:, unroll_step:unroll_step+1, :] = logits
        self.actions[:, unroll_step:unroll_step+1, :] = action
        self.rewards[:, unroll_step:unroll_step+1, :] = reward

    def validate(self):
        is_observation_good = ~(self.observations.isnan().any() | self.observations.isinf().any())
        is_logits_good = ~(self.logits.isnan().any() | self.logits.isinf().any())
        is_actions_good = ~(self.actions.isnan().any() | self.actions.isinf().any())
        is_rewards_good = ~(self.rewards.isnan().any() | self.rewards.isinf().any())
        return is_observation_good and is_logits_good and is_actions_good and is_rewards_good

    # TODO - do a type of validation that check for abnormal values during training (e.g. actions outside of bounds or logits very small or very big)


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
