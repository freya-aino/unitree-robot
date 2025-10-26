from enum import Enum
from pydantic import BaseModel, ConfigDict
from torch import Tensor, empty
from typing import Sequence

import numpy as np


class BACKENDS(Enum):
    GENERLIZED = "generalized" # high realism, low speed
    POSITIONAL = "positional" # medium realism, medium speed
    SPRING = "spring" # low realism, high speed

class NETWORK_INTERFACE(Enum):
   LOCAL = "lo"
   LAPTOP_1 = "enp0s13f0u1"


class UnrollData(BaseModel):
  observation: Tensor
  logits: Tensor
  action: Tensor
  reward: Tensor
  done: Tensor
  truncation: Tensor

  model_config = ConfigDict(arbitrary_types_allowed=True)

  @classmethod
  def initialize_empty(
    cls, 
    num_unrolls: int, 
    unroll_length: int,
    observation_shape: Sequence[int],
    action_shape: Sequence[int],
    reward_shape: Sequence[int], 
    device: str
  ):
    return UnrollData(
      observation=empty(size=[num_unrolls, unroll_length, *observation_shape], device=device),
      logits=empty(size=[num_unrolls, unroll_length, *action_shape], device=device), # TODO this could bei either `action_shape` or `2` (mean, var)
      action=empty(size=[num_unrolls, unroll_length, *action_shape], device=device),
      reward=empty(size=[num_unrolls, unroll_length, *reward_shape], device=device),
      done=empty(size=[num_unrolls, unroll_length], device=device),
      truncation=empty(size=[num_unrolls, unroll_length], device=device), # TODO this has an unclear shape at time moment, specify
    )

