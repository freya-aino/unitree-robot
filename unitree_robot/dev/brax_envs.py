import jax

import brax
from brax import envs
from brax.envs.base import ObservationSize, State

from jax._src.xla_bridge import backends_are_initialized
from scipy import constants
from enum import Enum

class BACKENDS(Enum):
    GENERLIZED = "generalized" # high realism, low speed
    POSITIONAL = "positional" # medium realism, medium speed
    SPRING = "spring" # low realism, high speed

class CustomEnv(envs.Env):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, state: State, action: jax.Array) -> State:
        return super().step(state, action)

    def reset(self, rng: jax.Array) -> State:
        return super().reset(rng)

    @property
    def observation_size(self) -> ObservationSize:
        return super().observation_size

    @property
    def action_size(self) -> int:
        return super().action_size

    @property
    def backend(self) -> str:
        return super().backend
