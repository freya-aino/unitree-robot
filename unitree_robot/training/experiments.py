from abc import ABC
from mujoco.mjx import Data
from torch import Tensor

from unitree_robot.training.rewards import (
    BaseOrientationReward,
    EnergyReward,
    BodyHeightReward,
    JointLimitReward,
    DistanceFromCenterReward,
)


class Experiment(ABC):
    def __init__(self):
        pass
    def __call__(self, mjx_data: Data) -> Tensor:
        raise NotImplementedError

class TestExperiment(Experiment):
    def __init__(self):
        super().__init__()
    def __call__(self, mjx_data: Data) -> Tensor:
        return Tensor(mjx_data.qacc.mean(axis=-1).copy())

# class StandUpExperiment(Experiment):
#     def __init__(
#         self,
#         mj_model: MjModel,
#         body_name: str,
#         body_angle_reward_scale: float,
#         body_height_reward_scale: float,
#         energy_reward_scale: float,
#         distance_from_origin_reward_scale: float,
#         joint_limit_reward_scale: float,
#     ):
#         self.rewards = {
#             "base_height_reward": BodyHeightReward(scale=body_height_reward_scale),
#             "base_orientation_reward": BaseOrientationReward(
#                 body_name=body_name, scale=body_angle_reward_scale
#             ),
#             "distance_from_origin_reward": DistanceFromCenterReward(
#                 scale=distance_from_origin_reward_scale
#             ),
#             "energy_reward": EnergyReward(scale=energy_reward_scale),
#             "joint_limit_reward": JointLimitReward(
#                 scale=joint_limit_reward_scale, mj_model=mj_model
#             ),
#         }
#
#         super().__init__()
#
#     def __call__(self, mj_data: MjData):
#         return super().__call__(**{k: self.rewards[k](mj_data) for k in self.rewards})
