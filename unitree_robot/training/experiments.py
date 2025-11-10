from abc import ABC
from mujoco.mjx import Data
from torch import Tensor, float32, from_numpy
import torch.nn.functional as F

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
    def __init__(self, initial_mjx_data: Data):
        self.qpos_target = from_numpy(initial_mjx_data.qpos.__array__().copy()).to(dtype=float32)
        self.qpos_target = self.qpos_target.unsqueeze_(0)
        super().__init__()
    def __call__(self, mjx_data: Data) -> Tensor:
        current_qpos = from_numpy(mjx_data.qpos.__array__().copy()).to(dtype=float32)
        loss = F.smooth_l1_loss(self.qpos_target.repeat([current_qpos.shape[0], 1]), current_qpos, reduction="none")
        return -loss.sum(dim=-1)

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
