from abc import ABC
from mujoco import MjData

from unitree_robot.train.rewards import BaseOrientationReward, EnergyReward, BodyHeightReward, JointLimitReward

class Experiment(ABC):
    def __init__(self):
        pass
    def __call__(self, **rewards):
        return {k: r for k, r in rewards.items()}
    def __len__(self):
        raise NotImplementedError


class StandUpExperiment(Experiment):
    def __init__(
        self,
        body_name: str,
        body_angle_reward_scale: float,
        body_height_reward_scale: float,
        energy_reward_scale: float,
        joint_limit_reward_scale: float
    ):

        self.base_height_reward = BodyHeightReward(scale=body_height_reward_scale)
        self.base_orientation_reward = BaseOrientationReward(body_name=body_name, scale=body_angle_reward_scale)
        self.energy_reward = EnergyReward(scale=energy_reward_scale)
        # self.joint_limit_loss = JointLimitReward(scale=joint_limit_reward_scale)

        super().__init__()

    def __call__(self, mj_data: MjData):
        return super().__call__(
            body_angle = self.base_orientation_reward(mj_data),
            body_height = self.base_height_reward(mj_data),
            energy = self.energy_reward(mj_data),
            # joint_limit = self.joint_limit_loss(mj_data),
        )

    def __len__(self):
        return 3