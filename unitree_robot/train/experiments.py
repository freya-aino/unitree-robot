from abc import ABC
from mujoco import MjData

from unitree_robot.train.rewards import (
    BaseOrientationReward,
    EnergyReward,
    BodyHeightReward,
    JointLimitReward,
)


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
        # joint_limit_reward_scale: float
    ):
        self.rewards = {
            # "base_height_reward": BodyHeightReward(scale=body_height_reward_scale),
            # "base_orientation_reward": BaseOrientationReward(
            #     body_name=body_name, scale=body_angle_reward_scale
            # ),
            "placeholder": lambda _: 0.0,
            # "energy_reward": EnergyReward(scale=energy_reward_scale),
        }
        # self.joint_limit_loss = JointLimitReward(scale=joint_limit_reward_scale)

        super().__init__()

    def __call__(self, mj_data: MjData):
        # stand_up_reward = self.base_orientation_reward(mj_data) + self.energy_reward(mj_data) + self.base_height_reward(mj_data)
        return super().__call__(**{k: self.rewards[k](mj_data) for k in self.rewards})
