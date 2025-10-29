from abc import ABC
from mujoco import MjData

from unitree_robot.train.rewards import BaseOrientationReward

class Experiment(ABC):
    def __init__(self):
        pass
    def calculate_reward(self, **rewards):
        return {k: l for k, l in rewards.items()}
    def get_reward_shape(self):
        raise NotImplementedError

class StandUpExperiment(Experiment):
    def __init__(self, body_name: str, body_angle_reward_scale: float):
        self.base_orientation_reward = BaseOrientationReward(body_name=body_name, scale=body_angle_reward_scale)
        super().__init__()

    def calculate_reward(self, mj_data: MjData):
        return super().calculate_reward(
            body_angle = self.base_orientation_reward.calculate_reward(mj_data),
        )

    def get_reward_shape(self):
        return [1]

