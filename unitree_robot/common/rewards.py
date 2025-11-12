from abc import ABC
import numpy as np
from typing import List
# from mujoco import MjData, MjModel
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R
from mujoco.mjx import Model as MjxModel
from mujoco.mjx import Data as MjData

# explanation for mjData fields: https://bhaswanth-a.github.io/posts/mujoco-basics/


def calc_angle(
    body_rotation_quat: NDArray[np.float32],
    angle_vector: NDArray[np.float32] = np.array([1.0, 0.0, 0.0]),
) -> float:
    rotated_angle_vector = R.from_quat(body_rotation_quat).apply(angle_vector)

    if np.isclose(rotated_angle_vector, angle_vector, atol=1e-4).all():
        return 0.0

    frac = np.dot(rotated_angle_vector, angle_vector) / (
        np.linalg.norm(rotated_angle_vector) * np.linalg.norm(angle_vector)
    )
    return np.arccos(frac) / np.pi


class Reward(ABC):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, reward: float) -> float:
        return reward * self.scale


class BaseOrientationReward(Reward):
    def __init__(
        self,
        body_index: str,
        scale: float,
        angle_vector: NDArray[np.float32] = np.array([1.0, 0.0, 0.0]),
    ):
        # self.body_name = body_name
        self.angle_vector = angle_vector
        super().__init__(scale=scale)

    def __call__(self, data: MjData) -> float:
        body_quat = data.body(self.body_name).xquat
        assert body_quat.sum() > 0, (
            "body rotation quaternion is not initialized at this point"
        )
        reward = 1 - calc_angle(
            body_rotation_quat=body_quat, angle_vector=self.angle_vector
        )
        return super().__call__(reward=reward)


class BodyDistanceReward(Reward):
    def __init__(
        self, body_names_from: List[str], body_names_to: List[str], scale: float = 1.0
    ):
        self.body_names_from = body_names_from
        self.body_names_to = body_names_to
        super().__init__(scale=scale)

    def __call__(self, data: MjData) -> float:
        from_positions = np.stack([data.body(n).xpos for n in self.body_names_from])
        to_positions = np.stack([data.body(n).xpos for n in self.body_names_to])

        dist = np.mean(np.abs(from_positions.mean(axis=0) - to_positions.mean(axis=0)))

        return super().__call__(dist)


class BodyHeightReward(BodyDistanceReward):
    def __init__(
        self,
        body_index_from: List[int] = [], # ["base_link"],
        body_index_to: List[int] = [], # ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
        scale: float = 1.0,
    ):
        super().__init__(
            # body_names_from=body_name_from, body_names_to=body_names_to, scale=scale
        )

    def __call__(self, data: MjData):
        return super().__call__(data=data)


class EnergyReward(Reward):
    def __init__(self, scale: float = 1.0):
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        actuator_force = np.mean(np.abs(data.actuator_force))
        return super().__call__(-actuator_force)


class JointLimitReward(Reward):
    def __init__(
        self,
        mjx_model: MjxModel,
        scale: float = 1.0,
        # joint_names: List = [
        #     "FL_calf_joint",
        #     "FL_hip_joint",
        #     "FL_thigh_joint",
        #     "FR_calf_joint",
        #     "FR_hip_joint",
        #     "FR_thigh_joint",
        #     "RL_calf_joint",
        #     "RL_hip_joint",
        #     "RL_thigh_joint",
        #     "RR_calf_joint",
        #     "RR_hip_joint",
        #     "RR_thigh_joint",
        # ],
    ):
        # self.joint_indecies = joint_indecies
        # self.joint_ranges_min, self.joint_ranges_max = np.stack(
        #     [mj_model.joint(n).range for n in joint_names]
        # ).T
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        current_joint_pos = np.concatenate(
            [data.joint(n).qpos for n in self.joint_names]
        )
        joint_min_max_scaled = (current_joint_pos - self.joint_ranges_min) / (
            self.joint_ranges_max - self.joint_ranges_min
        )
        joint_scaled_to_center = ((joint_min_max_scaled - 0.5) * 2) ** 2
        return super().__call__(-joint_scaled_to_center.mean())


class DistanceFromCenterReward(Reward):
    def __init__(self, scale: float = 1):
        super().__init__(scale)

    def __call__(self, data: MjData) -> float:
        pos = data.body("base_link").xpos
        mag = np.sqrt((pos**2).sum())

        return super().__call__(reward=mag)
