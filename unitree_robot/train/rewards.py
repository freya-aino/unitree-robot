from abc import ABC
import numpy as np
from typing import List
from mujoco import MjData
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


# explanation for mjData fields: https://bhaswanth-a.github.io/posts/mujoco-basics/

def calc_angle(body_rotation_quat: NDArray[np.float32], angle_vector: NDArray[np.float32] = np.array([1.0,0.0,0.0])):
    rotated_angle_vector = R.from_quat(body_rotation_quat).apply(angle_vector)
    frac = np.dot(rotated_angle_vector, angle_vector) / (np.linalg.norm(rotated_angle_vector) * np.linalg.norm(angle_vector))
    return np.arccos(frac) / np.pi

class Reward(ABC):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def __call__(self, reward):
        return reward, reward * self.scale


class BaseOrientationReward(Reward):
    def __init__(self, body_name: str, scale: float, angle_vector: NDArray[np.float32] = np.array([1.0,0.0,0.0])):
        self.body_name = body_name
        self.angle_vector = angle_vector
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        body_quat = data.body(self.body_name).xquat
        # assert body_quat.sum() > 0, "body rotation quaternion is not initialized at this point"
        if not body_quat.sum() > 0:
            reward = 0
        else:
            reward = (1 - calc_angle(body_rotation_quat=body_quat, angle_vector=self.angle_vector))
        return super().__call__(reward=reward)


class BodyDistanceReward(Reward):

    def __init__(
        self,
        body_names_from: List[str],
        body_names_to: List[str],
        scale: float = 1.0
    ):
        self.body_names_from = body_names_from
        self.body_names_to = body_names_to
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        # target_height = 0.3  # TODO

        # global_pos = data.xpos
        # foot_z_mean_position = np.mean(global_pos[[5, 9, 13, 17], 2])  # global_pos[[5,9,13,17], 2] = z-values of foots
        # actual_height = global_pos[1, 2] - z_mean  # global_pos[1, 2] = z-value of the basis
        # error_height = np.abs(actual_height - target_height)

        # weight_height = 1.0 # TODO
        # loss_height = weight_height * np.abs(error_height)

        # case 1: actual_heigt < target_height -> error_height < 0 -> loss_height > 0
        # case 2: actual_heigt > target_height -> error_height > 0 -> loss_height > 0
        # case 3: actual_heigt = target_height -> error_height = 0 -> loss_height = 0

        from_positions = np.stack([data.body(n).xpos for n in self.body_names_from])
        to_positions = np.stack([data.body(n).xpos for n in self.body_names_from])

        dist = np.mean(np.abs(from_positions.mean(axis=0) - to_positions.mean(axis=0)))
        return super().__call__(dist)

class BodyHeightReward(BodyDistanceReward):
    def __init__(
            self,
            body_name_from: List[str] = ["base_link"],
            body_names_to: List[str] = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
            scale: float = 1.0
    ):
        super().__init__(
            body_names_from=body_name_from,
            body_names_to=body_names_to,
            scale=scale
        )

    def __call__(self, data: MjData):
        return super().__call__(data=data)


class EnergyReward(Reward):

    def __init__(self, scale: float = 1.0):
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        actuator_force = np.mean(np.abs(data.actuator_force))

        # weight_energy = 0.1 # TODO
        # loss_energy = weight_energy * jp.sum(jp.abs(actuator_force))

        return super().__call__(-actuator_force)


class JointLimitReward(Reward):

    def __init__(self, scale: float = 1.0):
        super().__init__(scale=scale)

    def __call__(self, data: MjData):
        qpos = data.qpos[7:]  # Gibt die Positions-Werte für die 12 Beingelenke aus
        jnt_range = model.jnt_range[1:]  # Gibt die Gelenk-Limits für die 12 Beingelenke aus
        jnt_center = (jnt_range[:, 0] + jnt_range[:, 1]) / 2  # jnt_range[:, 0]: Min-Werte aller Gelenke/ jnt_range[:, 1]: Max Werte aller Gelenke
        jnt_half_range = (jnt_range[:, 1] - jnt_range[:, 0]) / 2  # Berechnet die halbe Reichweite
        normalized_deviation = (qpos - jnt_center) / jnt_half_range  # Normalisiert die Abweichung zwischen der tatsächlichen Gelenkposition und dessen Center auf -1 bis 1
        loss = np.mean(np.abs(normalized_deviation))
        # TODO: quadrating the loss might be desirable - check when training

        return super().__call__(-loss)
