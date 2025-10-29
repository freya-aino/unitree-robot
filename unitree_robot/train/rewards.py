from abc import ABC
import numpy as np
from mujoco import MjData
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R


# explanation for mjData fields: https://bhaswanth-a.github.io/posts/mujoco-basics/

def calc_angle(body_rotation_quat: NDArray[np.float32], angle_vector: NDArray[np.float32] = np.array([1.0,0.0,0.0])):
    rotated_angle_vector = R.from_quat(body_rotation_quat).apply(angle_vector)
    frac = np.dot(rotated_angle_vector, angle_vector) / (np.linalg.norm(rotated_angle_vector) * np.linalg.norm(angle_vector))
    return np.arccos(frac) / np.pi

class Loss(ABC):
    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def calculate_loss(self, data: MjData):
        pass
    
class BaseOrientationLoss(Loss):
    def __init__(self, body_name: str, scale: float, angle_vector: NDArray[np.float32] = np.array([1.0,0.0,0.0])):
        self.body_name = body_name
        self.angle_vector = angle_vector
        super().__init__(scale=scale)

    def calculate_loss(self, data: MjData):
        body_quat = data.body(self.body_name).xquat
        assert body_quat.sum() > 0, "body rotation quaternion is not initialized at this point"
        return self.scale * calc_angle(body_rotation_quat=body_quat, angle_vector=self.angle_vector)
