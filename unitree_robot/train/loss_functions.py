from abc import ABC

from mujoco import MjData


# explanation for mjData fields: https://bhaswanth-a.github.io/posts/mujoco-basics/

class Loss(ABC):

    def __init__(self, normalzie: bool):
        self.normalize = normalzie

    def calculate_loss(self, data: MjData):
        pass
    


class BaseOrientationLoss(Loss):

    def __init__(self, normalzie: bool = True):
        super().__init__(normalzie=normalzie)

    def calculate_loss(self, data: MjData):
        data
        pass
