from enum import Enum
from pydantic.dataclasses import dataclass

import numpy as np

class BACKENDS(Enum):
    GENERLIZED = "generalized" # high realism, low speed
    POSITIONAL = "positional" # medium realism, medium speed
    SPRING = "spring" # low realism, high speed

class NETWORK_INTERFACE(Enum):
   LOCAL = "lo"
   LAPTOP_1 = "enp0s13f0u1"

# @dataclass
# class LiDARPointField:
#     offset: np.float32
#     datatype: np.uint8
#     count: types.uint32

#     name: str = "sensor_msgs.msg.dds_.PointField_"
