from enum import Enum

class BACKENDS(Enum):
    GENERLIZED = "generalized" # high realism, low speed
    POSITIONAL = "positional" # medium realism, medium speed
    SPRING = "spring" # low realism, high speed
