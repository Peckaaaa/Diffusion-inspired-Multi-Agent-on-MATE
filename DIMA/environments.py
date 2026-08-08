from enum import Enum


class Env(str, Enum):
    FLATLAND = "flatland"
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    SMACv2 = "SMACv2"
    MATE = "mate"

RANDOM_SEED = 23
