from enum import Enum


class Env(str, Enum):
    FLATLAND = "flatland"
    STARCRAFT = "starcraft"
    PETTINGZOO = "pettingzoo"
    GRF = "football"
    MAMUJOCO = "mamujoco"
    SMACv2 = "SMACv2"
    MATE = "mate"  # added by the DIMA x MATE research layer; see research/UPSTREAM_PATCHES.md

RANDOM_SEED = 23
