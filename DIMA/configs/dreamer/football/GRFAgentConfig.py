from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config
from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class GRFDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_HIDDEN = 64    

        ## debug
        self.use_stack = True
        self.stack_obs_num = 4
