from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.dreamer.DreamerAgentConfig import DreamerConfig

from functools import partial


class Smacv2DreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 3
        self.ACTION_HIDDEN = 128  # 256

        self.use_bin = False
        self.bins = 256
        self.action_bins = 256

        self.use_valuenorm = True
        self.use_huber_loss = True
        self.use_clipped_value_loss = True
        self.huber_delta = 10.0

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5
