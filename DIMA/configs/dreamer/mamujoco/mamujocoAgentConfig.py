from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.dreamer.DreamerAgentConfig import DreamerConfig
from agent.world_models.vq import StateDecoderType

from functools import partial


class MAMujocoDreamerConfig(DreamerConfig):
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

        ## related to state decoder
        self.nums_obs_token = 12
        self.EMBED_DIM = 64 
        self.OBS_VOCAB_SIZE = 256  # 128
        self.ema_decay = 0.8
        self.alpha = 10.
        self.vq_type = 'vq' # 'fsq', 'vq'
        self.state_decoder_type = StateDecoderType.OPTION1

        self.contdisc = False
        
        self.policy_class = 'beta'

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5

        self.rew_end_model_type = 'transformer' # 'rnn' or 'transformer'
