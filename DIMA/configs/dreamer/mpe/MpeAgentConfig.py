from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.dreamer.DreamerAgentConfig import DreamerConfig
from agent.world_models.vq import StateDecoderType
from functools import partial

RSSM_STATE_MODE = 'discrete'


class MPEDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 3
        self.ACTION_HIDDEN = 128  # 256

        self.use_valuenorm = True
        self.use_huber_loss = True
        self.use_clipped_value_loss = True
        self.huber_delta = 10.0

        ## related to state decoder
        self.nums_obs_token = 12
        self.EMBED_DIM = 64 
        self.OBS_VOCAB_SIZE = 128
        self.ema_decay = 0.8
        self.alpha = 10.
        self.vq_type = 'fsq' # 'fsq', 'vq'
        self.state_decoder_type = StateDecoderType.OPTION1

        self.contdisc = False
        
        self.policy_class = 'beta'

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5

        self.rew_end_model_type = 'rnn' # 'rnn' or 'transformer'
