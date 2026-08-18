from collections import defaultdict, deque
from copy import deepcopy
import random
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from einops import rearrange

# from agent.world_models.diffusion.denoiser import Denoiser
from agent.world_models.vq import SimpleFSQAutoEncoder, SimpleVQAutoEncoder, StateDecoderType

from networks.dreamer.action import Actor,StochasticPolicy
from environments import Env
from utils import discretize_into_bins, bins2continuous, obs_bins2continuous, symexp, symlog, obs_split_into_bins


class DreamerController:

    def __init__(self, config):
        self.config = config
        self.env_type = config.ENV_TYPE

        self.config.denoiser_cfg.inner_model.state_dim  = config.STATE_DIM
        self.config.denoiser_cfg.inner_model.action_dim = config.ACTION_SIZE

        # if config.state_decoder_type == StateDecoderType.OPTION1:
        #     tokenizer_in_dim = config.STATE_DIM + config.NUM_AGENTS
        # else:
        #     tokenizer_in_dim = config.STATE_DIM + config.IN_DIM

        # if config.vq_type == 'fsq':
        #     levels = [8, 6, 5]  # [8, 5, 5, 5]
        #     self.state_decoder = SimpleFSQAutoEncoder(in_dim=tokenizer_in_dim, num_tokens=config.nums_obs_token, output_dim=config.IN_DIM,
        #                                               levels=levels).eval()

        # else:
        #     self.state_decoder = SimpleVQAutoEncoder(in_dim=tokenizer_in_dim, embed_dim=config.EMBED_DIM, num_tokens=config.nums_obs_token, output_dim=config.IN_DIM,
        #                                              codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True, decay=config.ema_decay).eval()


        ac_input_dim = config.IN_DIM if not self.config.use_stack else config.IN_DIM * config.stack_obs_num  # take rec obs as input
        if self.env_type == Env.STARCRAFT:
            self.actor = Actor(ac_input_dim, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).eval()
        else:
            self.actor = StochasticPolicy(ac_input_dim, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                          continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE, policy_class=config.policy_class).eval()
        
        self.eps = config.epsilon
        
        self.expl_decay = config.EXPL_DECAY
        self.expl_noise = config.EXPL_NOISE
        self.expl_min = config.EXPL_MIN
        self.init_rnns()
        self.init_buffer()

        self.temp = config.temperature
        self.use_continuous_action = config.CONTINUOUS_ACTION
        if hasattr(config, 'determinisitc'):
            self.use_deterministic_action = config.determinisitc
        else:
            self.use_deterministic_action = self.use_continuous_action

        self.running_mean_std = None

    def receive_params(self, params):
        # self.state_decoder.load_state_dict(params['state_decoder'])
        self.actor.load_state_dict(params['actor'])
        self.running_mean_std = params['running_mean_std']

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def init_rnns(self):
        self.prev_rnn_state = None
        self.prev_actions = None
        
        if self.config.use_stack:
            self.stack_obs = deque(maxlen=self.config.stack_obs_num)
            for _ in range(self.config.stack_obs_num):
                self.stack_obs.append(
                    torch.zeros(1, self.config.NUM_AGENTS, self.config.IN_DIM)
                )

    def dispatch_buffer(self):
        total_buffer = {k: np.asarray(v, dtype=np.float32) for k, v in self.buffer.items()}
        last = np.zeros_like(total_buffer['done'])
        last[-1] = 1.0
        total_buffer['last'] = last
        self.init_rnns()
        self.init_buffer()
        return total_buffer

    def update_buffer(self, items):
        for k, v in items.items():
            if v is not None:
                self.buffer[k].append(v.squeeze(0).detach().clone().numpy())
    
    @torch.no_grad()
    def step(self, observations, shared_obs, avail_actions, nn_mask):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        if self.config.use_stack:
            self.stack_obs.append(observations)
            feats = rearrange(torch.cat(list(self.stack_obs), dim=1), 'n b e -> 1 n (b e)')

        else:
            feats = observations

        if self.use_continuous_action:
            action, log_probs = self.actor(feats, deterministic = self.use_deterministic_action)
            ent = self.actor.action_std_print.clone().mean(-1)
        else:
            action, pi = self.actor(feats)
            if avail_actions is not None:
                pi[avail_actions == 0] = -1e10  # logits

            pi = pi / self.temp  # softmax temperature
            
            probs = F.softmax(pi, -1)
            ent = -((probs * torch.log(probs + 1e-6)).sum(-1))            
            
            if self.use_deterministic_action:
                # Greedy. Until now the discrete branch ignored use_deterministic_action
                # entirely (it is only read in the continuous branch above), so the eval
                # workers -- which DreamerRunner explicitly configures with
                # determinisitc=True -- were sampling exactly like the training workers.
                # With entropy ~2.85 of a ln(25)=3.22 maximum the policy spreads its mass
                # over ~exp(2.85)=17 of 25 actions, so sampling discards most of whatever
                # preference it had learned.
                action = F.one_hot(pi.argmax(-1), pi.size(-1)).to(pi.dtype)
            else:
                action_dist = OneHotCategorical(logits=pi)
                action = action_dist.sample()
            
            # epsilon exploration
            if random.random() < self.eps:
                action_dist = OneHotCategorical(probs=avail_actions / avail_actions.sum(-1, keepdim=True))
                action = action_dist.sample()
        
        return action.squeeze(0).clone(), ent.squeeze(0).clone()

    def advance_rnns(self, state):
        self.prev_rnn_state = deepcopy(state)

    def exploration(self, action):
        """
        :param action: action to take, shape (1,)
        :return: action of the same shape passed in, augmented with some noise
        """
        for i in range(action.shape[0]):
            if np.random.uniform(0, 1) < self.expl_noise:
                index = torch.randint(0, action.shape[-1], (1, ), device=action.device)
                transformed = torch.zeros(action.shape[-1])
                transformed[index] = 1.
                action[i] = transformed
        self.expl_noise *= self.expl_decay
        self.expl_noise = max(self.expl_noise, self.expl_min)
        return action
