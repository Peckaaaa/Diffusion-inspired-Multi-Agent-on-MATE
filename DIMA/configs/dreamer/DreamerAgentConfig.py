from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

from agent.world_models.diffusion.inner_model import StateInnerModelConfig
from agent.world_models.diffusion.denoiser import DenoiserConfig, SigmaDistributionConfig
from agent.world_models.diffusion.diffusion_sampler import DiffusionSamplerConfig
from agent.world_models.perceiver import PerceiverConfig
from agent.world_models.rew_end_model import StateRewEndModelConfig, TransformerConfig
from agent.world_models.world_model_env import WorldModelEnvConfig

from functools import partial

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        self.LOG_FOLDER = 'wandb/'

        # optimal smac config
        self.HIDDEN = 256
        self.MODEL_HIDDEN = 256
        self.EMBED = 256
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 256
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 3 # 2
        self.ACTION_HIDDEN = 128 # 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 256
        self.GAMMA = 0.99  # discount factor
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95  # lambda in dreamer v2
        self.IN_DIM = 30

        self.num_mini_batch = 1
        self.use_valuenorm = True           # False
        self.use_huber_loss = True          # False
        self.use_clipped_value_loss = True  # False
        self.huber_delta = 10.0

        self.contdisc = False

        self.nums_obs_token = 12
        self.EMBED_DIM = 64 
        self.OBS_VOCAB_SIZE = 128
        self.ema_decay = 0.8
        self.alpha = 10.

        # Indices, into the flattened joint observation the state decoder
        # reconstructs, of channels that are 0/1 flags rather than continuous
        # values.  The reconstruction loss is L1, whose optimum is the conditional
        # *median*: for a flag that is 1 only a minority of the time the median is
        # 0, so L1 pins those channels at 0 and they carry no information about
        # how likely the flag is.  Scoring them with a squared error instead puts
        # the optimum at the conditional mean, which is the probability itself.
        #
        # Left as None for environments that do not name any such channel, in
        # which case the loss is unchanged.  MATE fills it in via
        # research/config.py.
        self.obs_binary_indices = None
        # Weight on that block, relative to the continuous one.  The flags are a
        # small fraction of the observation, so a decoder short of capacity is
        # right to spend it elsewhere; raising this buys flag accuracy with
        # continuous accuracy.
        self.obs_binary_loss_weight = 1.0
        # Give those channels their own decoder branch, emitting logits scored
        # with BCEWithLogitsLoss instead of sharing the regression head.  See
        # agent/world_models/vq.py:_BinaryFlagHead and
        # DreamerLearner._reconstruction_loss.  Off by default so DIMA's own
        # environments and every checkpoint written before this are unaffected.
        self.obs_binary_head = False
        # Weight on the positive class of that loss.  A flag positive a fraction
        # p of the time is balanced at (1 - p) / p; MATE-4v8-9 measures p = 0.136,
        # so 6.0.
        self.obs_binary_pos_weight = 6.0

        # What the actor's lambda-return is built from.
        #
        # 'model' uses the learned reward head, which is what DIMA does.  On MATE
        # that head is trained on a team reward that is zero on most steps, and
        # measured on a trained checkpoint it moves by 1.0% of its state-to-state
        # spread when the action changes -- too little to tell the actor which
        # action was better.  research/planners.py reached the same conclusion for
        # planning and scores candidates by coverage instead; this makes the same
        # choice available to behaviour learning.
        #
        # 'coverage' reads the fraction of opponents at least one agent observes
        # directly out of the imagined observation, using the mask channels named
        # by imagined_coverage_indices.  It is dense, bounded in [0, 1], and
        # depends on the action through the imagined state.
        self.imagined_reward = 'model'
        # Indices, within a single agent's observation, of the per-opponent
        # presence flags.  Environment-specific, so the research layer fills it in.
        self.imagined_coverage_indices = None
        self.vq_type = 'vq' # 'fsq', 'vq'

        self.policy_class = 'discrete'

        ## denoiser params
        self.cond_channels = 256

        # How the conditioning actions reach the denoiser: 'sequential' is DIMA's
        # original masked, one-agent-per-denoising-step scheme, 'joint' feeds the
        # whole joint action at every step.  See
        # agent/world_models/perceiver.py:JointActionEmb.
        self.action_cond = 'sequential'
        self.agent_action_embed_dim = 64

        self.inner_model_cfg = StateInnerModelConfig(
            state_dim = -1,
            num_steps_conditioning = 3,
            cond_channels = self.cond_channels,
            depths = [2, 2, 2],
            channels = [64, 64, 64],
            attn_depths = [0, 0, 0],
            action_dim = -1,
            action_cond = self.action_cond,
            agent_action_embed_dim = self.agent_action_embed_dim,
        )

        self.perceiver_cfg = PerceiverConfig(
            dim = self.cond_channels,
            latent_dim = 512,
            num_latents = 32,  # 256
            depth = 2,
            cross_heads = 1,
            cross_dim_head = 64,
            latent_heads = 8,
            latent_dim_head = 64,
            attn_dropout = 0.,  # 0.1
            ff_dropout = 0.,    # 0.1
            output_dim = self.cond_channels,
            final_proj_head = True,
        )

        self.denoiser_cfg = DenoiserConfig(
            inner_model = self.inner_model_cfg,
            perceiver = self.perceiver_cfg,
            sigma_data = 0.5,
            sigma_offset_noise = 0.3,
        )

        self.num_autoregressive_steps = 3   # 1

        self.sigma_distribution = SigmaDistributionConfig(
            loc = -0.4,
            scale=1.2,
            sigma_min=2e-3,
            sigma_max=20., # 20  # here we may need to set sigma_max equal to that during inference
        )

        ## Rew_And_End_model params
        # self.rewendmodel_cfg = StateRewEndModelConfig(
        #     lstm_dim = 512,
        #     num_enc_layers = 2,
        #     enc_dim = 128,      # 256
        #     latent_dim = 256,   # 512
        #     simnorm_dim = 8,
        #     mlp_dim = 256,      # 512
        # )

        self.rewendmodel_cfg = StateRewEndModelConfig(
            lstm_dim = 512,
            cond_channels = 128,
            depths = [2],
            dim = 128,
            dim_mults = [1],
            attn_depths = [0],
            mlp_dim = 512,
        )

        self.contdisc = False

        ## World Model Env params
        self.horizon = 15 # 25
        self.diffusion_sampler_cfg = DiffusionSamplerConfig(
            num_steps_denoising=3,  # equal to number of agent
            sigma_min=2e-3,
            sigma_max=5.0,
            rho=7,
            order=1,
            s_churn=0.0,
            s_tmin=0.0,
            s_tmax=float('inf'),
            s_noise=1.0,
            agent_order="default"  # "default" | "reverse" | "random"
        )

        self.worldmodel_env_cfg = WorldModelEnvConfig(
            horizon=self.horizon,
            num_batches_to_preload=10,
            diffusion_sampler=self.diffusion_sampler_cfg,
        )

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5

        # for TransRewEndModel
        self.TRANS_EMBED_DIM = 256 # 256
        self.HEADS = 4
        self.DROPOUT = 0.1
        self.trans_config = TransformerConfig(
            tokens_per_block=2,
            max_blocks=self.horizon,
            attention='causal',
            num_layers=6, # 10
            num_heads=self.HEADS,
            embed_dim=self.TRANS_EMBED_DIM,
            embed_pdrop=self.DROPOUT,
            resid_pdrop=self.DROPOUT,
            attn_pdrop=self.DROPOUT,
        )

        self.rew_end_model_type = 'rnn' # 'rnn' or 'transformer'



@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
