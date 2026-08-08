from configs.dreamer.DreamerAgentConfig import DreamerConfig
from agent.world_models.vq import StateDecoderType

RSSM_STATE_MODE = 'discrete'


class MATEDreamerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        # DiscreteCamera(levels=5) -> Discrete(25) per camera.
        # Overwritten by get_env_info() anyway, kept for a sane default.
        self.ACTION_SIZE = 25
        self.ACTION_LAYERS = 3
        self.ACTION_HIDDEN = 128

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

        # MATE cameras use a discrete action set, unlike MPE's default 'beta'.
        self.policy_class = 'discrete'

        ## debug
        self.use_stack = False
        self.stack_obs_num = 5

        # Must be 'transformer': WorldModelEnv drives the reward/end model through
        # rew_end_model.config / .transformer / KV caching, which only
        # TransRewEndModel provides.  With 'rnn' the actor-critic imagination
        # rollout raises AttributeError as soon as it starts.
        self.rew_end_model_type = 'transformer' # 'rnn' or 'transformer'
