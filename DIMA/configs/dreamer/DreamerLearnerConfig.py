from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.DreamerAgentConfig import DreamerConfig


class DreamerLearnerConfig(DreamerConfig):
    def __init__(self):
        super().__init__()
        # optimal smac config
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4  # 5e-4
        self.VALUE_LR = 5e-4  # 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 5000 # 500
        self.MODEL_EPOCHS = 200 # 60
        self.WM_EPOCHS = 50  # DIMA: 200, at a quarter of the batch size
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 30 # 40; 27m bs should be 10, agents_num ~ 10 should be 20
        self.BATCH_SIZE = 30 # 40; 27m bs should be 8, agents_num ~ 10 should be 20

        # Batch sizes of the three world-model training loops.  DIMA hardcoded these
        # (256 / 64 / 128); at those sizes a modern GPU sits mostly idle waiting on the
        # single-threaded sampling path, so they are raised here and the matching epoch
        # counts below are divided by the same factor -- same samples seen per round,
        # a quarter of the optimizer steps and sampling calls.
        self.state_decoder_batch_size = 1024
        self.denoiser_batch_size = 256
        self.rew_end_batch_size = 512
        # self.ac_batch_size = 600  # 600
        self.SEQ_LENGTH = self.horizon
        
        self.N_SAMPLES = 100  # 1
        self.EPOCHS = 5 # 4; 27m epochs should be 20, agents_num ~ 10 should be 20

        self.TARGET_UPDATE = 20  # 1
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        self.ENTROPY = 0.001 # 0.001  # with larger 0.01, we can obtain a little bit better performance on 2m_vs_1z
        self.ENTROPY_ANNEALING = 0.99998  # 1.0
        self.GRAD_CLIP_POLICY = 10.0        # 100.0

        self.sample_temperature = 'inf'

        self.max_grad_norm = 10.0

        ## control whether average the predicted rewards
        self.critic_average_r = False

        ## discrete regression
        self.critic_dist_config = {
            'symlog_transform': False,
            'loss_type': 'regression', # 'regression' | 'hlgauss'
            'min_v': -10., 
            'max_v': 10.,
            'bins': 21, # 51
        }
        self.tau = 0.5

        ### Denoiser learning params
        self.grad_acc_steps = 1
        self.denoiser_max_grad_norm = 1.0
        self.denoiser_steps_first_epoch = 50  # DIMA: 200, at a quarter of the batch size
        self.denoiser_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }
        self.denoiser_lr_warmup_steps = 100

        ### rew_end_model learning params
        self.remodel_steps_first_epoch = 15   # DIMA: 60, at a quarter of the batch size
        self.remodel_steps = 15
        self.rew_end_model_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }
        self.remodel_lr_warmup_steps = 100
        self.remodel_max_grad_norm = 100.

        ### World model env params
        self.ac_batch_size = 600 # 32
        self.update_manner = "REINFORCE" # "REINFORCE" | "PPO"
        self.ac_steps_first_epoch = 5 # 250 # 5000
        self.ac_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }
        self.ac_lr_warmup_steps = 100
        self.ac_max_grad_norm = 100.  # 10.
        self.clip_param = 0.2

        self.compute_end_in_TD = True # False
        

    def create_learner(self):
        return DreamerLearner(self)
