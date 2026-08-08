from agent.learners.DreamerLearner import DreamerLearner
from configs.dreamer.mpe.MpeAgentConfig import MPEDreamerConfig


class MPEDreamerLearnerConfig(MPEDreamerConfig):
    def __init__(self):
        super().__init__()
        # self.MODEL_LR = 2e-4
        # self.ACTOR_LR = 5e-4
        # self.VALUE_LR = 5e-4
        # self.CAPACITY = 500000 # The buffer length is exactly the maximum training steps
        # self.MIN_BUFFER_SIZE = 100
        # self.MODEL_EPOCHS = 1
        # self.EPOCHS = 1
        # self.PPO_EPOCHS = 5
        # self.MODEL_BATCH_SIZE = 40
        # self.BATCH_SIZE = 40
        # self.SEQ_LENGTH = 50
        # self.N_SAMPLES = 1
        # self.TARGET_UPDATE = 1
        # self.DEVICE = 'cpu'
        # self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        # self.ENTROPY = 0.001
        # self.ENTROPY_ANNEALING = 0.99998
        # self.GRAD_CLIP_POLICY = 100.

        # optimal smac config
        self.MODEL_LR = 2e-4
        self.ACTOR_LR = 5e-4  # 5e-4
        self.VALUE_LR = 5e-4  # 5e-4
        self.CAPACITY = 250000
        self.MIN_BUFFER_SIZE = 5000 # 500
        self.MODEL_EPOCHS = 100 # 60
        self.WM_EPOCHS = 200  # 200
        self.PPO_EPOCHS = 5
        self.MODEL_BATCH_SIZE = 40 # 40; 27m bs should be 10, agents_num ~ 10 should be 20
        self.BATCH_SIZE = 30 # 40; 27m bs should be 8, agents_num ~ 10 should be 20
        self.ac_batch_size = 600  # 600
        # self.SEQ_LENGTH = 20
        self.SEQ_LENGTH = self.horizon
        
        self.N_SAMPLES = 200  # 1
        self.EPOCHS = 5 # 4; 27m epochs should be 20, agents_num ~ 10 should be 20

        self.TARGET_UPDATE = 20  # 1
        self.clip_param = 0.2
        self.DEVICE = 'cuda'
        self.GRAD_CLIP = 100.0
        # self.HORIZON = 15
        self.ENTROPY = 0.001
        self.ENTROPY_ANNEALING = 1.0
        self.GRAD_CLIP_POLICY = 10.0

        # tokenizer
        ## batch size
        self.t_bs = 512
        ## learning rate
        self.t_lr = 1e-4

        # world model
        ## batch size
        self.wm_bs = 64
        ## learning rate
        self.wm_lr = 1e-4 # 5e-4
        self.wm_weight_decay = 0.01

        self.max_grad_norm = 10.0

        self.sample_temperature = 'inf'

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

        ### Autoencoder learning params
        self.ae_grad_acc_steps = 1
        self.ae_max_grad_norm = 10.0 # maybe 10.0 would be better
        self.ae_steps_first_epoch = 5000
        self.ae_lr = 0.0001
        self.ae_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }

        ### Denoiser learning params
        self.grad_acc_steps = 1
        self.ema_decay = 0.995
        self.ema_update_every = 10
        self.denoiser_opt_mode = 'robodreamer'
        self.denoiser_max_grad_norm = 1.0 # 10.0
        self.denoiser_steps_first_epoch = 200 # 10000
        self.denoiser_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }
        self.denoiser_lr_warmup_steps = 100

        ### rew_end_model learning params
        self.remodel_steps_first_epoch = 60   # 200  LSTM-based seems better at 60
        self.remodel_steps = 60
        self.rew_end_model_opt_cfg = {
            'lr': 0.0001,
            'weight_decay': 0.01,
            'eps': 1e-08,
        }
        self.remodel_lr_warmup_steps = 100
        self.remodel_max_grad_norm = 10.

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
        self.ac_max_grad_norm = 10.
        self.clip_param = 0.1

        self.compute_end_in_TD = True

        self.offline_epochs = 20

    def create_learner(self):
        return DreamerLearner(self)
