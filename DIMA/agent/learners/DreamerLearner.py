import os
import sys
from copy import deepcopy
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Dict
from termcolor import cprint
from einops import rearrange, repeat
from torch.utils.data.dataloader import DataLoader

import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from agent.memory.DreamerMemory import DreamerMemory, ObsDataset
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout, continuous_actor_loss
from agent.optim.loss import huber_loss, mse_loss, rollout_diffusion_world_models
from agent.optim.utils import advantage
from agent.optim.utils import compute_return as compute_return_mamba
from agent.utils.valuenorm import ValueNorm
from agent.utils.running_mean_std import RunningMeanStd

from environments import Env
from networks.dreamer.action import Actor, StochasticPolicy
from networks.dreamer.critic import AugmentedCritic, Critic, FeatureNormedAugmentedCritic, VNet

# world model related
from agent.world_models.diffusion import Denoiser, DiffusionSampler
from agent.world_models.vq import SimpleFSQAutoEncoder, SimpleVQAutoEncoder, StateDecoderType
from agent.world_models.rew_end_model import RewEndModel, TransRewEndModel
from agent.world_models.world_model_env import WorldModelEnv
from agent.world_models.actor_critic import compute_lambda_returns, normalize_advantage, compute_lambda_returns_with_pcont_wo_end
from agent.coroutines.env_loop import make_env_loop

from utils import configure_optimizer
from utils import CommonTools, configure_opt, get_lr_sched, count_parameters, wandb_log, mujoco_visualization
from episode import SC2Episode, MpeEpisode, GRFEpisode, MamujocoEpisode
from dataset import MultiAgentEpisodesDataset, convert_to_batch
from tb_logger import LOGGER

import wandb
import ipdb

def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config):
        self.config = config
        self.env_type = config.ENV_TYPE

        torch.autograd.set_detect_anomaly(True)

        self.replay_buffer = MultiAgentEpisodesDataset(max_ram_usage="30G", name="train_dataset", sample_weights=[0.1, 0.1, 0.1, 0.7],
                                                       capacity          = config.CAPACITY,
                                                       diffusion_seq_len = config.denoiser_cfg.inner_model.num_steps_conditioning + 1 + 2,
                                                       condition_steps   = config.denoiser_cfg.inner_model.num_steps_conditioning,
                                                       sample_temp       = config.sample_temperature,)

        self.config.denoiser_cfg.inner_model.state_dim  = config.STATE_DIM
        self.config.denoiser_cfg.inner_model.action_dim = config.ACTION_SIZE

        self.denoiser = Denoiser(
            self.config.denoiser_cfg,
            num_agents        = config.NUM_AGENTS,
            clip_denoised     = False, # (self.env_type in [Env.STARCRAFT, Env.SMACv2]),
            is_continuous_act = config.CONTINUOUS_ACTION).to(config.DEVICE).eval()
        self.denoiser.setup_training(self.config.sigma_distribution)

        # self.global_state_normalizer = ValueNorm(config.STATE_DIM, device=config.DEVICE)
        self.state_rms = RunningMeanStd(shape=(config.STATE_DIM))

        # state decoder
        if config.state_decoder_type == StateDecoderType.OPTION1:
            cprint(f"Using Option 1 state decoder: s_t -> joint o_t.", "cyan", attrs=["bold"])
            tokenizer_in_dim = config.STATE_DIM
        else:
            cprint(f"Using Option 2 state decoder: s_t + o_t-1 -> o_t.", "yellow", attrs=["bold"])
            tokenizer_in_dim = config.STATE_DIM + config.IN_DIM

        if config.vq_type == 'fsq':
            levels = [8, 6, 5]  # [8, 5, 5, 5]
            self.state_decoder = SimpleFSQAutoEncoder(in_dim=tokenizer_in_dim, num_tokens=config.nums_obs_token, output_dim=config.NUM_AGENTS * config.IN_DIM,
                                                      levels=levels).to(config.DEVICE).eval()
            self.obs_vocab_size = np.prod(levels)

        else:
            self.state_decoder = SimpleVQAutoEncoder(in_dim=tokenizer_in_dim, embed_dim=config.EMBED_DIM, num_tokens=config.nums_obs_token, output_dim=config.NUM_AGENTS * config.IN_DIM,
                                                     codebook_size=config.OBS_VOCAB_SIZE, learnable_codebook=False, ema_update=True, decay=config.ema_decay).to(config.DEVICE).eval()
            self.obs_vocab_size = config.OBS_VOCAB_SIZE

        cprint(f"Using {config.rew_end_model_type}-based rew & end model.", "cyan", attrs=["bold"])
        if config.rew_end_model_type == "rnn":
            self.rew_end_model = RewEndModel(
                self.config.rewendmodel_cfg,
                num_agents=config.NUM_AGENTS,
                state_dim=config.STATE_DIM,
                action_dim=config.ACTION_SIZE,
                is_continuous_act=config.CONTINUOUS_ACTION,
                # -----Divider------
                pred_shared_reward=True,
                pred_shared_continuation=True,
                pred_av_action=(self.env_type in [Env.STARCRAFT, Env.SMACv2]),
                use_ce_for_cont=config.use_ce_for_cont,
            ).to(config.DEVICE).eval()

        else:
            self.rew_end_model = TransRewEndModel(
                state_dim=config.STATE_DIM,
                act_vocab_size=config.ACTION_SIZE,
                num_agents=config.NUM_AGENTS,
                config=config.trans_config,
                action_dim=config.ACTION_SIZE,
                is_discrete_action=not config.CONTINUOUS_ACTION,
                use_ce_for_end=config.use_ce_for_cont,
                use_ce_for_av_action=True,
                enable_av_pred=(self.env_type in [Env.STARCRAFT, Env.SMACv2]),
            ).to(config.DEVICE).eval()

        ac_input_dim = config.IN_DIM if not self.config.use_stack else config.IN_DIM * config.stack_obs_num  # take rec obs as input
        # critic_output_shape = () if config.critic_dist_config['loss_type'] == 'regression' else config.critic_dist_config['bins']
        
        if self.env_type == Env.STARCRAFT:
            self.actor = Actor(ac_input_dim, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to(config.DEVICE)
        else:
            self.actor = StochasticPolicy(ac_input_dim, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS,
                                          continuous_action=config.CONTINUOUS_ACTION, continuous_action_space=config.ACTION_SPACE, policy_class=config.policy_class).to(config.DEVICE)

        self.critic = VNet(config.STATE_DIM + config.NUM_AGENTS * config.ACTION_SIZE, config.VALUE_HIDDEN, config.VALUE_LAYERS).to(config.DEVICE)

        self.value_normalizer = ValueNorm(1, device=config.DEVICE)
        self.use_valuenorm = config.use_valuenorm

        if self.use_valuenorm:
            cprint(f"Use value normalization.", "cyan", attrs=["bold"])
        else:
            cprint("Disable value normalization.", "yellow", attrs=["bold"])

        self.num_batch_train = CommonTools(0, 0, 0, 0)

        self.update_manner = self.config.update_manner
        if not config.CONTINUOUS_ACTION and self.env_type == Env.STARCRAFT:
            initialize_weights(self.actor)
            # initialize_weights(self.critic, mode='xavier')

        self.old_critic = deepcopy(self.critic)
        self.mamba_replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, config.STATE_DIM,
                                                 config.NUM_AGENTS, config.DEVICE, config.ENV_TYPE, config.sample_temperature)

        self.entropy = config.ENTROPY
        self.step_count = -1
        self.train_count = 0
        self.cur_wandb_epoch = 0
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)

        self.tqdm_vis = True
        self.use_valuenorm = config.use_valuenorm
        self.use_huber_loss = config.use_huber_loss
        self.use_clipped_value_loss = config.use_clipped_value_loss

        print("")
        print(f"{count_parameters(self.denoiser)} parameters in denoiser")
        print(f"{count_parameters(self.rew_end_model)} parameters in rew_end_model")
        print(f"{count_parameters(self.actor)} parameters in actor")
        print(f"{count_parameters(self.critic)} parameters in critic")
        print("")

        self._binary_index_tensor = None
        self._continuous_index_tensor = None

        self.train_ac_only = False
        self.evaluate = False
        if config.load_pretrained:
            assert config.load_path is not None
            self.load_pretrained(config.load_path)

    def init_optimizers(self):
        self.state_decoder_optimizer = torch.optim.AdamW(self.state_decoder.parameters(), lr=3e-4)

        self.denoiser_opt = configure_opt(self.denoiser, **self.config.denoiser_opt_cfg)
        self.denoiser_lr_sched = get_lr_sched(self.denoiser_opt, self.config.denoiser_lr_warmup_steps)

        if self.config.rew_end_model_type == 'rnn':
            self.rew_end_model_opt = configure_opt(self.rew_end_model, **self.config.rew_end_model_opt_cfg)
            # self.rew_end_model_lr_sched = get_lr_sched(self.rew_end_model_opt, self.config.remodel_lr_warmup_steps)

        else:
            self.rew_end_model_opt = configure_optimizer(self.rew_end_model, self.config.rew_end_model_opt_cfg['lr'], self.config.rew_end_model_opt_cfg['weight_decay'])

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR,
                                                 weight_decay=0.0 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 0.00001,
                                                 eps=1e-5 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 1e-8)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR,
                                                 weight_decay=0.0 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 0.00001,
                                                 eps=1e-5 if self.env_type in [Env.PETTINGZOO, Env.GRF, Env.MAMUJOCO] else 1e-8)

    def params(self):
        return {'state_decoder': {k: v.cpu() for k, v in self.state_decoder.state_dict().items()},
                'rew_end_model': {k: v.cpu() for k, v in self.rew_end_model.state_dict().items()},
                'denoiser': {k: v.cpu() for k, v in self.denoiser.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()},
                'running_mean_std': self.state_rms.copy(),
            }
        
    def load_pretrained(self, load_path):
        print(f"Loading from {load_path}")
        ckpt = torch.load(load_path)

        if 'state_decoder' in ckpt:
            self.state_decoder.load_state_dict(ckpt['state_decoder'])

        if 'denoiser' in ckpt:
            self.denoiser.load_state_dict(ckpt['denoiser'])

        if 'rew_end_model' in ckpt:
            self.rew_end_model.load_state_dict(ckpt['rew_end_model'])

        self.state_decoder.eval()
        self.denoiser.eval()
        self.rew_end_model.eval()

        self.train_ac_only = True
        if 'actor' in ckpt:
            self.actor.load_state_dict(ckpt['actor'])

            self.train_ac_only = False
            self.evaluate = True

    def save(self, save_path):
        torch.save(self.params(), save_path)

    # ---- full training-state checkpointing (resume after an interrupted run) ----

    # Everything that has to survive a restart for training to continue identically.
    _CKPT_MODULES = ('state_decoder', 'denoiser', 'rew_end_model', 'actor', 'critic',
                     'old_critic', 'value_normalizer')
    _CKPT_OPTIMIZERS = ('state_decoder_optimizer', 'denoiser_opt', 'rew_end_model_opt',
                        'actor_optimizer', 'critic_optimizer')
    _CKPT_COUNTERS = ('step_count', 'train_count', 'cur_wandb_epoch', 'cur_update',
                      'accum_samples', 'total_samples', 'n_agents', 'entropy',
                      'train_ac_only', 'evaluate')

    def training_state(self, save_buffer=False):
        state = {
            'modules': {name: getattr(self, name).state_dict() for name in self._CKPT_MODULES},
            'optimizers': {name: getattr(self, name).state_dict() for name in self._CKPT_OPTIMIZERS},
            'counters': {name: getattr(self, name) for name in self._CKPT_COUNTERS},
            'state_rms': {'mean': self.state_rms.mean, 'var': self.state_rms.var, 'count': self.state_rms.count},
            'num_batch_train': self.num_batch_train.state_dict(),
            'torch_rng': torch.get_rng_state(),
            'numpy_rng': np.random.get_state(),
        }

        if self.denoiser_lr_sched is not None:
            state['denoiser_lr_sched'] = self.denoiser_lr_sched.state_dict()

        if torch.cuda.is_available():
            state['cuda_rng'] = torch.cuda.get_rng_state_all()

        if save_buffer:
            state['replay_buffer'] = self.replay_buffer.state_dict()
            state['mamba_replay_buffer'] = self.mamba_replay_buffer.state_dict()

        return state

    def save_full(self, save_path, runner_state=None, save_buffer=False):
        """Write a resumable checkpoint atomically, so an interrupt mid-write
        cannot leave a truncated file where the previous good one was."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        ckpt = {'learner': self.training_state(save_buffer=save_buffer),
                'runner': runner_state,
                'has_buffer': save_buffer}

        tmp_path = save_path.with_suffix(save_path.suffix + '.tmp')
        torch.save(ckpt, tmp_path)
        os.replace(tmp_path, save_path)

    def load_full(self, load_path):
        """Restore a checkpoint written by save_full. Returns the runner state.

        Unlike load_pretrained this keeps the learner in training mode: it is a
        resume, not an evaluation of frozen weights.
        """
        print(f"Resuming from {load_path}")
        ckpt = torch.load(load_path, map_location=self.config.DEVICE, weights_only=False)
        state = ckpt['learner']

        for name, sd in state['modules'].items():
            getattr(self, name).load_state_dict(sd)

        for name, sd in state['optimizers'].items():
            getattr(self, name).load_state_dict(sd)

        for name, value in state['counters'].items():
            setattr(self, name, value)

        rms = state['state_rms']
        self.state_rms.mean, self.state_rms.var, self.state_rms.count = rms['mean'], rms['var'], rms['count']
        self.num_batch_train.load_state_dict(state['num_batch_train'])

        if 'denoiser_lr_sched' in state and self.denoiser_lr_sched is not None:
            self.denoiser_lr_sched.load_state_dict(state['denoiser_lr_sched'])

        torch.set_rng_state(state['torch_rng'].cpu().to(torch.uint8))
        np.random.set_state(state['numpy_rng'])
        if 'cuda_rng' in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state['cuda_rng'])

        if ckpt.get('has_buffer'):
            self.replay_buffer.load_state_dict(state['replay_buffer'])
            self.mamba_replay_buffer.load_state_dict(state['mamba_replay_buffer'])
            print(f"Replay buffer restored: {self.replay_buffer.num_steps} steps, {len(self.replay_buffer)} episodes.")
        else:
            print("Checkpoint holds no replay buffer; it will refill before training resumes.")

        # The world models are kept in eval mode outside their own training loops,
        # matching how __init__ leaves them.
        self.state_decoder.eval()
        self.denoiser.eval()
        self.rew_end_model.eval()

        return ckpt.get('runner')

    def normalize_state(self, state):
        b, t, d = state.shape
        device = state.device
        dtype = state.dtype
        state_mean, state_var = self.state_rms.mean, self.state_rms.var

        normed_state = (state - torch.as_tensor(state_mean, device=device, dtype=dtype).expand(1, 1, -1)) / torch.sqrt(
            torch.as_tensor(state_var + 1e-8, device=device, dtype=dtype).expand(1, 1, -1)
        )
        return normed_state

    def step(self, rollout):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])

        self.add_experience_to_dataset(rollout)
        self.mamba_replay_buffer.append(rollout['observation'], rollout['shared_obs'], rollout['next_shared_obs'],
                                        rollout['action'], rollout['reward'], rollout['done'],
                                        rollout['fake'], rollout['last'], rollout.get('avail_action'))
        
        # self.global_state_normalizer.update(rollout['shared_obs'].mean(1))    # 这个是错的，MAPPO的value norm的init var是有问题的
        # print(self.global_state_normalizer.running_mean_var_cpu())

        self.state_rms.update(rollout['shared_obs'].mean(1))
        
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if self.replay_buffer.num_steps < self.config.MIN_BUFFER_SIZE:
            return
        
        if self.evaluate:
            self.eval_ac_in_wm()
            sys.exit()
            return

        self.accum_samples = 0
        sys.stdout.flush()

        self.train_count += 1
        total_to_log = []

        # train state decoder
        to_log = []
        pbar = tqdm(range(self.config.WM_EPOCHS if self.cur_wandb_epoch > 0 else 200),
                    desc=f"Training State Decoder", file=sys.stdout, disable=not self.tqdm_vis)
        for _ in pbar:
            samples = self.mamba_replay_buffer.sample_batch(bs=self.config.state_decoder_batch_size, sl=1 if self.config.state_decoder_type == StateDecoderType.OPTION1 else 2, mode="tokenizer")
            samples = self._to_device(samples)

            # normalize state
            samples['shared_obs'] = self.normalize_state(samples['shared_obs'].mean(2))

            if self.config.vq_type == 'fsq':
                metrics = self.train_fsq_tokenizer(
                    samples['shared_obs'],
                    rearrange(samples['observation'], 'b t n d -> b t (n d)'),
                )

                pbar.set_description(
                    f"Training state_decoder:"
                    + f"rec loss: {metrics[self.config.vq_type + '/rec_loss']:.4f}, "
                    + f"active %: {metrics[self.config.vq_type + '/active']:.3f}"
                )
            
            else:
                metrics = self.train_vq_tokenizer(
                    samples['shared_obs'],
                    rearrange(samples['observation'], 'b t n d -> b t (n d)'),
                )

                pbar.set_description(
                    f"Training state_decoder:"
                    + f"rec loss: {metrics[self.config.vq_type + '/rec_loss']:.4f}, "
                    + f"cmt loss: {metrics[self.config.vq_type + '/cmt_loss']:.4f}, "
                    + f"active %: {metrics[self.config.vq_type + '/active']:.3f}"
                )

            to_log.append(metrics)
        
        to_log = [{f"state_decoder/train/{k}": v for k, v in d.items()} for d in to_log]
        total_to_log += to_log
        
        # train denoiser
        self.denoiser.train()
        self.denoiser_opt.zero_grad()

        pbar = tqdm(range(self.config.WM_EPOCHS if self.cur_wandb_epoch > 0 else self.config.denoiser_steps_first_epoch),
                    desc=f"Training Denoiser", file=sys.stdout, disable=not self.tqdm_vis)
        to_log = []
        for i in pbar:
            ## FINISH: already added temperature sampling batch
            samples = self.replay_buffer.sample_batch(batch_num_samples=self.config.denoiser_batch_size,
                                                      sequence_length=self.config.denoiser_cfg.inner_model.num_steps_conditioning + 1 + 2,
                                                      sample_from_start=False,
                                                      valid_sample=False)
            # samples = self.replay_buffer.sample_batch_new(batch_num_samples=128)
            
            samples = samples.to(self.config.DEVICE)

            # normalize the global state
            # samples.shared_obs = torch.cat([samples.shared_obs, samples.next_shared_obs[:, -1:]], dim=1)
            # samples.mask_padding = torch.cat([samples.mask_padding, torch.ones_like(samples.mask_padding[:, -1:])], dim=1)

            samples.shared_obs = self.normalize_state(samples.shared_obs.mean(2))
            samples.shared_obs[samples.mask_padding.logical_not()] = torch.zeros_like(samples.shared_obs[samples.mask_padding.logical_not()], device=self.config.DEVICE)

            # ensure the state following the sigma_data of our hyperparams
            samples.shared_obs = self.denoiser.encode(samples.shared_obs)

            loss, metrics = self.denoiser(samples)
            loss.backward()

            num_batch = self.num_batch_train.get('denoiser')
            metrics[f"num_batch_train_denoiser"] = num_batch
            self.num_batch_train.set('denoiser', num_batch + 1)

            if (i + 1) % self.config.grad_acc_steps == 0:
                if self.config.denoiser_max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), self.config.denoiser_max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                self.denoiser_opt.step()
                self.denoiser_opt.zero_grad()

                if getattr(self, 'denoiser_lr_sched', None):
                    metrics["lr"] = self.denoiser_lr_sched.get_last_lr()[0]
                    self.denoiser_lr_sched.step()

            to_log.append(metrics)

            pbar.set_description(
                f"Training Denoiser: "
                + f"loss_denoising: {metrics['loss_denoising']:.4f}, "
                + f"norm: {metrics['grad_norm_before_clip']:.4f}"
            )
        
        to_log = [{f"denoiser/train/{k}": v for k, v in d.items()} for d in to_log]
        total_to_log += to_log

        self.denoiser.eval()

        ## train rew_end_model
        self.rew_end_model.train()
        self.rew_end_model_opt.zero_grad()

        pbar = tqdm(range(self.config.remodel_steps if self.cur_wandb_epoch > 0 else self.config.remodel_steps_first_epoch),
                    desc=f"Training Function", file=sys.stdout, disable=not self.tqdm_vis)
        to_log = []
        for i in pbar:
            samples = self.mamba_replay_buffer.sample_batch(bs=self.config.rew_end_batch_size, sl=self.config.horizon if self.rew_end_model.__class__ == TransRewEndModel else 20, mode='rew_end_model')
            samples = self._to_device(samples)

            samples['shared_obs'] = self.normalize_state(samples['shared_obs'].mean(2))
            samples['next_shared_obs'] = self.normalize_state(samples['next_shared_obs'].mean(2))

            if self.rew_end_model.__class__ == TransRewEndModel:
                attn_mask = self.mamba_replay_buffer.generate_attn_mask(samples["done"], self.rew_end_model.config.tokens_per_block).to(self.config.DEVICE)
                metrics = self.train_model(samples, attn_mask)

            else:
                loss, metrics = self.rew_end_model(samples, gamma = self.config.GAMMA, contdisc = self.config.contdisc)
                loss.backward()

            num_batch = self.num_batch_train.get('rew_end_model')
            metrics[f"num_batch_train_rew_end_model"] = num_batch
            self.num_batch_train.set('rew_end_model', num_batch + 1)

            if (i + 1) % self.config.grad_acc_steps == 0:
                if self.config.remodel_max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.rew_end_model.parameters(), self.config.remodel_max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                self.rew_end_model_opt.step()
                self.rew_end_model_opt.zero_grad()

                if getattr(self, 'rew_end_model_lr_sched', None):
                    metrics["lr"] = self.rew_end_model_lr_sched.get_last_lr()[0]
                    self.rew_end_model_lr_sched.step()

            to_log.append(metrics)

            pbar.set_description(
                f"Training Function: "
                + f"loss_rew: {metrics['loss_rew']:.4f}, "
                + f"loss_con: {metrics['loss_end']:.4f}, "
                + f"loss_av_action: {metrics['loss_av_action']:.4f}, "
                + f"total: {metrics['loss_total']:.4f}"
            )
        
        to_log = [{f"rew_end_model/train/{k}": v for k, v in d.items()} for d in to_log]
        total_to_log += to_log

        self.rew_end_model.eval()

        ## train actor_critic
        if self.train_count == 10:
            print('Start training actor & critic...')

        if self.train_count > 9:
            to_log = []

            pbar = tqdm(range(self.config.EPOCHS if self.cur_wandb_epoch > 0 else self.config.ac_steps_first_epoch),
                        desc=f"Training actor-critic", file=sys.stdout, disable=not self.tqdm_vis)

            for i in pbar:
                metrics = self.train_agent()

                num_batch = self.num_batch_train.get('actor_critic')
                metrics[f"num_batch_train_actor_critic"] = num_batch
                self.num_batch_train.set('actor_critic', num_batch + 1)

                to_log.append(metrics)

                pbar.set_description(
                    f"Training actor-critic: "
                    + f"Max reward: {metrics['Max reward']:.4f}, "
                    + f"Min reward: {metrics['Min reward']:.4f}, "
                    + f"Value: {metrics['Value']:.4f}, "
                    + f"Disc: {metrics['Discount']:.4f}, "
                    + f"End: {metrics['End']:.4f}"
                )

            to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in to_log]
            total_to_log += to_log

            self.actor.eval()
            self.critic.eval()

        wandb_log(total_to_log, self.cur_wandb_epoch)

        for d in total_to_log:
            for k, v in d.items():
                LOGGER.log_scalar(k, v, self.cur_wandb_epoch)

        self.cur_wandb_epoch += 1

    #### train state decoder
    def _reconstruction_loss(self, out, obs):
        """Reconstruction loss for the state decoder.

        L1 everywhere, except on the channels ``config.obs_binary_indices`` names
        as 0/1 flags: those get a squared error. L1's optimum is the conditional
        median, which for a flag that is rarely 1 is exactly 0 -- the decoder then
        predicts "never" for every flag and the prediction says nothing about how
        likely it is. A squared error puts the optimum at the conditional mean,
        which for a 0/1 variable *is* the probability, and so is directly
        comparable against the 0.5 cut a reader applies to it.

        Returns the total loss and its two parts, so a run can be watched for the
        flag block improving while the continuous block holds.
        """

        binary_indices = getattr(self.config, 'obs_binary_indices', None)
        if not binary_indices:
            rec_loss = (out - obs).abs().mean()
            return rec_loss, rec_loss, None

        if self._binary_index_tensor is None or self._binary_index_tensor.device != out.device:
            self._binary_index_tensor = torch.as_tensor(
                binary_indices, dtype=torch.long, device=out.device
            )
            mask = torch.zeros(out.shape[-1], dtype=torch.bool, device=out.device)
            mask[self._binary_index_tensor] = True
            self._continuous_index_tensor = torch.nonzero(~mask, as_tuple=False).squeeze(-1)

        binary_loss = (
            (out[..., self._binary_index_tensor] - obs[..., self._binary_index_tensor])
            .square()
            .mean()
        )
        continuous_loss = (
            (out[..., self._continuous_index_tensor] - obs[..., self._continuous_index_tensor])
            .abs()
            .mean()
        )
        weight = float(getattr(self.config, 'obs_binary_loss_weight', 1.0))
        return continuous_loss + weight * binary_loss, continuous_loss, binary_loss

    def train_vq_tokenizer(self, state, obs):
        assert type(self.state_decoder) == SimpleVQAutoEncoder

        b, t = state.shape[:2]

        self.state_decoder.train()

        ### prepare state decoder input
        # if self.config.state_decoder_type == StateDecoderType.OPTION1:
        #     agent_id = torch.eye(self.n_agents, dtype=torch.float32, device=state.device).detach()
        #     agent_id = repeat(agent_id, 'n d -> b t n d', b = b, t = t).detach()
        #     state = torch.cat(
        #         [repeat(state, ' b t d -> b t n d', n=self.n_agents), agent_id], dim = -1
        #     )
        #     obs_target = obs.clone()

        # else:
        #     state = torch.cat(
        #         [repeat(state[:, 1:], ' b t d -> b t n d', n=self.n_agents), obs[:, :-1]], dim = -1
        #     )
        #     obs_target = obs[:, 1:].clone()


        out, indices, cmt_loss = self.state_decoder(state, True, True)      # tokenzier 内部的预处理
        rec_loss, continuous_loss, binary_loss = self._reconstruction_loss(out, obs)
        loss = rec_loss + self.config.alpha * cmt_loss

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.state_decoder_optimizer, self.state_decoder, loss, self.config.max_grad_norm)
        self.state_decoder.eval()

        loss_dict = {
            self.config.vq_type + "/cmt_loss": cmt_loss.item(),
            self.config.vq_type + "/rec_loss": rec_loss.item(),
            self.config.vq_type + "/active": active_rate,
        }
        if binary_loss is not None:
            loss_dict[self.config.vq_type + "/rec_loss_continuous"] = continuous_loss.item()
            loss_dict[self.config.vq_type + "/rec_loss_binary"] = binary_loss.item()

        return loss_dict

    def train_fsq_tokenizer(self, state, obs):
        assert type(self.state_decoder) == SimpleFSQAutoEncoder

        b, t = state.shape[:2]

        self.state_decoder.train()

        ### prepare state decoder input
        # if self.config.state_decoder_type == StateDecoderType.OPTION1:
        #     agent_id = torch.eye(self.n_agents, dtype=torch.float32, device=state.device).detach()
        #     agent_id = repeat(agent_id, 'n d -> b t n d', b = b, t = t).detach()
        #     state = torch.cat(
        #         [repeat(state, ' b t d -> b t n d', n=self.n_agents), agent_id], dim = -1
        #     )
        #     obs_target = obs.clone()

        # else:
        #     state = torch.cat(
        #         [repeat(state[:, 1:], ' b t d -> b t n d', n=self.n_agents), obs[:, :-1]], dim = -1
        #     )
        #     obs_target = obs[:, 1:].clone()

        out, indices = self.state_decoder(state, True, True)
        loss, continuous_loss, binary_loss = self._reconstruction_loss(out, obs)

        active_rate = indices.detach().unique().numel() / self.obs_vocab_size * 100

        self.apply_optimizer(self.state_decoder_optimizer, self.state_decoder, loss, self.config.max_grad_norm)
        self.state_decoder.eval()

        loss_dict = {
            self.config.vq_type + "/rec_loss": loss.item(),
            self.config.vq_type + "/active": active_rate,
        }
        if binary_loss is not None:
            loss_dict[self.config.vq_type + "/rec_loss_continuous"] = continuous_loss.item()
            loss_dict[self.config.vq_type + "/rec_loss_binary"] = binary_loss.item()

        return loss_dict

    ### train rew end model
    def train_model(self, samples, attn_mask = None):
        self.rew_end_model.train()
        
        loss, loss_dict = self.rew_end_model.compute_loss(samples, attn_mask)
        self.apply_optimizer(self.rew_end_model_opt, self.rew_end_model, loss, self.config.remodel_max_grad_norm) # or GRAD_CLIP
        self.rew_end_model.eval()
        return loss_dict
        
    def visualize_attention_map(self, epoch, save_mode='interval'):
        if save_mode == 'interval':
            save_path = Path(self.config.RUN_DIR) / "visualization" / "attn" / f"epoch_{epoch}"
        elif save_mode == 'final':
            save_path = Path(self.config.RUN_DIR) / "visualization" / "attn" / "final"
        
        self.model.eval()
        self.tokenizer.eval()
        sample = self.replay_buffer.sample_batch(batch_num_samples=1,
                                                    sequence_length=self.config.HORIZON,
                                                    sample_from_start=True,
                                                    valid_sample=True)
        sample = self._to_device(sample)
        self.model.visualize_attn(sample, self.tokenizer, save_path)
    
    def train_agent(self, ):
        log_metrics = {}

        self.state_decoder.eval()
        self.denoiser.eval()
        self.rew_end_model.eval()

        # obs: (batch_size, horizon, num_agents, obs_dim)
        # shared_obs: (batch_size, horizon, state_dim)
        # act: (batch_size, horizon, num_agents, act_dim)
        obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, av_actions \
            = rollout_diffusion_world_models(self.replay_buffer, self.state_rms, self.state_decoder, self.denoiser, self.rew_end_model,
                                             self.actor, self.critic, self.config, env_type=self.env_type)

        # obs, shared_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, av_actions, _ = self.wm_env_loop.send(self.config.horizon)

        if self.use_valuenorm:
            val_shape = val_bootstrap.shape
            val_bootstrap = self.value_normalizer.denormalize(rearrange(val_bootstrap, 'b l 1 -> (b l) 1'))
            val_bootstrap = rearrange(val_bootstrap, '(b l) 1 -> b l 1', b = val_shape[0], l = val_shape[1])

            val_unnormalized = self.value_normalizer.denormalize(rearrange(val, 'b l 1 -> (b l) 1'))
            val_unnormalized = rearrange(val_unnormalized, '(b l) 1 -> b l 1', b = val_shape[0], l = val_shape[1])
        
        if self.config.compute_end_in_TD:
            ### 这是最初 raw state diffusion 计算return的方式
            # cprint('using end in return', 'light_magenta')
            lambda_returns = compute_lambda_returns(rew.squeeze(-1), end, trunc, val_bootstrap.squeeze(-1), self.config.GAMMA, self.config.DISCOUNT_LAMBDA).unsqueeze(-1)
        else:
            # cprint('using pcont in return', 'light_blue')
            lambda_returns = compute_lambda_returns_with_pcont_wo_end(
                rew.view(*rew.shape, 1),
                pcont.view(*pcont.shape, 1),
                val_bootstrap,
                gamma = 1.0 if self.config.contdisc else self.config.GAMMA, lmbda = self.config.DISCOUNT_LAMBDA,
            )
        
        if self.use_valuenorm:
            adv = (lambda_returns - val_unnormalized).detach()
        else:
            adv = (lambda_returns - val).detach()

        # normalize adv
        adv = advantage(adv)
        adv = repeat(adv, 'b h d -> b h n d', n = self.config.NUM_AGENTS)

        # reshape
        obs            = rearrange(obs,            'b h n d -> (b h) n d')
        shared_obs     = rearrange(shared_obs,     'b h d -> (b h) d')
        act            = rearrange(act,            'b h n d -> (b h) n d')
        logits_act     = rearrange(logits_act,     'b h n d -> (b h) n d')
        lambda_returns = rearrange(lambda_returns, 'b h d -> (b h) d')
        val            = rearrange(val,            'b h d -> (b h) d')
        adv            = rearrange(adv,            'b h n d -> (b h) n d')
        av_actions     = rearrange(av_actions,     'b h n d -> (b h) n d') if av_actions is not None else None

        logits_act = logits_act.detach()

        log_metrics['Returns'] = lambda_returns.detach().mean()
        tmp = {'Max reward': rew.detach().max(), 'Min reward': rew.detach().min(),
               'Reward': rew.detach().mean(), 'Discount': pcont.detach().to(torch.float32).mean(), 'End': end.detach().to(torch.float32).mean(),
               'Value': val.detach().mean()}
        
        # print(f"Value/Max reward: {rew.max():.4f}, Value/Min reward: {rew.min():.4f}, Value/Value: {val.mean():.4f}, "
        #       + f"Value/Discount: {pcont.detach().to(torch.float32).mean()}")

        log_metrics.update(tmp)

        self.cur_update += 1
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(obs.shape[0])

            step = 2000
            if self.env_type in [Env.MAMUJOCO]:
                # if environment is MAMujoco, we set the step according to the num_mini_batch
                step = int(len(inds) / self.config.num_mini_batch)

            for i in range(0, len(inds), step):
                idx = inds[i:i + step]

                if not self.config.CONTINUOUS_ACTION:
                    loss = actor_loss(obs[idx], act[idx], av_actions[idx] if av_actions is not None else None,
                                      logits_act[idx], adv[idx], self.actor, self.entropy, clip_param=self.config.clip_param)
                else:
                    loss = continuous_actor_loss(obs[idx], act[idx], None,
                                                 logits_act[idx], adv[idx], self.actor, self.entropy, self.config.clip_param)
                
                actor_grad_norm = self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING

                # using value normalization
                if self.use_valuenorm:
                    # get new values
                    values = self.critic(shared_obs[idx])
                    
                    value_pred_clipped = val[idx] + (values - val[idx]).clamp(
                        - self.config.clip_param, self.config.clip_param
                    )

                    self.value_normalizer.update(lambda_returns[idx])
                    normalized_returns = self.value_normalizer.normalize(lambda_returns[idx])
                    error_clipped  = normalized_returns.clone() - value_pred_clipped
                    error_original = normalized_returns.clone() - values

                    if self.use_huber_loss:
                        value_loss_clipped = huber_loss(error_clipped, self.config.huber_delta)
                        value_loss_original = huber_loss(error_original, self.config.huber_delta)
                    else:
                        value_loss_clipped = mse_loss(error_clipped)
                        value_loss_original = mse_loss(error_original)

                    if self.use_clipped_value_loss:
                        val_loss = torch.max(value_loss_original, value_loss_clipped)
                    else:
                        val_loss = value_loss_original

                    val_loss = val_loss.mean()

                else:
                    val_loss = value_loss(self.critic, shared_obs[idx], lambda_returns[idx])


                tmp = {'Val_loss': val_loss.detach(), 'Actor_loss': loss.detach()}
                log_metrics.update(tmp)

                critic_grad_norm = self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                
                tmp = {'Agent/actor_grad_norm': actor_grad_norm.detach(), 'Agent/critic_grad_norm': critic_grad_norm.detach()}
                log_metrics.update(tmp)
        
        # hard update critic
        if self.cur_update % self.config.TARGET_UPDATE == 0:
            self.old_critic = deepcopy(self.critic)
            self.cur_update = 0

        return log_metrics

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        return grad_norm

    ## add data to dataset
    def add_experience_to_dataset(self, data, mode='train'):
        if self.env_type == Env.STARCRAFT:
            episode = SC2Episode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                shared_obs=torch.FloatTensor(data['shared_obs'].copy()),                # (Length, n_agents, state_dim)
                next_shared_obs=torch.FloatTensor(data['next_shared_obs'].copy()),      # (Length, n_agents, state_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                av_action=torch.FloatTensor(data['avail_action'].copy()) if 'avail_action' in data else None,   # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.SMACv2:
            episode = SC2Episode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                shared_obs=torch.FloatTensor(data['shared_obs'].copy()),                # (Length, n_agents, state_dim)
                next_shared_obs=torch.FloatTensor(data['next_shared_obs'].copy()),      # (Length, n_agents, state_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                av_action=torch.FloatTensor(data['avail_action'].copy()) if 'avail_action' in data else None,   # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.PETTINGZOO:
            # episode = MpeEpisode(
            #     observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
            #     action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
            #     reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
            #     done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
            #     filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            # )
            episode = MamujocoEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                shared_obs=torch.FloatTensor(data['shared_obs'].copy()),                # (Length, n_agents, state_dim)
                next_shared_obs=torch.FloatTensor(data['next_shared_obs'].copy()),
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.GRF:
            raise NotImplementedError
            episode = GRFEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.MAMUJOCO:
            episode = MamujocoEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                shared_obs=torch.FloatTensor(data['shared_obs'].copy()),                # (Length, n_agents, state_dim)
                next_shared_obs=torch.FloatTensor(data['next_shared_obs'].copy()),
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        elif self.env_type == Env.MATE:
            # added by the DIMA x MATE research layer; see research/UPSTREAM_PATCHES.md.
            # MATE has no action mask, so MamujocoEpisode (no av_action field) is the
            # matching episode type -- same seven fields, discrete one-hot actions.
            episode = MamujocoEpisode(
                observation=torch.FloatTensor(data['observation'].copy()),              # (Length, n_agents, obs_dim)
                shared_obs=torch.FloatTensor(data['shared_obs'].copy()),                # (Length, n_agents, state_dim)
                next_shared_obs=torch.FloatTensor(data['next_shared_obs'].copy()),      # (Length, n_agents, state_dim)
                action=torch.FloatTensor(data['action'].copy()),                        # (Length, n_agents, act_dim)
                reward=torch.FloatTensor(data['reward'].copy()),                        # (Length, n_agents, 1)
                done=torch.FloatTensor(data['done'].copy()),                            # (Length, n_agents, 1)
                filled=torch.ones(data['done'].shape[0], dtype=torch.bool)
            )

        else:
            raise NotImplementedError

        if mode == 'train':
            self.replay_buffer.add_episode(episode)
        elif mode == 'val':
            return episode
        else:
            raise NotImplementedError

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k: batch[k].to(self.config.DEVICE) if batch[k] is not None else None for k in batch}

    @torch.no_grad()
    def log_compounding_errors(self, trajs):
        episodes = []
        if trajs is not None:
            for traj in trajs:
                episodes.append(
                    self.add_experience_to_dataset(traj, mode='val')
                )
        
        else:
            episodes = self.replay_buffer.episodes

        smad_obs_l1_errors = []
        smad_obs_l2_errors = []
        smad_state_l1_errors = []
        smad_state_l2_errors = []
        vae_rec_obs_l1_errors = []

        horizon = self.config.horizon
        num_steps_conditioning = self.config.denoiser_cfg.inner_model.num_steps_conditioning
        
        sampler = DiffusionSampler(self.denoiser, self.config.worldmodel_env_cfg.diffusion_sampler)
        
        import random
        for _ in range(20):
            sampled_traj = random.choice(episodes)
            length = len(sampled_traj)
            end = np.random.randint(horizon + 1, length + 1)
            start = end - (horizon + 1) - (num_steps_conditioning - 1)

            segment = sampled_traj.segment(start, end, should_pad = True)

            rec_obses = torch.zeros_like(segment.observation, device = self.config.DEVICE, dtype=segment.observation.dtype)
            rec_obses[segment.filled] = rearrange(
                self.state_decoder.encode_decode(segment.shared_obs[segment.filled].mean(1).to(self.config.DEVICE), True, True), 'b (n d) -> b n d', n = self.n_agents, d=segment.observation.size(-1),
            )
            
            if self.env_type in [Env.STARCRAFT, Env.SMACv2]:
                rec_obses = rec_obses.clamp(-1., 1.)
            
            gt_state     = segment.shared_obs.mean(1).unsqueeze(0).to(self.config.DEVICE)
            state_buffer = segment.shared_obs[:num_steps_conditioning].mean(1).unsqueeze(0).to(self.config.DEVICE)
            
            gt_obs = segment.observation.unsqueeze(0).to(self.config.DEVICE)
            gt_act = segment.action.unsqueeze(0).to(self.config.DEVICE)

            pred_states = []
            pred_obses  = []
            for i in range(horizon):
                act_buffer = gt_act[:, i : i + num_steps_conditioning]
                pred_state, denoised_traj = sampler.sample(state_buffer, act_buffer)

                pred_obs = rearrange(
                    self.state_decoder.encode_decode(pred_state.squeeze(1)), 'b (n d) -> b n d', n = self.n_agents, d=gt_obs.size(-1),
                )
                    
                if self.env_type in [Env.STARCRAFT, Env.SMACv2]:
                    pred_obs = pred_obs.clamp(-1., 1.)

                pred_state = pred_state.squeeze(1)

                pred_states.append(pred_state)
                pred_obses.append(pred_obs)

                state_buffer = state_buffer.roll(-1, dims=1)
                state_buffer[:, -1] = pred_state

            pred_obses  = torch.stack(pred_obses, dim=1)
            pred_states = torch.stack(pred_states, dim=1)

            # compute obs compounding errors
            obs_l1_errors = (pred_obses - gt_obs[:, num_steps_conditioning : ]).abs().mean(-1) #.mean()
            obs_l2_errors = (pred_obses - gt_obs[:, num_steps_conditioning : ]).pow(2).mean(-1) #.mean()

            rec_obs_l1_errors = (rec_obses.unsqueeze(0)[:, num_steps_conditioning : ] - gt_obs[:, num_steps_conditioning : ]).abs().mean(-1) #.mean()

            # compute state compounding errors
            state_l1_errors = (pred_states - gt_state[:, num_steps_conditioning : ]).abs().mean(-1) #.mean()
            state_l2_errors = (pred_states - gt_state[:, num_steps_conditioning : ]).pow(2).mean(-1) #.mean()

            smad_obs_l1_errors.append(obs_l1_errors.cpu().numpy())
            smad_obs_l2_errors.append(obs_l2_errors.cpu().numpy())

            smad_state_l1_errors.append(state_l1_errors.cpu().numpy())
            smad_state_l2_errors.append(state_l2_errors.cpu().numpy())

            vae_rec_obs_l1_errors.append(rec_obs_l1_errors.cpu().numpy())

            mujoco_visualization(self.config.env_name, segment.observation[num_steps_conditioning:].cpu().numpy(), self.n_agents, prefix_name = f'ground_truth_b{_}')
            mujoco_visualization(self.config.env_name, pred_obses[0].cpu().numpy(), self.n_agents, prefix_name = f'imagination_b{_}')
                
        smad_obs_l1_errors = np.concatenate(smad_obs_l1_errors, axis=0)
        smad_obs_l2_errors = np.concatenate(smad_obs_l2_errors, axis=0)

        smad_state_l1_errors = np.concatenate(smad_state_l1_errors, axis=0)
        smad_state_l2_errors = np.concatenate(smad_state_l2_errors, axis=0)
        vae_rec_obs_l1_errors = np.concatenate(vae_rec_obs_l1_errors, axis=0)

        state_l1_errors_cum = np.cumsum(smad_state_l1_errors, axis=1).mean(0)
        obs_l1_errors_cum = np.cumsum(smad_obs_l1_errors, axis=1).mean(0)
        rec_obs_l1_errors_cum = np.cumsum(vae_rec_obs_l1_errors, axis=1).mean(0)

        # import ipdb; ipdb.set_trace()

        print(f'At horizon {horizon:2d},')
        print(f'  state cum L1 errors: {state_l1_errors_cum[-1]:6.4f}')
        print(f'  obs cum L1 errors:     [{" ".join(f"{x:6.4f}" for x in obs_l1_errors_cum[-1])}]')
        print(f'  rec obs cum L1 errors: [{" ".join(f"{x:6.4f}" for x in rec_obs_l1_errors_cum[-1])}]')
        

    @torch.no_grad()
    def eval_ac_in_wm(self,):
        log_metrics = {}

        self.state_decoder.eval()
        self.denoiser.eval()
        self.rew_end_model.eval()
        self.actor.eval()
        self.critic.eval()

        self.log_compounding_errors(None)

