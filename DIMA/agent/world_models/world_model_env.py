from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple, Union
from einops import repeat, rearrange
from termcolor import cprint

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader
import torch.distributions as td

from agent.coroutines import coroutine
from agent.world_models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from agent.world_models.diffusion.flow_matching_denoiser import FlowMatchingDenoiser
from agent.world_models.diffusion.flow_matching_sampler import FlowMatchingSampler, FlowMatchingSamplerConfig
from agent.world_models.rew_end_model import TransRewEndModel, RewEndModel
from agent.world_models.vq import SimpleVQAutoEncoder, SimpleFSQAutoEncoder, StateDecoderType

from dataset import MultiAgentEpisodesDataset
from environments import Env

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]

THRES = 1e-2

@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        running_mean_std,
        state_decoder: Union[SimpleFSQAutoEncoder, SimpleVQAutoEncoder],
        denoiser: Union[Denoiser, FlowMatchingDenoiser],
        rew_end_model: TransRewEndModel,
        dataset: MultiAgentEpisodesDataset,
        num_envs: int,
        cfg: WorldModelEnvConfig,
        env_type,
        state_decoder_type,
        device,
        return_denoising_trajectory: bool = False,
        mode: str = "non-ensemble",
        should_reset_with_dead: bool = False,
        **kwargs,
    ) -> None:
        self.running_mean_std = running_mean_std

        if isinstance(denoiser, FlowMatchingDenoiser):
            fm_sampler_cfg = FlowMatchingSamplerConfig(
                num_steps_sampling=cfg.diffusion_sampler.num_steps_denoising,
                agent_order=cfg.diffusion_sampler.agent_order
            )
            self.sampler = FlowMatchingSampler(denoiser, fm_sampler_cfg)
        else:
            self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)

        self.is_continuous_act = denoiser.is_continuous_act

        self.pred_av_action = rew_end_model.pred_av_action

        self.state_decoder = state_decoder.to(device).eval()
        self.rew_end_model = rew_end_model.to(device).eval()
        self.num_agents = denoiser.num_agents
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory

        self.tokens_per_block = self.rew_end_model.config.tokens_per_block

        self.num_envs = num_envs
        self.sl = denoiser.cfg.inner_model.num_steps_conditioning
        self.generator_init = self.make_generator_init(dataset, cfg.num_batches_to_preload)
        
        self.cfg = cfg
        self.mode = mode  # 'ensemble'
        self.env_type = env_type
        self.state_decoder_type = state_decoder_type
        self.should_reset_with_dead = should_reset_with_dead

        ### 以下是个可以调整的超参
        self.end_thres = 0.5

        ## debug
        self.config = kwargs.get('config', None)
        self.is_debug = kwargs.get('is_debug', False)
        # self.use_rnn_rew_end_model = hasattr(rew_end_model, 'lstm', False)

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device
    
    @torch.no_grad()
    def get_global_state(self):
        last_action = rearrange(self.act_buffer[:, -2], 'b n d -> b (n d)')
        global_state_per_agent = torch.cat([self.state_buffer[:, -1], last_action], dim=-1)
        global_state = torch.stack([global_state_per_agent for _ in range(self.num_agents)], dim = 1)
        return global_state_per_agent

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        state, obs, act, av_action = self.generator_init.send(self.num_envs)
        self.state_buffer = state                   # state_buffer: b, t, state_dim
        self.obs_buffer   = obs                     # obs_buffer  : b, t, n, obs_dim
        self.act_buffer   = act                     # act_buffer  : b, t, n, act_dim

        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        self.keys_values_rew_end = self.rew_end_model.transformer.generate_empty_keys_values(n = self.num_envs, max_tokens = self.rew_end_model.config.max_tokens)
        self.flipped_attn_mask = torch.tril(torch.ones(self.tokens_per_block, self.tokens_per_block, device=self.device))[None, :, :].repeat(self.num_envs, 1, 1)
        self.flipped_attn_mask = self.flipped_attn_mask.flip(dims=[-1])
        return self.obs_buffer[:, -1], self.get_global_state(), av_action[:, -1] if self.pred_av_action else None

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        state, obs, act, av_action = self.generator_init.send(dead.sum().item())

        self.state_buffer[dead] = state
        self.obs_buffer[dead]   = obs
        self.act_buffer[dead]   = act

        self.ep_len[dead]       = 0

        self.flipped_attn_mask[dead] = torch.zeros_like(self.flipped_attn_mask[dead])
        self.flipped_attn_mask[dead, :, :self.tokens_per_block] = torch.tril(torch.ones(self.tokens_per_block, self.tokens_per_block, device=self.device))[None, :, :].repeat(dead.sum(), 1, 1)
        self.flipped_attn_mask = self.flipped_attn_mask.flip(dims=[-1])

        return av_action[:, -1] if self.pred_av_action else None
    
    ## debug
    @torch.no_grad()
    def check_termination(self, next_state: torch.FloatTensor):
        assert self.config is not None and self.is_debug
        next_state = next_state.clone().squeeze(1)

        al_feat = next_state[..., : self.config.nf_al * self.config.n_allies]
        en_feat = next_state[..., self.config.nf_al * self.config.n_allies : ]
        
        al_feat = al_feat.view(-1, self.config.n_allies,  self.config.nf_al)
        en_feat = en_feat.view(-1, self.config.n_enemies, self.config.nf_en)
        
        al_health = al_feat[..., 0]
        en_health = en_feat[..., 0]
        
        # import ipdb; ipdb.set_trace()
        
        al_all_dead = (al_health < THRES).all(-1)
        en_all_dead = (en_health < THRES).all(-1)

        game_end = al_all_dead | en_all_dead
        return game_end.to(torch.int64)

    ## -----

    @torch.no_grad()
    def step(self, act: torch.LongTensor) -> StepOutput:
        self.act_buffer[:, -1] = act

        if self.mode == 'ensemble':
            next_state, denoising_trajectory = self.predict_next_state()
            next_state = next_state[0]
        else:
            next_state, denoising_trajectory = self.predict_next_state()
        
        next_obs = self.state_decoder.encode_decode(self.sampler.decode(next_state.squeeze(1)))
        # next_obs = self.predict_next_obs(self.sampler.decode(next_state.squeeze(1)))
        next_obs = rearrange(next_obs, 'b (n d) -> b n d', n = self.num_agents, d = self.obs_buffer.size(-1))

        if self.env_type in [Env.SMACv2, Env.STARCRAFT]:
            next_obs = next_obs.clamp(-1., 1.)

        rew, pcont, end, next_av_action = self.predict_rew_end(next_state)  # 这里无论单智能体还是多智能体都是(b,) shape的end

        next_state = next_state.squeeze(1)
        # next_state = repeat(next_state.squeeze(1), 'b d -> b n d', n=self.num_agents)

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()
        if trunc.ndim != end.ndim:
            trunc = repeat(trunc, 'b -> b n', n = self.num_agents)

        ## post process the attn mask
        self.flipped_attn_mask = torch.cat(
            [self.flipped_attn_mask, torch.ones(self.num_envs, self.tokens_per_block, self.tokens_per_block, dtype=self.flipped_attn_mask.dtype, device=self.device)], dim=-1
        )

        self.state_buffer = self.state_buffer.roll(-1, dims=1)
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)

        self.obs_buffer[:, -1]   = next_obs
        self.state_buffer[:, -1] = next_state

        if trunc.ndim == 2:
            dead = torch.logical_or(end, trunc).any(dim=-1) if self.should_reset_with_dead else torch.zeros_like(self.ep_len, device=self.ep_len.device).to(torch.bool)
            # dead = torch.zeros_like(self.ep_len, device=self.ep_len.device).to(torch.bool)
        else:
            dead = torch.logical_or(end, trunc) if self.should_reset_with_dead else torch.zeros_like(self.ep_len, device=self.ep_len.device).to(torch.bool)
            # dead = torch.zeros_like(self.ep_len, device=self.ep_len.device).to(torch.bool)

        info = {}
        if next_av_action is not None:
            info['av_action'] = next_av_action

        if self.return_denoising_trajectory:
            if self.mode == "ensemble":
                denoising_trajectory = [torch.stack(e, dim=1) for e in denoising_trajectory]
                info["denoising_trajectory"] = torch.concat(denoising_trajectory, dim=0)
            else:
                info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            # cprint('trigger dead in wm env', 'light_magenta')
            dead_next_state = next_state[dead]
            dead_last_action = rearrange(self.act_buffer[dead, -2], 'b n d -> b (n d)')
            dead_global_state_per_agent = torch.cat([dead_next_state, dead_last_action], dim=-1)
            dead_global_state = torch.stack([dead_global_state_per_agent for _ in range(self.num_agents)], dim = 1)

            info["final_state"] = dead_global_state_per_agent # dead_global_state
            av_action = self.reset_dead(dead)
            info["burnin_obs"] = self.obs_buffer[dead, :-1]
            if av_action is not None:
                info['av_action'][dead] = av_action

        return self.obs_buffer[:, -1], self.get_global_state(), rew, pcont, end, trunc, info

    @torch.no_grad()
    def predict_next_state(self) -> Tuple[Tensor, List[Tensor]]:
        if self.mode == 'ensemble':
            return self.sampler.ensemble_sample(self.state_buffer, self.act_buffer)
        else:
            return self.sampler.sample(self.state_buffer, self.act_buffer)

    # @torch.no_grad()
    # def predict_next_obs(self, next_state):
    #     if self.state_decoder_type == StateDecoderType.OPTION1:
    #         agent_id = torch.eye(self.num_agents, dtype=torch.float32, device=next_state.device).detach()
    #         agent_id = repeat(agent_id, 'n d -> b n d', b = next_state.size(0)).detach()
    #         decoder_input = torch.cat(
    #             [repeat(next_state, ' b d -> b n d', n=self.num_agents), agent_id], dim = -1
    #         )

    #     else:
    #         decoder_input = torch.cat(
    #             [repeat(next_state, ' b d -> b n d', n=self.num_agents), self.obs_buffer[:, -1]], dim = -1
    #         )

    #     return self.state_decoder.encode_decode(decoder_input)

    @torch.no_grad()
    def predict_rew_end(self, next_state: Tensor) -> Tuple[Tensor, Tensor]:
        ### transformer-based
        current_state = self.sampler.decode(self.state_buffer[:, -1:].clone())
        act_cond = self.rew_end_model.get_act_emb(self.act_buffer[:, -1]).unsqueeze(1)

        input_rew_end = torch.cat(
            [current_state, torch.empty_like(current_state, device=current_state.device, dtype=current_state.dtype)],
            dim=1,
        )

        outputs_rew_end = self.rew_end_model(input_rew_end, perattn_out=act_cond, past_keys_values = self.keys_values_rew_end, attention_mask = torch.flip(self.flipped_attn_mask, dims=[-1]))
        rew = outputs_rew_end.pred_rewards.float().squeeze(1).squeeze(-1)


        if self.rew_end_model.use_ce_for_end:
            end = Categorical(logits=outputs_rew_end.logits_ends).sample().squeeze(1)
            pcont = (1 - end).clone().to(torch.float32)   # act like a placeholder

        else:
            pred_cons = td.independent.Independent(td.Bernoulli(logits=outputs_rew_end.logits_ends), 1)
            pcont = pred_cons.mean.squeeze(1).squeeze(-1)
            end = (pcont.clone() < 0.7)

        if self.pred_av_action:
            av_action_dist = Categorical(logits=outputs_rew_end.pred_avail_action)
            next_av_action = av_action_dist.sample().squeeze(1)
        else:
            next_av_action = None

        return rew, pcont, end, next_av_action

    @coroutine
    def make_generator_init(
        self,
        dataset: MultiAgentEpisodesDataset,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield

        while True:
            # Preload on device and burnin rew/end model
            # obs_, act_, hx_, cx_ = [], [], [], []
            state_, obs_, act_ = [], [], []
            av_action_ = [] if self.pred_av_action else None

            for _ in range(num_batches_to_preload):
                batch = dataset.sample_batch(batch_num_samples=self.num_envs,
                                             sequence_length=self.sl,
                                             sample_from_start=False,
                                             valid_sample=False)
                state     = batch.shared_obs.to(self.device)
                obs       = batch.obs.to(self.device)
                act       = batch.act.to(self.device)
                av_action = batch.av_action.to(self.device) if self.pred_av_action else None
                mask      = batch.mask_padding.to(self.device)

                # with torch.no_grad():
                #     *_, (hx, cx) = self.rew_end_model.predict_rew_end(state[:, :-1].clone().mean(2), act[:, :-1],
                #                                                       state[:, 1:].clone().mean(2))  # Burn-in of rew/end model

                # import ipdb; ipdb.set_trace()
                with torch.no_grad():
                    b, t, d = state.mean(2).shape
                    state = (state.mean(2) - torch.as_tensor(self.running_mean_std.mean, dtype=state.dtype, device=self.device).expand(1, 1, -1)) / torch.sqrt(
                        torch.as_tensor(self.running_mean_std.var + 1e-8, dtype=state.dtype, device=self.device).expand(1, 1, -1)
                    )

                    state[mask.logical_not()] = torch.zeros_like(state[mask.logical_not()], device=self.device)

                    # reconstruct obs
                    # if self.state_decoder_type == StateDecoderType.OPTION1:
                    #     agent_id = torch.eye(self.num_agents, dtype=torch.float32, device=state.device).detach()
                    #     agent_id = repeat(agent_id, 'n d -> b t n d', b = state.size(0), t = state.size(1)).detach()
                    #     decoder_input = torch.cat(
                    #         [repeat(state.clone(), ' b t d -> b t n d', n=self.num_agents), agent_id], dim = -1
                    #     )

                    # else:
                    #     raise NotImplementedError
                    #     decoder_input = torch.cat(
                    #         [repeat(state.clone(), ' b d -> b n d', n=self.num_agents), self.obs_buffer[:, -1]], dim = -1
                    #     )

                    obs[mask] = rearrange(self.state_decoder.encode_decode(state[mask], True, True), 'b (n d) -> b n d', n = self.num_agents, d = obs.size(-1))

                    if self.env_type in [Env.SMACv2, Env.STARCRAFT]:
                        obs = obs.clamp(-1., 1.)

                    state = self.sampler.encode(state)

                state_.extend(list(state))
                obs_.extend(list(obs))
                act_.extend(list(act))
                if av_action is not None:
                    av_action_.extend(list(av_action))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(state_):
                # print(f"dataset size: {len(dataset)}")
                state     = torch.stack(state_[c : c + num_dead])
                obs       = torch.stack(obs_[c : c + num_dead])
                act       = torch.stack(act_[c : c + num_dead])
                av_action = torch.stack(av_action_[c : c + num_dead]) if self.pred_av_action else None
                c += num_dead
                # num_dead = yield state, obs, act, av_action, (hx, cx)
                num_dead = yield state, obs, act, av_action
