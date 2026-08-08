from dataclasses import dataclass
from typing import List, Tuple
from einops import repeat

import torch
import torch.nn.functional as F
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    agent_order: str = ""

    loc: float = -0.4
    scale: float = 1.2


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser.eval()
        self.cfg = cfg
        
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def encode(self, state):
        return self.denoiser.encode(state)

    @torch.no_grad()
    def decode(self, state):
        return self.denoiser.decode(state)

    @torch.no_grad()
    def sample_agent_order(self, num_agents: int, order: str = "default"):
        assert self.cfg.num_steps_denoising % num_agents == 0

        if order == 'default':
            agent_order = torch.flip(torch.arange(num_agents), [0])

        elif order == 'reverse':
            agent_order = torch.arange(num_agents)
        
        elif order == 'random':
            agent_order = torch.randperm(num_agents)
        
        else:
            raise NotImplementedError('Plz specify the agent order for denoising.')
        
        # agent_order = repeat(agent_order, 'n -> n k', k=denoising_steps_per_agent).reshape(-1,)
        return agent_order

    @torch.no_grad()
    def sample(self, prev_state: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_state.device
        if prev_state.ndim == 4:   # (b, seq_length, num_agents, state_dim)
            prev_state = prev_state.mean(dim=2)

        if not self.denoiser.is_continuous_act and prev_act.ndim == 4:
            prev_act = prev_act.argmax(-1)

        b, t, d = prev_state.size()
        # prev_state = prev_state.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1) # TODO: 意义还没搞清楚
        x = torch.randn(b, 1, d, device=device) * self.sigmas[0]
        trajectory = [x]

        # implement sequential causal graph
        num_agents = self.denoiser.num_agents
        agent_order = self.sample_agent_order(num_agents, self.cfg.agent_order)

        if self.cfg.num_steps_denoising != len(agent_order):
            agent_order = torch.repeat_interleave(agent_order, repeats=2)
            
        # 每一个sigma就是一个denoising step
        for idx, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5

            act_mask = torch.ones(*prev_act.shape[:3], device=device, dtype=torch.long)
            act_mask[:, -1] = F.one_hot(agent_order[idx], num_classes=num_agents).expand(act_mask.size(0), -1)

            denoised = self.denoiser.denoise(x, sigma, prev_state, prev_act, act_mask)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, prev_state, prev_act)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)

        return x, trajectory
    
    @torch.no_grad()
    def ensemble_sample(self, prev_obs: Tensor, prev_act: Tensor):
        xs = []
        trajs = []
        ori_agent_order = self.cfg.agent_order
        
        x, trajectory = self.sample(prev_obs, prev_act)
        xs.append(x)
        trajs.append(trajectory)
        
        self.cfg.agent_order = "reverse"
        x, trajectory = self.sample(prev_obs, prev_act)
        xs.append(x)
        trajs.append(trajectory)

        self.cfg.agent_order = ori_agent_order
        return xs, trajs


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

