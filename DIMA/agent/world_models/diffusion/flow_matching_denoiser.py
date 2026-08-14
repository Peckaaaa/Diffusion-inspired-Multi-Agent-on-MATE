from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from .inner_model import InnerModel, InnerModelConfig, StateInnerModelConfig
from ..perceiver import PerceiverConfig
from utils import LossAndLogs


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


@dataclass
class FlowMatchingDenoiserConfig:
    inner_model: StateInnerModelConfig
    perceiver: PerceiverConfig


class FlowMatchingDenoiser(nn.Module):
    def __init__(self, cfg: FlowMatchingDenoiserConfig,
                 num_agents: int = None,
                 is_continuous_act: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(
            cfg.inner_model,
            cfg.perceiver,
            num_agents=num_agents,
            is_continuous_act=is_continuous_act
        )
        self.is_continuous_act = is_continuous_act
        self.num_agents = num_agents

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    @torch.no_grad()
    def encode(self, latent: Tensor) -> Tensor:
        return latent

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        return latent

    def compute_velocity(self, z_tau: Tensor, tau: Tensor, obs: Tensor, act: Tensor, act_mask: Tensor) -> Tensor:
        """
        Evaluates the velocity network v_theta(z_tau, tau, obs, act, act_mask).
        tau is in [0, 1]. Map tau to time embedding (c_noise = tau).
        """
        assert act.size(2) == self.num_agents
        c_noise = tau
        return self.inner_model(z_tau, c_noise, obs, act, act_mask)

    def forward(self, batch) -> Tuple[Tensor, Dict[str, Any]]:
        assert batch.act.size(2) == self.num_agents

        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.shared_obs.size(1) - n

        all_obs = batch.shared_obs.clone()  # Global normalized states (b, seq_len+n, state_dim)
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]             # (b, seq_len, state_dim)
            next_obs = all_obs[:, n + i]            # z1: clean normalized state (b, state_dim)
            act = batch.act[:, i : n + i]           # (b, seq_len, n, act_dim)
            mask = batch.mask_padding[:, n + i]     # (b, seq_len)

            if not self.is_continuous_act:
                act = act.argmax(-1)

            b, t, d = obs.shape
            z1 = next_obs.unsqueeze(1)               # (b, 1, d)

            # Sample tau ~ U[0, 1] for each item in batch
            tau = torch.rand(b, device=self.device)   # (b,)
            
            # Sample z0 ~ N(0, I)
            z0 = torch.randn_like(z1, device=self.device)  # (b, 1, d)

            # Interpolation path: z_tau = (1 - tau) * z0 + tau * z1
            tau_expanded = add_dims(tau, z1.ndim)          # (b, 1, 1)
            z_tau = (1.0 - tau_expanded) * z0 + tau_expanded * z1

            # Target velocity: u = z1 - z0
            target_velocity = z1 - z0

            # Sample random agent k for conditioning (identical to DIMA's denoiser.py)
            activate_action_indices = torch.randint(0, self.num_agents, (b,), device=self.device)
            act_mask = torch.ones(b, t - 1, self.num_agents, device=self.device, dtype=torch.long)
            act_mask = torch.cat(
                (act_mask, F.one_hot(activate_action_indices, num_classes=self.num_agents).unsqueeze(1).to(torch.long)),
                dim=1
            )

            # Compute predicted velocity field v_theta
            predicted_velocity = self.compute_velocity(z_tau, tau, obs, act, act_mask)

            # MSE Loss against target velocity (z1 - z0)
            step_loss = F.mse_loss(predicted_velocity[mask], target_velocity[mask])
            loss += step_loss

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach(), "loss_velocity": loss.detach()}
