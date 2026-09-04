from dataclasses import dataclass
from typing import List, Optional, Tuple
from einops import rearrange

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet
from ..temporal_unet import UNet1D, Conv3x1
from ..perceiver import JointActionEmb, SequentialActionEmb, PerceiverConfig


@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None

@dataclass
class StateInnerModelConfig:
    state_dim: int
    num_steps_conditioning: int     # 该数值+1后应该能被4或者8整除
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    action_dim: Optional[int] = None
    dim: int = 128
    dim_mults: Tuple[int] =(1, 4, 8)
    #: How the conditioning actions reach the denoiser.
    #:
    #: ``'sequential'`` is DIMA's original sequential-causal encoder: the sampler
    #: exposes one agent's action per denoising step through ``act_mask``.
    #: ``'joint'`` feeds every agent's action at every step, so influence is not a
    #: function of which denoising step an agent was assigned -- see
    #: ``JointActionEmb`` for the measurement that motivates it.
    action_cond: str = 'sequential'
    #: Width of the per-agent action embedding used by ``'joint'``.
    agent_action_embed_dim: int = 64
    
class InnerModel(nn.Module):
    def __init__(self, cfg: StateInnerModelConfig,
                 perceiver_cfg: PerceiverConfig,
                 num_agents: int,
                 is_continuous_act: bool = False,) -> None:
        super().__init__()
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.is_continuous_act = is_continuous_act

        self.action_cond = getattr(cfg, 'action_cond', 'sequential')
        if self.action_cond == 'joint':
            self.act_emb = JointActionEmb(
                num_agents=num_agents,
                num_steps_conditioning=cfg.num_steps_conditioning,
                action_dim=cfg.action_dim,
                is_continuous_act=is_continuous_act,
                output_dim=perceiver_cfg.output_dim,
                agent_embed_dim=getattr(cfg, 'agent_action_embed_dim', 64),
            )
        elif self.action_cond == 'sequential':
            # No matter whether continuous action or discrete action, using Perceiver as action_emb
            self.act_emb = SequentialActionEmb(
                num_agents=num_agents,
                num_steps_conditioning=cfg.num_steps_conditioning,
                action_dim=cfg.action_dim,
                is_continuous_act=is_continuous_act,
                perceiver_cfg=perceiver_cfg,
            )
        else:
            raise ValueError(
                f"action_cond must be 'sequential' or 'joint', got {self.action_cond!r}."
            )

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        # self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])
        self.conv_in = Conv3x1(cfg.state_dim, cfg.channels[0])

        # for 2D
        # self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        # for 1D (state-based)
        self.unet = UNet1D(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x1(cfg.channels[0], cfg.state_dim)

        nn.init.zeros_(self.conv_out.weight)
    
    def compute_action_cond(self, act: Tensor, mask: Tensor, return_cross_attn: bool = False) -> Tensor:
        act_cond, cross_attn = self.act_emb(act, mask, return_cross_attn)

        return act_cond, cross_attn

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, act: Tensor, act_mask: Tensor) -> Tensor:
        act_cond, _ = self.compute_action_cond(act, act_mask, False)
        cond = self.cond_proj(self.noise_emb(c_noise) + act_cond)
        x = torch.cat((obs, noisy_next_obs), dim=1)

        x = rearrange(x, 'b h t -> b t h')
        x = self.conv_in(x)
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        
        x = rearrange(x, 'b t h -> b h t')
        # 注意这里x的horizon的最后一个才是我们想要的
        return x[:, -1].unsqueeze(1)
