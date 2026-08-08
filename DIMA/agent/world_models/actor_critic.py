from collections import namedtuple
from dataclasses import dataclass
import math
from typing import List, Optional, Tuple, Union
import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

from .blocks import Conv3x3, SmallResBlock
from agent.coroutines.env_loop import make_env_loop
# from envs import TorchEnv, WorldModelEnv
from utils import init_lstm, LossAndLogs
from einops import repeat

ActorCriticOutput = namedtuple("ActorCriticOutput", "logits_act val hx_cx")


@dataclass
class ActorCriticLossConfig:
    backup_every: int
    gamma: float
    lambda_: float
    weight_value_loss: float
    weight_entropy_loss: float


@dataclass
class ActorCriticConfig:
    lstm_dim: int
    img_channels: int
    img_size: int
    channels: List[int]
    down: List[int]
    num_actions: Optional[int] = None


class IndependentActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig, num_agents: int, mode: str = "self-play") -> None:
        '''
        params `mode`: "self-play", "cooperative", "random"
        '''
        super().__init__()
        self.is_ma = True
        self.num_agents = num_agents
        self._actual_num_agents = num_agents
        self.mode = mode
        self.cfg = cfg

        if mode == "self-play":
            # Here we only consider 2 agents
            self._actual_num_agents = 1     # only control the main agent, i.e., 1st agent
        elif mode == "random":
            self._actual_num_agents = 1     # only control the main agent, i.e., 1st agent

        self.agents = []
        for _ in range(self._actual_num_agents):
            self.agents.append(
                ActorCritic(cfg)
            )
        
        self.agents = nn.ModuleList(self.agents)

        if mode == "self-play":
            self.competetitor = ActorCritic(cfg)
            self.competetitor.load_state_dict(self.agents[-1].state_dict())
            self.competetitor.requires_grad_(False)

        self.lstm_dim = cfg.lstm_dim

        self.env_loop = None
        self.loss_cfg = None

    @property
    def device(self) -> torch.device:
        return self.agents[0].lstm.weight_hh.device
    
    def update_other_agents(self):
        if self.mode == "self-play":
            self.competetitor.load_state_dict(self.agents[-1].state_dict())
            self.competetitor.requires_grad_(False)

    def setup_training(self, rl_env, loss_cfg: ActorCriticLossConfig) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        self.env_loop = make_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 5    # (b, n, c, h, w)
        input_hx, input_cx = hx_cx  # input_hx -> (n, b, lstm_dim)

        output_hx = torch.empty_like(input_hx)
        output_cx = torch.empty_like(input_cx)

        logits_act = []
        vals = []

        for i, agent in enumerate(self.agents):
            x = agent.encoder(obs[:, i])
            x = x.flatten(start_dim=1)
            hx, cx = agent.lstm(x, (input_hx[i], input_cx[i]))
            logits_act.append(agent.actor_linear(hx))
            vals.append(agent.critic_linear(hx).squeeze(dim=1))

            output_hx[i] = hx
            output_cx[i] = cx

        for j in range(self.num_agents - self._actual_num_agents):
            if self.mode == "random":
                logits_act.append((torch.ones_like(logits_act[-1], device=vals[-1].device) / self.cfg.num_actions).detach())
                vals.append(torch.zeros_like(vals[-1], device=vals[-1].device).detach())
            elif self.mode == "self-play":
                x = agent.encoder(obs[:, j + self._actual_num_agents])
                x = x.flatten(start_dim=1)
                hx, cx = agent.lstm(x, (input_hx[j + self._actual_num_agents], input_cx[j + self._actual_num_agents]))
                logits_act.append(agent.actor_linear(hx).detach())
                vals.append(agent.critic_linear(hx).squeeze(dim=1).detach())

                output_hx[j + self._actual_num_agents] = hx
                output_cx[j + self._actual_num_agents] = cx

        logits_act = torch.stack(logits_act, dim=1)
        vals = torch.stack(vals, dim=1)
        return ActorCriticOutput(logits_act, vals, (output_hx, output_cx))

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = self.env_loop.send(c.backup_every)

        if self.is_ma:
            # compute returns first
            end     = repeat(end, 'b h -> b h n', n=self.num_agents)
            trunc   = repeat(trunc, 'b h -> b h n', n=self.num_agents)
            lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

            metrics = {}
            loss = 0.
            for agent_id in range(self._actual_num_agents):
                d = Categorical(logits=logits_act[:, :, agent_id])
                entropy = d.entropy().mean()

                loss_actions = (-d.log_prob(act[:, :, agent_id]) * (lambda_returns[:, :, agent_id] - val[:, :, agent_id]).detach()).mean()
                loss_values = c.weight_value_loss * F.mse_loss(val[:, :, agent_id], lambda_returns[:, :, agent_id])
                loss_entropy = -c.weight_entropy_loss * entropy

                cur_agent_loss = loss_actions + loss_entropy + loss_values

                loss += cur_agent_loss

                cur_metrics = {
                    f"agent_{agent_id}/policy_entropy": entropy.detach() / math.log(2),
                    f"agent_{agent_id}/loss_actions": loss_actions.detach(),
                    f"agent_{agent_id}/loss_entropy": loss_entropy.detach(),
                    f"agent_{agent_id}/loss_values": loss_values.detach(),
                    f"agent_{agent_id}/loss_total": cur_agent_loss.detach(),
                }
                metrics.update(cur_metrics)

            metrics['loss_all_agents'] = loss.detach()

        else:
            d = Categorical(logits=logits_act)
            entropy = d.entropy().mean()

            lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

            loss_actions = (-d.log_prob(act) * (lambda_returns - val).detach()).mean()
            loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
            loss_entropy = -c.weight_entropy_loss * entropy

            loss = loss_actions + loss_entropy + loss_values

            metrics = {
                "policy_entropy": entropy.detach() / math.log(2),
                "loss_actions": loss_actions.detach(),
                "loss_entropy": loss_entropy.detach(),
                "loss_values": loss_values.detach(),
                "loss_total": loss.detach(),
            }

        return loss, metrics
    

class ActorCritic(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        self.is_ma = False
        self.encoder = ActorCriticEncoder(cfg)
        self.lstm_dim = cfg.lstm_dim
        input_dim_lstm = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.lstm = nn.LSTMCell(input_dim_lstm, cfg.lstm_dim)
        self.critic_linear = nn.Linear(cfg.lstm_dim, 1)
        self.actor_linear = nn.Linear(cfg.lstm_dim, cfg.num_actions)

        self.actor_linear.weight.data.fill_(0)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data.fill_(0)
        self.critic_linear.bias.data.fill_(0)
        init_lstm(self.lstm)

        self.env_loop = None
        self.loss_cfg = None

    @property
    def device(self) -> torch.device:
        return self.lstm.weight_hh.device

    def setup_training(self, rl_env, loss_cfg: ActorCriticLossConfig) -> None:
        assert self.env_loop is None and self.loss_cfg is None
        self.env_loop = make_env_loop(rl_env, self)
        self.loss_cfg = loss_cfg

    def predict_act_value(self, obs: Tensor, hx_cx: Tuple[Tensor, Tensor]) -> ActorCriticOutput:
        assert obs.ndim == 4
        x = self.encoder(obs)
        x = x.flatten(start_dim=1)
        hx, cx = self.lstm(x, hx_cx)
        return ActorCriticOutput(self.actor_linear(hx), self.critic_linear(hx).squeeze(dim=1), (hx, cx))

    def forward(self) -> LossAndLogs:
        c = self.loss_cfg
        _, act, rew, end, trunc, logits_act, val, val_bootstrap, _ = self.env_loop.send(c.backup_every)

        d = Categorical(logits=logits_act)
        entropy = d.entropy().mean()

        lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, c.gamma, c.lambda_)

        loss_actions = (-d.log_prob(act) * (lambda_returns - val).detach()).mean()
        loss_values = c.weight_value_loss * F.mse_loss(val, lambda_returns)
        loss_entropy = -c.weight_entropy_loss * entropy

        loss = loss_actions + loss_entropy + loss_values

        metrics = {
            "policy_entropy": entropy.detach() / math.log(2),
            "loss_actions": loss_actions.detach(),
            "loss_entropy": loss_entropy.detach(),
            "loss_values": loss_values.detach(),
            "loss_total": loss.detach(),
        }

        return loss, metrics


class ActorCriticEncoder(nn.Module):
    def __init__(self, cfg: ActorCriticConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)


@torch.no_grad()
def compute_lambda_returns(
    rew: Tensor,
    end: Tensor,
    trunc: Tensor,
    val_bootstrap: Tensor,
    gamma: float,
    lambda_: float,
) -> Tensor:
    # assert rew.ndim == 2 and rew.size() == end.size() == trunc.size() == val_bootstrap.size()
    assert rew.size() == end.size() == trunc.size() == val_bootstrap.size()

    # rew = rew.sign()  # clip reward

    end_or_trunc = (end + trunc).clip(max=1)
    not_end = 1 - end
    not_trunc = 1 - trunc

    lambda_returns = rew + not_end * gamma * (not_trunc * (1 - lambda_) + trunc) * val_bootstrap

    if lambda_ == 0:
        return lambda_returns

    last = val_bootstrap[:, -1]
    for t in reversed(range(rew.size(1))):
        lambda_returns[:, t] += end_or_trunc[:, t].logical_not() * gamma * lambda_ * last
        last = lambda_returns[:, t]

    return lambda_returns

@torch.no_grad()
def normalize_advantage(
    adv: Tensor,
    end: Tensor,
    num_agents: int,
):
    device = adv.device
    dtype = adv.dtype
    adv = adv.cpu().numpy()
    end = end.cpu().numpy()
    assert adv.ndim <= 3
    if adv.ndim == 3 and adv.shape[-1] == num_agents:
        adv_ori = adv.copy()
        adv_copy = adv.copy()
        adv_copy[end == 1] = np.nan
        mean_adv = np.nanmean(adv_copy)
        std_adv = np.nanstd(adv_copy)
        normalized_adv = (adv_ori - mean_adv) / (std_adv + 1e-5)

    else:
        adv_ori_list = []
        adv_copy_list = []
        for agent_id in range(num_agents):
            adv_ori = adv.copy()
            adv_ori_list.append(adv_ori)
            adv_copy = adv.copy()
            adv_copy[end == 1] = np.nan
            adv_copy_list.append(adv_copy)
        adv_ori_tensor = np.stack(adv_ori_list, axis=-1)
        adv_copy_tensor = np.stack(adv_copy_list, axis=-1)
        mean_adv = np.nanmean(adv_copy_tensor)
        std_adv = np.nanstd(adv_copy_tensor)
        normalized_adv = (adv_ori_tensor - mean_adv) / (std_adv + 1e-5)
    
    return torch.as_tensor(normalized_adv, device=device, dtype=dtype)


def compute_lambda_returns_with_pcont_wo_end(reward, pcont, val_bootstrap, lmbda, gamma):
    '''
    val_bootstrap: next_value
    pcont: probability of continuation
    '''
    discount = pcont * gamma
    target = reward + gamma * discount * val_bootstrap * (1 - lmbda)

    outputs = []
    accumulated_reward = val_bootstrap[:, -1]
    for t in reversed(range(reward.shape[1])):
        accumulated_reward = target[:, t] + discount[:, t] * accumulated_reward * lmbda
        outputs.append(accumulated_reward)

    returns = torch.flip(torch.stack(outputs, dim=1), [1])
    return returns