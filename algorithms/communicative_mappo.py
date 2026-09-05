"""Communicative MAPPO trained on imagined FSQ rollouts.

Decentralized actor ``pi(a, m | hat_o^i, hat_m^i)`` -- the message the camera will
broadcast next is part of the sampled action, so it is trained by the same policy
gradient as the rotation/zoom command.  Nothing else could train it: the world
model's transition is a categorical sampling over FSQ codes, so no gradient flows
back through imagination into a deterministic message head.

Centralized critic ``V(hat_s)`` reads the global state decoded from the same FSQ
code the actor's observation came from.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, activation=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CommunicativeActor(nn.Module):
    """Gaussian over ``[a_t^i, m_{t+1}^i]`` conditioned on ``(hat_o_t^i, hat_m_t^i)``."""

    def __init__(self, obs_dim, msg_dim, action_dim, hidden_dim=256, init_log_std=-0.5):
        super().__init__()

        self.action_dim = action_dim
        self.msg_dim = msg_dim
        self.output_dim = action_dim + msg_dim

        self.body = nn.Sequential(
            nn.Linear(obs_dim + msg_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mean = nn.Linear(hidden_dim, self.output_dim)
        self.log_std = nn.Parameter(torch.full((self.output_dim,), float(init_log_std)))

        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    def distribution(self, obs, messages):
        features = self.body(torch.cat([obs, messages], dim=-1))
        # tanh on the mean keeps the command inside the normalized action box the
        # env wrapper expects; the sample itself stays Gaussian, so the PPO
        # log-probability needs no change-of-variable correction.
        mean = torch.tanh(self.mean(features))
        std = self.log_std.clamp(-5.0, 2.0).exp().expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def forward(self, obs, messages, deterministic=False):
        dist = self.distribution(obs, messages)
        sample = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(sample).sum(dim=-1)
        action, message = torch.split(sample, [self.action_dim, self.msg_dim], dim=-1)
        return action, message, log_prob, sample

    def evaluate(self, obs, messages, sample):
        dist = self.distribution(obs, messages)
        log_prob = dist.log_prob(sample).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """``V(hat_s_t)`` on the FSQ-decoded global state."""

    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.net = mlp([state_dim, hidden_dim, hidden_dim, 1])

    def forward(self, state):
        return self.net(state).squeeze(-1)


class CommunicativeMAPPO:
    """PPO-clip on imagined trajectories with generalized lambda-returns."""

    def __init__(
        self,
        obs_dim,
        msg_dim,
        action_dim,
        state_dim,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_ratio=0.2,
        entropy_coef=1e-3,
        value_coef=0.5,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        ppo_epochs=5,
        num_minibatches=4,
        device='cpu',
    ):
        self.device = torch.device(device)
        self.actor = CommunicativeActor(obs_dim, msg_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CentralizedCritic(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches

    # -------------------------------------------------------------- interaction

    @torch.no_grad()
    def act_numpy(self, obs, messages, deterministic=False):
        """``(n, obs_dim)``/``(n, msg_dim)`` numpy in -> numpy action and message out."""

        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        msg_t = torch.as_tensor(messages, dtype=torch.float32, device=self.device)
        action, message, _, _ = self.actor(obs_t, msg_t, deterministic=deterministic)
        return (
            action.clamp(-1.0, 1.0).cpu().numpy(),
            message.cpu().numpy(),
        )

    # ------------------------------------------------------------------ returns

    def lambda_returns(self, rewards, values, continues, bootstrap):
        """Generalized lambda-return over an imagined horizon.

        Args:
            rewards, values, continues: ``(H, B)``; ``continues`` is ``gamma * (1 - done)``.
            bootstrap: ``(B,)`` value of the state after the last imagined step.
        """

        horizon = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        next_return = bootstrap
        for t in reversed(range(horizon)):
            next_value = values[t + 1] if t + 1 < horizon else bootstrap
            next_return = rewards[t] + continues[t] * (
                (1.0 - self.lam) * next_value + self.lam * next_return
            )
            returns[t] = next_return
        return returns

    # ------------------------------------------------------------------- update

    def update(self, batch):
        """PPO-clip update on a flattened imagined batch.

        ``batch`` holds tensors of shape ``(M, ...)`` where ``M = H * B * n`` for the
        actor tensors and ``M`` is broadcast from ``H * B`` for the critic ones:
        ``obs``, ``messages``, ``samples``, ``old_log_probs``, ``advantages``,
        ``states``, ``returns``, ``old_values``.
        """

        obs = batch['obs']
        messages = batch['messages']
        samples = batch['samples']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        states = batch['states']
        returns = batch['returns']
        old_values = batch['old_values']

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total = obs.shape[0]
        minibatch_size = max(total // self.num_minibatches, 1)
        metrics = {'ppo/policy_loss': 0.0, 'ppo/value_loss': 0.0, 'ppo/entropy': 0.0, 'ppo/clip_frac': 0.0}
        updates = 0

        for _ in range(self.ppo_epochs):
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                index = permutation[start : start + minibatch_size]

                log_probs, entropy = self.actor.evaluate(
                    obs[index], messages[index], samples[index]
                )
                ratio = (log_probs - old_log_probs[index]).exp()
                unclipped = ratio * advantages[index]
                clipped = ratio.clamp(1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages[index]
                policy_loss = -torch.min(unclipped, clipped).mean()
                entropy_loss = -entropy.mean()

                self.actor_optimizer.zero_grad(set_to_none=True)
                (policy_loss + self.entropy_coef * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                values = self.critic(states[index])
                value_clipped = old_values[index] + (values - old_values[index]).clamp(
                    -self.clip_ratio, self.clip_ratio
                )
                value_loss = self.value_coef * torch.max(
                    F.mse_loss(values, returns[index], reduction='none'),
                    F.mse_loss(value_clipped, returns[index], reduction='none'),
                ).mean()

                self.critic_optimizer.zero_grad(set_to_none=True)
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_ratio).float().mean().item()

                metrics['ppo/policy_loss'] += policy_loss.item()
                metrics['ppo/value_loss'] += value_loss.item()
                metrics['ppo/entropy'] += entropy.mean().item()
                metrics['ppo/clip_frac'] += clip_frac
                updates += 1

        return {k: v / max(updates, 1) for k, v in metrics.items()}

    # ---------------------------------------------------------------- serialize

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.actor_optimizer.load_state_dict(d['actor_optimizer'])
        self.critic_optimizer.load_state_dict(d['critic_optimizer'])
