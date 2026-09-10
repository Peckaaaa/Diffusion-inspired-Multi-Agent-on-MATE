"""MAPPO over the belief pipeline, trained on imagined or real rollouts.

The decentralized actor is ``pi(a | f^i, a_prev^i, h^i)`` where the feature it
reads is everything the camera knows after the peer-to-peer round:

    f_t^i = [ hat_o_t^i , b_t^i , hat_p_t^i ]

its own observation, the belief it merged from its neighbours' target slots, and
the short forward roll the trajectory head predicts from that belief.  The
recurrent state ``h`` carries what none of those three hold: what the camera has
been doing and seeing over the episode.

Every one of those inputs is there because a measurement said the ones before it
were not enough -- a stateless actor on a single observation explains 35% of the
reference agent's commands, adding the previous command takes that to 85% on the
states that agent visits but leaves an error of 1.3 (against an action variance
of 0.446) on the states a cloned policy reaches itself.

The trajectory head is trained by supervised regression against true future
positions rather than by the policy gradient, and it is detached where it enters
the actor: it is a perception module with its own signal, not a value-driven
one.  The centralized critic ``V(hat_s)`` reads the global state decoded from
the same FSQ code the actor's observation came from.

The message a camera broadcasts is no longer part of its action.  It publishes
its observed target slots, which is what makes the received block mergeable and
the consensus term meaningful; a learned vector could carry as much information
but nothing downstream could hold two cameras to a shared meaning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trajectory import TrajectoryLearner


def mlp(sizes, activation=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class CommunicativeActor(nn.Module):
    """Gaussian over the rotation/zoom command, given the belief the GRU carries."""

    def __init__(self, feature_dim, action_dim, hidden_dim=256, init_log_std=-0.5):
        super().__init__()

        self.action_dim = action_dim

        self.body = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.memory = nn.GRUCell(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), float(init_log_std)))

        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.zeros_(self.mean.bias)

    LOG_STD_BOUNDS = (-1.2, 0.0)

    def std(self):
        """Per-dimension standard deviation, bounded at both ends.

        Upper bound 0.0 caps sigma at 1.0: the mean is tanh-squashed into the
        normalized action box, so a wider Gaussian can only produce samples the
        env wrapper clips away while the entropy bonus keeps pushing log_std up.

        Lower bound -1.2 floors sigma at 0.30.  With the entropy coefficient at
        its own floor the PPO gradient still drove sigma down without slowing,
        and coverage fell once it went under ~0.37: the policy was sharpening
        onto a mode the real environment does not reward.  A hard bound holds
        where a bonus term cannot.
        """
        return self.log_std.clamp(*self.LOG_STD_BOUNDS).exp()

    def initial_state(self, batch_size, device=None):
        """The belief a camera starts an episode with: nothing remembered."""

        return torch.zeros(batch_size, self.hidden_dim, device=device or self.log_std.device)

    def distribution(self, features, prev_actions, hidden):
        """Returns ``(distribution, next_hidden)``; the caller carries the state."""

        encoded = self.body(torch.cat([features, prev_actions], dim=-1))
        hidden = self.memory(encoded, hidden)
        # tanh on the mean keeps the command inside the normalized action box the
        # env wrapper expects; the sample itself stays Gaussian, so the PPO
        # log-probability needs no change-of-variable correction.
        mean = torch.tanh(self.mean(hidden))
        std = self.std().expand_as(mean)
        return torch.distributions.Normal(mean, std), hidden

    def forward(self, features, prev_actions, hidden, deterministic=False):
        dist, hidden = self.distribution(features, prev_actions, hidden)
        sample = dist.mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(sample).sum(dim=-1)
        return sample, log_prob, hidden

    def evaluate(self, features, prev_actions, hidden, sample):
        dist, _ = self.distribution(features, prev_actions, hidden)
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
    """PPO-clip on the actor, supervised regression on the trajectory head."""

    def __init__(
        self,
        obs_dim,
        belief_dim,
        action_dim,
        state_dim,
        n_targets,
        trajectory_horizon=4,
        consensus_weight=0.1,
        trajectory_lr=3e-4,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=1e-3,
        clip_ratio=0.2,
        entropy_coef=3e-3,
        entropy_coef_min=3e-4,
        entropy_decay_steps=100000,
        value_coef=0.5,
        max_grad_norm=0.5,
        gamma=0.99,
        lam=0.95,
        ppo_epochs=5,
        num_minibatches=4,
        target_kl=0.015,
        device='cpu',
    ):
        self.device = torch.device(device)
        self.n_targets = n_targets
        self.trajectory_horizon = trajectory_horizon
        self.consensus_weight = consensus_weight

        # The same head the planner uses, trained the same way; here it feeds
        # the actor's features instead of a rollout score.
        self.trajectory_learner = TrajectoryLearner(
            belief_dim=belief_dim,
            n_targets=n_targets,
            horizon=trajectory_horizon,
            hidden_dim=hidden_dim,
            lr=trajectory_lr,
            consensus_weight=consensus_weight,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.trajectory = self.trajectory_learner.predictor

        feature_dim = obs_dim + belief_dim + n_targets * trajectory_horizon * 2
        self.actor = CommunicativeActor(feature_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CentralizedCritic(state_dim, hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.clip_ratio = clip_ratio

        self.entropy_coef_init = entropy_coef
        self.entropy_coef_min = entropy_coef_min
        self.entropy_decay_steps = entropy_decay_steps
        self.entropy_coef = entropy_coef

        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.target_kl = target_kl

    # ----------------------------------------------------------------- features

    def features(self, obs, beliefs):
        """``[hat_o, b, hat_p]`` for every camera, with the prediction detached.

        Detached because the trajectory head answers to its own supervision: a
        policy gradient reaching into it would let the actor trade prediction
        accuracy for whatever the critic currently likes, and the consensus term
        would then be pulling against the return.
        """

        with torch.no_grad():
            predicted = self.trajectory(beliefs)
        return torch.cat([obs, beliefs, predicted.flatten(start_dim=1)], dim=-1)

    # -------------------------------------------------------------- interaction

    @torch.no_grad()
    def act_numpy(self, obs, beliefs, prev_actions, hidden, deterministic=False):
        """``(n, *)`` numpy in -> action and the belief state to carry on."""

        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        features = self.features(tensor(obs), tensor(beliefs))
        sample, _, hidden_t = self.actor(
            features, tensor(prev_actions), tensor(hidden), deterministic=deterministic
        )
        return sample.clamp(-1.0, 1.0).cpu().numpy(), hidden_t.cpu().numpy()

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

    # ------------------------------------------------------- trajectory head

    def update_trajectory(self, beliefs, future_positions):
        return self.trajectory_learner.update(beliefs, future_positions)

    # ------------------------------------------------------------------- update

    def update(self, batch, total_env_steps=0):
        """PPO-clip update on a flattened batch of imagined or real steps."""

        if self.entropy_decay_steps > 0:
            progress = min(1.0, total_env_steps / self.entropy_decay_steps)
            self.entropy_coef = self.entropy_coef_init - progress * (
                self.entropy_coef_init - self.entropy_coef_min
            )

        obs = batch['obs']
        beliefs = batch['beliefs']
        prev_actions = batch['prev_actions']
        hidden = batch['hidden']
        samples = batch['samples']
        old_log_probs = batch['old_log_probs']
        advantages = batch['advantages']
        states = batch['states']
        returns = batch['returns']
        old_values = batch['old_values']

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total = obs.shape[0]
        minibatch_size = max(total // self.num_minibatches, 1)
        metrics = {
            'ppo/policy_loss': 0.0,
            'ppo/value_loss': 0.0,
            'ppo/entropy': 0.0,
            'ppo/clip_frac': 0.0,
            'ppo/approx_kl': 0.0,
        }
        updates = 0
        epochs_run = 0

        for _ in range(self.ppo_epochs):
            epoch_kl, epoch_minibatches = 0.0, 0
            permutation = torch.randperm(total, device=self.device)
            for start in range(0, total, minibatch_size):
                index = permutation[start : start + minibatch_size]

                # Recomputed rather than stored: the trajectory head moves under
                # the actor between rollout and update, and the feature the actor
                # is scored on has to be the one it would read now.
                features = self.features(obs[index], beliefs[index])
                log_probs, entropy = self.actor.evaluate(
                    features, prev_actions[index], hidden[index], samples[index]
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
                    log_ratio = log_probs - old_log_probs[index]
                    approx_kl = (ratio - 1.0 - log_ratio).mean().item()

                metrics['ppo/policy_loss'] += policy_loss.item()
                metrics['ppo/value_loss'] += value_loss.item()
                metrics['ppo/entropy'] += entropy.mean().item()
                metrics['ppo/clip_frac'] += clip_frac
                metrics['ppo/approx_kl'] += approx_kl
                updates += 1
                epoch_kl += approx_kl
                epoch_minibatches += 1

            epochs_run += 1
            # Stop the remaining epochs once this one has already moved the
            # policy far enough: late in training the update kept growing while
            # sigma shrank, which is the policy sharpening rather than improving.
            if self.target_kl > 0 and epoch_kl / max(epoch_minibatches, 1) > self.target_kl:
                break

        averaged = {k: v / max(updates, 1) for k, v in metrics.items()}
        with torch.no_grad():
            averaged['ppo/action_std_mean'] = self.actor.std().mean().item()
        averaged['ppo/entropy_coef'] = self.entropy_coef
        averaged['ppo/epochs_run'] = float(epochs_run)
        return averaged

    # ---------------------------------------------------------------- serialize

    def state_dict(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'trajectory': self.trajectory_learner.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }

    def load_state_dict(self, d):
        self.actor.load_state_dict(d['actor'])
        self.critic.load_state_dict(d['critic'])
        self.trajectory_learner.load_state_dict(d['trajectory'])
        self.actor_optimizer.load_state_dict(d['actor_optimizer'])
        self.critic_optimizer.load_state_dict(d['critic_optimizer'])
