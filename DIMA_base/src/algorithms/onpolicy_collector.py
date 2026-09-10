"""Real-environment rollouts in the shape ``CommunicativeMAPPO.update`` expects.

This is the model-free floor: the same actor, the same critic, the same PPO
update as the world-model run, with imagination replaced by the real MATE
environment.  Any gain the world model claims has to be measured against this,
per environment step -- otherwise the diffusion stack is being credited for what
plain MAPPO already does.

Transitions are also written to the replay buffer, even though PPO itself never
reads them: the trajectory head is fitted on windows of real target positions,
and that fit should not depend on which of the two training modes is running.

``WorldModelTrainer.imagine`` is the counterpart that produces the same
dictionary from imagined states, so the two paths stay swappable behind one
``policy.update`` call.
"""

import numpy as np
import torch

from evaluate import weighted_mean


class OnPolicyCollector:
    """Steps the real environment and returns one on-policy PPO batch.

    The running episode is kept across calls, so an iteration boundary does not
    truncate an episode; the value of the state the rollout stopped in
    bootstraps the returns.
    """

    def __init__(self, env, policy, gamma, buffer=None):
        self.env = env
        self.policy = policy
        self.gamma = gamma
        self.buffer = buffer
        self.device = policy.device

        self.current = env.reset()
        self.prev_actions = torch.zeros(env.n_agents, env.action_dim, device=self.device)
        self.hidden = policy.actor.initial_state(env.n_agents, self.device)
        self.episode_return = 0.0
        self.episode_coverage = []
        self.finished_episodes = []

    def _tensor(self, array):
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def collect(self, num_steps):
        """Step until ``num_steps`` MATE steps are consumed.

        Returns ``(rollout, consumed, stats)``.  The budget counts MATE steps,
        not decisions, so it is comparable with the world-model run at the same
        frame skip.
        """

        n_agents = self.env.n_agents
        obs_seq, belief_seq, prev_seq, hidden_seq = [], [], [], []
        sample_seq, logp_seq = [], []
        state_seq, value_seq, reward_seq, continue_seq = [], [], [], []

        consumed = 0
        while consumed < num_steps:
            obs = self._tensor(self.current['obs'])
            beliefs = self._tensor(self.current['belief'])
            state = self._tensor(self.current['state'])

            features = self.policy.features(obs, beliefs)
            sample, log_prob, hidden = self.policy.actor(
                features, self.prev_actions, self.hidden
            )
            action = sample.clamp(-1.0, 1.0)
            value = self.policy.critic(state.unsqueeze(0)).squeeze(0)

            nxt, reward, done, info = self.env.step(action.cpu().numpy())

            obs_seq.append(obs)
            belief_seq.append(beliefs)
            prev_seq.append(self.prev_actions)
            hidden_seq.append(self.hidden)
            sample_seq.append(sample)
            logp_seq.append(log_prob)
            state_seq.append(state)
            value_seq.append(value)
            reward_seq.append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
            continue_seq.append(
                torch.as_tensor(
                    self.gamma * (1.0 - float(done)), dtype=torch.float32, device=self.device
                )
            )

            if self.buffer is not None:
                self.buffer.add(
                    state=self.current['state'],
                    obs=self.current['obs'],
                    beliefs=self.current['belief'],
                    actions=action.cpu().numpy(),
                    prev_actions=self.prev_actions.cpu().numpy(),
                    hidden=self.hidden.cpu().numpy(),
                    target_positions=self.current['target_positions'],
                    reward=reward,
                    next_state=nxt['state'],
                    next_beliefs=nxt['belief'],
                    done=done,
                )

            consumed += info['env_steps']
            self.prev_actions = action
            self.hidden = hidden
            self.episode_return += reward
            self.episode_coverage.append((info['coverage_rate'], info['env_steps']))
            self.current = nxt

            if done:
                self.finished_episodes.append(
                    {
                        'return': self.episode_return,
                        'coverage_rate': weighted_mean(self.episode_coverage),
                    }
                )
                self.episode_return = 0.0
                self.episode_coverage = []
                self.current = self.env.reset()
                self.prev_actions = torch.zeros_like(self.prev_actions)
                self.hidden = torch.zeros_like(self.hidden)

        horizon = len(reward_seq)
        bootstrap = self.policy.critic(self._tensor(self.current['state']).unsqueeze(0))

        # (H, 1): one trajectory, as many steps as the budget bought.  The team
        # reward, the global state and the critic are all per step, not per
        # agent, which is the same layout imagination produces with B rollouts.
        rewards = torch.stack(reward_seq).view(horizon, 1)
        values = torch.stack(value_seq).view(horizon, 1)
        continues = torch.stack(continue_seq).view(horizon, 1)
        returns = self.policy.lambda_returns(rewards, values, continues, bootstrap)
        advantages = returns - values

        repeat = lambda x: x.unsqueeze(-1).expand(-1, -1, n_agents).reshape(-1)

        rollout = {
            'obs': torch.cat(obs_seq, dim=0),
            'beliefs': torch.cat(belief_seq, dim=0),
            'prev_actions': torch.cat(prev_seq, dim=0),
            'hidden': torch.cat(hidden_seq, dim=0),
            'samples': torch.cat(sample_seq, dim=0),
            'old_log_probs': torch.cat(logp_seq, dim=0),
            'advantages': repeat(advantages),
            'states': torch.stack(state_seq)
            .unsqueeze(1)
            .expand(-1, n_agents, -1)
            .reshape(horizon * n_agents, -1),
            'returns': repeat(returns),
            'old_values': repeat(values),
        }
        stats = {
            'rollout/reward_mean': rewards.mean().item(),
            'rollout/value_mean': values.mean().item(),
            'rollout/return_mean': returns.mean().item(),
            'rollout/decisions': float(horizon),
        }
        return rollout, consumed, stats

    def drain_stats(self):
        if not self.finished_episodes:
            return {}
        stats = {
            'env/episode_return': float(np.mean([e['return'] for e in self.finished_episodes])),
            'env/coverage_rate': float(
                np.mean([e['coverage_rate'] for e in self.finished_episodes])
            ),
            'env/episodes': len(self.finished_episodes),
        }
        self.finished_episodes = []
        return stats
