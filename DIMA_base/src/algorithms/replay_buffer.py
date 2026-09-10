"""Ring buffer of real MATE transitions.

Four fields are worth explaining:

``beliefs``       what each camera knew after the peer-to-peer round -- its own
                  target slots merged with its neighbours', plus how stale each
                  slot is.  This is the actor's input and the autoencoder's
                  reconstruction target, so imagination can hand the policy the
                  same kind of thing the environment does.

``prev_actions``  the command each camera issued one step earlier.  The
                  observation carries the camera's current angle but not the
                  direction it was turning, so without this the policy cannot
                  tell a sweep from a hold, and a camera that currently sees
                  nothing has no way to keep doing what it was doing.

``hidden``        the actor's recurrent belief when it chose that transition's
                  action, so imagination can start from the state the policy was
                  actually in rather than from a blank memory.

``target_positions``  where the targets truly were, in belief coordinates.  A
                  window of these is the label the trajectory head is fitted to;
                  it comes from the global state, so it is training-time
                  information only and never reaches a camera.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(
        self,
        capacity,
        n_agents,
        n_targets,
        state_dim,
        obs_dim,
        belief_dim,
        action_dim,
        hidden_dim,
    ):
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.beliefs = np.zeros((capacity, n_agents, belief_dim), dtype=np.float32)
        self.next_beliefs = np.zeros((capacity, n_agents, belief_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.prev_actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.hidden = np.zeros((capacity, n_agents, hidden_dim), dtype=np.float32)
        self.target_positions = np.zeros((capacity, n_targets, 2), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        # Episode id per slot, so sequence sampling never crosses a reset.
        self.episode_ids = np.full((capacity,), -1, dtype=np.int64)
        self.current_episode = 0

    def __len__(self):
        return self.size

    def add(
        self,
        state,
        obs,
        beliefs,
        actions,
        prev_actions,
        hidden,
        target_positions,
        reward,
        next_state,
        next_beliefs,
        done,
    ):
        i = self.position
        self.states[i] = state
        self.obs[i] = obs
        self.beliefs[i] = beliefs
        self.actions[i] = actions
        self.prev_actions[i] = prev_actions
        self.hidden[i] = hidden
        self.target_positions[i] = target_positions
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.next_beliefs[i] = next_beliefs
        self.dones[i] = float(done)
        self.episode_ids[i] = self.current_episode

        if done:
            self.current_episode += 1

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    # ------------------------------------------------------------------ sampling

    def _gather(self, index, device):
        as_tensor = lambda x: torch.as_tensor(x[index], device=device)
        return {
            'states': as_tensor(self.states),
            'next_states': as_tensor(self.next_states),
            'obs': as_tensor(self.obs),
            'beliefs': as_tensor(self.beliefs),
            'next_beliefs': as_tensor(self.next_beliefs),
            'actions': as_tensor(self.actions),
            'prev_actions': as_tensor(self.prev_actions),
            'hidden': as_tensor(self.hidden),
            'target_positions': as_tensor(self.target_positions),
            'rewards': as_tensor(self.rewards),
            'dones': as_tensor(self.dones),
        }

    def sample(self, batch_size, device):
        """Independent transitions.  Every field has a leading batch dimension."""

        index = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return self._gather(index, device)

    def sample_sequences(self, batch_size, length, device):
        """Contiguous windows of ``length`` steps from a single episode.

        Fields come back as ``(batch, length, ...)``.  Falls back to
        :meth:`sample` reshaped to ``length = 1`` when the buffer holds no window
        that long yet.
        """

        if length <= 1 or self.size < length + 1:
            batch = self.sample(batch_size, device)
            return {k: v.unsqueeze(1) for k, v in batch.items()}

        starts = []
        # Rejection sampling: a window is valid when it neither wraps the write
        # head nor spans two episodes.
        for _ in range(8 * batch_size):
            if len(starts) >= batch_size:
                break
            start = np.random.randint(0, self.size - length)
            window = np.arange(start, start + length)
            if self.size == self.capacity and (
                self.position - 1 in window[:-1] or self.position in window[:-1]
            ):
                continue
            if self.episode_ids[start] != self.episode_ids[start + length - 1]:
                continue
            starts.append(window)

        if not starts:
            batch = self.sample(batch_size, device)
            return {k: v.unsqueeze(1) for k, v in batch.items()}

        return self._gather(np.stack(starts), device)
