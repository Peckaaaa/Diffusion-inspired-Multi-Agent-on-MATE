"""Ring buffer of real MATE transitions.

Two kinds of message are stored, and they are not interchangeable:

``messages_in``   what each camera received -- the actor's input and the
                  autoencoder's reconstruction target.
``messages_out``  what each camera broadcast -- the diffusion model's
                  conditioning, because the emission is what actually moves the
                  next state, and because conditioning on it is the only thing
                  that gives the actor's message head a policy gradient.
"""

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, n_agents, state_dim, obs_dim, msg_dim, action_dim):
        self.capacity = capacity
        self.size = 0
        self.position = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.obs = np.zeros((capacity, n_agents, obs_dim), dtype=np.float32)
        self.messages_in = np.zeros((capacity, n_agents, msg_dim), dtype=np.float32)
        self.messages_out = np.zeros((capacity, n_agents, msg_dim), dtype=np.float32)
        self.next_messages_in = np.zeros((capacity, n_agents, msg_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
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
        messages_in,
        messages_out,
        actions,
        reward,
        next_state,
        next_messages_in,
        done,
    ):
        i = self.position
        self.states[i] = state
        self.obs[i] = obs
        self.messages_in[i] = messages_in
        self.messages_out[i] = messages_out
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_states[i] = next_state
        self.next_messages_in[i] = next_messages_in
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
            'messages_in': as_tensor(self.messages_in),
            'messages_out': as_tensor(self.messages_out),
            'next_messages_in': as_tensor(self.next_messages_in),
            'actions': as_tensor(self.actions),
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
