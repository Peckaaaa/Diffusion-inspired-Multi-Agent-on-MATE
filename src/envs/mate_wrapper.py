"""MATE wrapper that produces the tuple the world model is trained on.

One ``step`` yields ``(s_t, o_t^{1:n}, m_t^{in}, m_t^{emit}, a_t^{1:n}, r_t, s_{t+1}, done)``
with continuous camera actions and MATE's own peer-to-peer camera messages.
Only the camera team is controlled; the targets run MATE's ``GreedyTargetAgent``.

All entity counts and dimensions are read back from the scenario the resolver
picked -- nothing here is hard-coded to a particular ``NC vs. NT`` setup.

``MATE-main`` is a gymnasium port of MATE, so ``reset`` returns
``(observation, info)`` and ``step`` returns the five-tuple.
"""

import numpy as np
import torch

from envs.config_resolver import ensure_mate_importable, resolve_scenario


class RunningMeanStd:
    """Welford statistics for observation and state normalization.

    MATE's observation and state spaces both carry ``+inf`` upper bounds (target
    bounties and warehouse counters are unbounded), so min-max rescaling against
    the declared Box is not usable; empirical statistics are.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).reshape(-1, *self.mean.shape)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total

        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        out = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -clip, clip).astype(np.float32)

    def denormalize(self, x):
        if isinstance(x, torch.Tensor):
            mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(np.sqrt(self.var + 1e-8), dtype=x.dtype, device=x.device)
            return x * std + mean
        return np.asarray(x) * np.sqrt(self.var + 1e-8) + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class MATEEnv:
    """MATE ``MultiAgentTracking`` with continuous camera control and a P2P channel.

    Actions handed to :meth:`step` are normalized to ``[-1, 1]`` per dimension and
    rescaled to MATE's camera Box (``[-5, 5] x [-2.5, 2.5]``) here.
    """

    def __init__(
        self,
        scenario='MATE-4v8-9',
        max_episode_steps=200,
        msg_dim=8,
        camera_comm=True,
        reward_scale=1.0,
        seed=0,
        normalize=True,
    ):
        ensure_mate_importable()

        import mate
        from mate.agents import GreedyTargetAgent

        self.scenario = resolve_scenario(scenario)
        self.msg_dim = msg_dim
        self.camera_comm = camera_comm
        self.reward_scale = reward_scale
        self.max_episode_steps = max_episode_steps

        # make_environment() instead of gym.make(): it builds MultiAgentTracking
        # directly, with no checker wrapper in front of MATE's
        # (camera_obs, target_obs) joint-observation tuple.
        base_env = mate.make_environment(
            config=self.scenario, max_episode_steps=max_episode_steps
        )
        self.env = mate.MultiCamera(base_env, target_agent=GreedyTargetAgent())

        unwrapped = self.env.unwrapped
        self.n_agents = unwrapped.num_cameras
        self.n_targets = unwrapped.num_targets
        self.n_obstacles = unwrapped.num_obstacles
        self.obs_dim = unwrapped.camera_observation_space.shape[0]
        self.state_dim = unwrapped.state_space.shape[0]
        self.action_dim = unwrapped.camera_action_space.shape[0]

        self.action_low = unwrapped.camera_action_space.low.astype(np.float64)
        self.action_high = unwrapped.camera_action_space.high.astype(np.float64)

        self.normalize = normalize
        self.obs_rms = RunningMeanStd((self.obs_dim,))
        self.state_rms = RunningMeanStd((self.state_dim,))

        self.episode_step = 0
        # MATE-main's wrapper.seed() still calls the RandomState-only randint on
        # gymnasium's Generator, so seeding goes through the first reset instead.
        self._pending_seed = seed

    # ------------------------------------------------------------------ helpers

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()

    def describe(self):
        return (
            f'{self.scenario}: {self.n_agents} cameras, {self.n_targets} targets, '
            f'{self.n_obstacles} obstacles | state {self.state_dim}, obs {self.obs_dim}, '
            f'action {self.action_dim}, msg {self.msg_dim}'
        )

    def _norm_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float64).reshape(self.n_agents, self.obs_dim)
        if not self.normalize:
            return obs.astype(np.float32)
        self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def _norm_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(self.state_dim)
        if not self.normalize:
            return state.astype(np.float32)
        self.state_rms.update(state[None])
        return self.state_rms.normalize(state)

    def _scale_action(self, actions):
        """``[-1, 1]`` per dimension -> MATE's camera Box."""

        actions = np.clip(np.asarray(actions, dtype=np.float64), -1.0, 1.0)
        actions = actions.reshape(self.n_agents, self.action_dim)
        return self.action_low + 0.5 * (actions + 1.0) * (self.action_high - self.action_low)

    def _exchange_messages(self, messages):
        """Broadcast each camera's message and return what every camera received.

        Routing -- and therefore any communication-range, delay or dropout
        wrapper MATE has applied -- is done by the environment.  A camera's own
        broadcast is excluded from its incoming set; the remainder is mean-pooled
        so the received vector keeps a fixed ``msg_dim`` regardless of how many
        peers were in range.
        """

        received = np.zeros((self.n_agents, self.msg_dim), dtype=np.float32)
        if not self.camera_comm:
            return received

        from mate.utils import Message, Team

        messages = np.asarray(messages, dtype=np.float32).reshape(self.n_agents, self.msg_dim)
        self.env.send_messages(
            [
                Message(sender=i, recipient=None, content=messages[i].copy(), team=Team.CAMERA)
                for i in range(self.n_agents)
            ]
        )

        for i, inbox in enumerate(self.env.receive_messages()):
            peer_contents = [
                np.asarray(m.content, dtype=np.float32) for m in inbox if m.sender != i
            ]
            if peer_contents:
                received[i] = np.mean(peer_contents, axis=0)
        return received

    # ------------------------------------------------------------------- env API

    def reset(self):
        self.episode_step = 0
        if self._pending_seed is not None:
            obs, _ = self.env.reset(seed=self._pending_seed)
            self._pending_seed = None
        else:
            obs, _ = self.env.reset()
        return {
            'state': self._norm_state(self.env.unwrapped.state()),
            'obs': self._norm_obs(obs),
            'messages': np.zeros((self.n_agents, self.msg_dim), dtype=np.float32),
        }

    def step(self, actions, messages=None):
        """Exchange messages, then advance MATE by one step.

        Args:
            actions: ``(n_agents, action_dim)`` in ``[-1, 1]``.
            messages: ``(n_agents, msg_dim)`` broadcast at this step.

        Returns:
            ``(next, reward, done, info)``.  ``next['messages']`` is what each
            camera received -- the message input of the *next* decision, one step
            later, because a simultaneous exchange would be circular.
        """

        if messages is None:
            messages = np.zeros((self.n_agents, self.msg_dim), dtype=np.float32)

        received = self._exchange_messages(messages)
        obs, _, terminated, truncated, infos = self.env.step(self._scale_action(actions))
        # MATE-main folds the episode-step limit into `terminated`; `truncated` is
        # always False, but both are honored here.
        done = bool(terminated) or bool(truncated)
        self.episode_step += 1

        # MATE's raw team reward is unbounded below (measured min -198); its own
        # coverage_rate metric is the team-mean tracking rate in [0, 1] and is
        # what this task optimizes.
        coverage_rate = float(infos[0]['coverage_rate'])

        nxt = {
            'state': self._norm_state(self.env.unwrapped.state()),
            'obs': self._norm_obs(obs),
            'messages': received,
        }
        info = {
            'coverage_rate': coverage_rate,
            'real_coverage_rate': float(infos[0]['real_coverage_rate']),
            'mean_transport_rate': float(infos[0]['mean_transport_rate']),
        }
        return nxt, coverage_rate * self.reward_scale, done, info
