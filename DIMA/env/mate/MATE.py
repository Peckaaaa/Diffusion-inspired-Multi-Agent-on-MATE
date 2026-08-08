import numpy as np

_LEGACY_RANDOM_SHIM_INSTALLED = False


def _install_legacy_np_random_shim():
    """Let MATE 0.1.0 run on gym 0.26.

    gym 0.26 hands out a ``np.random.Generator``, but MATE still calls the
    ``RandomState``-only helpers (``randint``/``rand``/``randn``) in
    ``mate/agents/base.py``, ``mate/agents/mixture.py`` and
    ``mate/wrappers/single_team.py``.  ``Generator`` is an immutable C type, so
    it cannot be patched in place -- we wrap it and hand the wrapper out from
    ``gym.utils.seeding.np_random`` instead.  Everything except the three legacy
    aliases is delegated straight through to the real generator.
    """

    global _LEGACY_RANDOM_SHIM_INSTALLED
    if _LEGACY_RANDOM_SHIM_INSTALLED:
        return

    from gym.utils import seeding

    class LegacyRandomProxy:
        def __init__(self, generator):
            object.__setattr__(self, "_generator", generator)

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_generator"), name)

        def randint(self, low, high=None, size=None, dtype=int):
            return object.__getattribute__(self, "_generator").integers(low, high, size=size, dtype=dtype)

        def rand(self, *shape):
            return object.__getattribute__(self, "_generator").random(shape if shape else None)

        def randn(self, *shape):
            return object.__getattribute__(self, "_generator").standard_normal(shape if shape else None)

    original_np_random = seeding.np_random

    def np_random(seed=None):
        generator, seed = original_np_random(seed)
        return LegacyRandomProxy(generator), seed

    seeding.np_random = np_random
    _LEGACY_RANDOM_SHIM_INSTALLED = True


class MATEEnv:
    """Adapts MATE (Multi-Agent Tracking Environment) to the dict-based interface
    that ``DreamerWorker`` expects, in the same shape as ``PettingZooMPEEnv``.

    Only the camera team is controlled; the targets are driven by MATE's builtin
    greedy agents, so this is a single-team multi-agent task.
    """

    def __init__(self, env_name, seed, levels=5, max_episode_steps=200, reward_scale=10.0):
        _install_legacy_np_random_shim()

        import mate
        from mate.agents import GreedyTargetAgent

        self.scenario = env_name
        self.levels = levels
        self.reward_scale = reward_scale
        self.max_time_steps = max_episode_steps

        config = env_name if env_name.endswith((".yaml", ".json")) else env_name + ".yaml"

        # make_environment() instead of gym.make(): gym 0.26 wraps the env in a
        # PassiveEnvChecker that mis-reads MATE's (camera_obs, target_obs) reset
        # tuple as (obs, info) and then fails the observation-space check.
        base_env = mate.make_environment(config=config, max_episode_steps=max_episode_steps)

        # DiscreteCamera turns each camera's Box(2,) rotation/zoom command into
        # Discrete(levels * levels); DIMA's discrete policy needs a finite set.
        self.env = mate.MultiCamera(
            mate.DiscreteCamera(base_env, levels=levels),
            target_agent=GreedyTargetAgent(),
        )

        unwrapped = self.env.unwrapped
        self.n_agents = unwrapped.num_cameras
        self.n_obs = unwrapped.camera_observation_space.shape[0]
        self.state_dim = unwrapped.state_space.shape[0]
        self.n_actions = levels * levels
        self.discrete = True

        self.observation_space = self.repeat(unwrapped.camera_observation_space)
        self.share_observation_space = self.repeat(unwrapped.state_space)
        self.action_space = list(self.env.action_space.spaces)

        self.cur_step = 0
        self._seed = seed
        self.seed(seed)

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        joint_action = np.asarray(actions, dtype=np.int64).ravel()
        obs, _, done, infos = self.env.step(joint_action)
        self.cur_step += 1

        # MATE returns one shared scalar reward for the whole camera team, and its
        # raw scale (measured min -198, max +5) sits well outside the critic's
        # [-10, 10] support.  coverage_rate is MATE's own tracking metric and is
        # bounded in [0, 1], so it is rescaled instead.
        coverage_rate = float(infos[0]["coverage_rate"])
        reward = coverage_rate * self.reward_scale

        rewards = {i: [reward] for i in range(self.n_agents)}
        dones = {i: bool(done) for i in range(self.n_agents)}

        return (
            self.wrap(obs),
            self.wrap_state(),
            rewards,
            dones,
            infos,
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self.cur_step = 0
        obs = self.env.reset()
        return self.wrap(obs), self.wrap_state(), self.get_avail_actions()

    def get_avail_actions(self):
        # Every camera action is always legal.  DreamerMemory only allocates an
        # av_actions buffer for SC2/SMACv2, so this has to stay None.
        return None

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed
        return self.env.seed(seed)

    def wrap(self, l):
        return {i: l[i] for i in range(self.n_agents)}

    def wrap_state(self):
        state = self.env.unwrapped.state()
        return {i: state for i in range(self.n_agents)}

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
