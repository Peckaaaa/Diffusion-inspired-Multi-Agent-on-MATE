"""MATE -> DIMA environment adapter (brief section 8).

This is the only place in the project that knows both MATE's API and DIMA's
tensor conventions.  It builds a MATE environment out of MATE's own wrappers --
it does not reimplement any of them -- and exposes:

* the six numbers DIMA's ``train.py:get_env_info()`` reads off an environment
  (``n_agents``, ``n_obs``, ``state_dim``, ``n_actions``, ``discrete``,
  ``max_time_steps``), so DIMA's configuration stays dimension-free;
* a step/reset pair returning a :class:`MATEObservation`, which keeps the
  decentralised and the privileged views in *separate, differently named fields*
  (brief section 9).

Wrapper stack
-------------
::

    mate.make_environment(config=<scenario>.yaml, max_episode_steps=N)
      -> mate.DiscreteCamera(levels=L)              # Box(2,) -> Discrete(L*L)
        -> mate.MultiCamera(target_agent=...)       # targets played by MATE itself
          -> mate.RepeatedRewardIndividualDone      # per-camera reward / done
            -> [mate.AuxiliaryCameraRewards]        # optional, only if shaping asked for

This is MATE's own canonical discrete-camera stack; compare
``mate/examples/iql/camera/config.py:19-51``.

Deliberate difference from that example: ``mate.RescaledObservation`` is **not**
in the stack.  Planners in this project are MATE ``CameraAgentBase`` agents, and
MATE's rule-based agents (``GreedyCameraAgent`` in particular) compute distances
and sight ranges in MATE world units -- feeding them rescaled observations would
silently degrade the very baseline section 15 asks us to reproduce faithfully.
Rescaling is therefore applied *only* to the copy handed to DIMA, using MATE's
own :func:`mate.rescale_observation`, which is the identical function
``RescaledObservation`` calls.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

import mate
from mate.agents.utils import normalize_observation, rescale_observation
from mate.utils import Team


__all__ = [
    'MATEObservation',
    'MATEEnv',
    'resolve_scenario',
    'denormalize_observation',
    'DEFAULT_SCENARIO',
]


def denormalize_observation(rescaled: np.ndarray, observation_space) -> np.ndarray:
    """Exact inverse of :func:`mate.agents.utils.normalize_observation`.

    MATE ships the forward direction only.  The world model predicts observations
    in the rescaled space it was trained on, so turning a prediction back into
    something a planner can reason about geometrically -- positions, angles,
    sight ranges -- needs the inverse.

    The forward pass (``mate/agents/utils.py:97-128``) does, in order::

        x[bounded_below] -= low[bounded_below]
        x[mask]           = 2 * x[mask] / (high - low)[mask] - 1

    where ``mask = bounded_below & bounded_above & (high > low)``, so ``mask`` is
    a subset of ``bounded_below``.  Undoing it means reversing both steps.
    Dimensions that are unbounded above (entity counts, the agent index, the
    freight/bounty tail of the global state) are shifted but never scaled, and
    round-trip exactly.
    """

    x = np.array(rescaled, dtype=np.float64, copy=True)
    low = observation_space.low
    high = observation_space.high

    bounded_below = observation_space.bounded_below
    bounded_above = observation_space.bounded_above
    mask = np.logical_and(np.logical_and(bounded_below, bounded_above), high > low)

    x[..., mask] = (x[..., mask] + 1.0) * (high - low)[mask] / 2.0
    x[..., bounded_below] = x[..., bounded_below] + low[bounded_below]
    return x


DEFAULT_SCENARIO = 'MATE-4v2-9'

#: Scenario yaml files shipped by MATE.  Nothing here is re-declared: the number
#: of cameras, targets, obstacles and the episode limit all come out of the yaml.
SCENARIO_DIR = Path(mate.ASSETS_DIR)


def resolve_scenario(scenario: str) -> str:
    """Resolve a scenario name or path to a MATE configuration file.

    ``'MATE-4v2-9'``, ``'MATE-4v2-9.yaml'`` and an absolute path all work.
    """

    candidate = Path(scenario)
    if candidate.suffix in {'.yaml', '.yml', '.json'} and candidate.is_file():
        return str(candidate)

    name = scenario if scenario.endswith(('.yaml', '.yml', '.json')) else f'{scenario}.yaml'
    path = SCENARIO_DIR / name
    if not path.is_file():
        available = sorted(p.stem for p in SCENARIO_DIR.glob('MATE*.yaml'))
        raise FileNotFoundError(
            f'Unknown MATE scenario {scenario!r}. Available: {", ".join(available)}'
        )
    return str(path)


@dataclass(frozen=True)
class MATEObservation:
    """One environment timestep, with the CTDE boundary made explicit.

    Brief section 9 requires local observations and privileged state never to be
    confused.  The two are separate fields here, and every privileged field is
    named ``state*`` and documented as such.

    Decentralised (safe for execution-time policies and planners):
        ``obs_raw``  -- ``(n_cameras, obs_dim)``, MATE world units.
        ``obs``      -- ``(n_cameras, obs_dim)``, rescaled to ``[-1, 1]``.

    PRIVILEGED (training / world model / diagnostics only -- never hand these to
    a planner that claims to be decentralised):
        ``state_raw`` -- ``(state_dim,)``, ``MultiAgentTracking.state()``.
        ``state``     -- ``(state_dim,)``, rescaled where the state space is bounded.
    """

    obs_raw: np.ndarray
    obs: np.ndarray
    state_raw: np.ndarray
    state: np.ndarray
    infos: List[dict]
    episode_step: int

    @property
    def coverage_rate(self) -> float:
        """MATE's own tracking metric for this step (0 before the first step)."""

        return float(self.infos[0]['coverage_rate']) if self.infos else 0.0


class MATEEnv:
    """A MATE camera-team environment presented in the terms DIMA needs.

    Parameters
    ----------
    scenario:
        MATE scenario name or yaml path.  Everything about the environment's
        size -- number of cameras, targets, obstacles, episode limit -- is read
        from this file; nothing is re-declared in this project (brief section 4).
    discrete_levels:
        ``levels`` for :class:`mate.DiscreteCamera`.  This is the one genuinely
        free parameter: MATE's camera action is continuous ``Box(2,)`` and DIMA's
        discrete policy/denoiser path needs a finite action set, so the
        discretisation resolution must be chosen, not detected.  ``5`` gives
        ``Discrete(25)`` and matches MATE's own examples.
    max_episode_steps:
        Overrides the scenario's ``max_episode_steps`` (``10000`` for
        ``MATE-4v2-9``, far too long for world-model debugging).  ``None`` keeps
        the scenario value.
    target_agent_factory:
        Callable returning the ``TargetAgentBase`` that plays the target team.
        Defaults to MATE's ``GreedyTargetAgent``, matching MATE's own examples
        and ``mate.evaluate``'s default opponent.
    reward_coefficients:
        ``None`` (default) keeps MATE's raw camera-team reward untouched.  A dict
        activates MATE's own :class:`mate.AuxiliaryCameraRewards` wrapper, e.g.
        ``{'coverage_rate': 1.0}``.  No reward shaping is implemented here.
    """

    def __init__(
        self,
        scenario: str = DEFAULT_SCENARIO,
        seed: int = 0,
        discrete_levels: int = 5,
        max_episode_steps: Optional[int] = 200,
        target_agent_factory: Optional[Callable[[], Any]] = None,
        reward_coefficients: Optional[Dict[str, float]] = None,
        reward_reduction: str = 'mean',
    ) -> None:
        self.scenario = scenario
        self.config_path = resolve_scenario(scenario)
        self.discrete_levels = int(discrete_levels)
        self._seed = int(seed)
        self.reward_coefficients = reward_coefficients

        overrides: Dict[str, Any] = {}
        if max_episode_steps is not None:
            overrides['max_episode_steps'] = int(max_episode_steps)

        # `mate.make_environment` rather than `mate.make`/`gym.make`: from gym
        # 0.26 `gym.make` wraps the env in PassiveEnvChecker, which reads MATE's
        # `reset() -> (camera_obs, target_obs)` tuple as `(obs, info)` and then
        # fails an observation-space assertion.
        base_env = mate.make_environment(config=self.config_path, **overrides)

        base_env = mate.DiscreteCamera(base_env, levels=self.discrete_levels)

        factory = target_agent_factory or (lambda: mate.GreedyTargetAgent(seed=seed))
        self.target_agent = factory()
        env = mate.MultiCamera(base_env, target_agent=self.target_agent)

        env = mate.RepeatedRewardIndividualDone(env)
        if reward_coefficients:
            env = mate.AuxiliaryCameraRewards(
                env, coefficients=dict(reward_coefficients), reduction=reward_reduction
            )

        self.env = env
        self.unwrapped = env.unwrapped

        # ---- metadata: everything below is *read from MATE*, never declared ----
        self.n_agents: int = int(self.unwrapped.num_cameras)
        self.n_targets: int = int(self.unwrapped.num_targets)
        self.n_obstacles: int = int(self.unwrapped.num_obstacles)
        self.n_obs: int = int(self.unwrapped.camera_observation_space.shape[0])
        self.state_dim: int = int(self.unwrapped.state_space.shape[0])
        self.n_actions: int = self.discrete_levels**2
        self.discrete: bool = True
        self.max_time_steps: int = int(self.unwrapped.max_episode_steps)

        # DIMA's get_env_info() only reads `individual_action_space` for
        # continuous-action environments; kept for interface parity.
        self.individual_action_space = None

        self.camera_observation_space = self.unwrapped.camera_observation_space
        self.state_space = self.unwrapped.state_space
        self.camera_action_space = self.env.action_space[0]

        # The continuous action grid DiscreteCamera maps onto, in MATE units.
        # Used by the reactive-greedy planner to go continuous -> discrete and by
        # the action diagnostics to describe what each discrete index means.
        self.action_grid = (
            np.asarray(
                [self.unwrapped.camera_rotation_step, self.unwrapped.camera_zooming_step],
                dtype=np.float64,
            )
            * mate.DiscreteCamera.discrete_action_grid(levels=self.discrete_levels)
        )

        self._observation_kwargs = dict(
            team=Team.CAMERA,
            num_cameras=self.n_agents,
            num_targets=self.n_targets,
            num_obstacles=self.n_obstacles,
        )

        self.episode_step = 0
        self._last_infos: List[dict] = []
        self.seed(seed)

    # ------------------------------------------------------------------ meta --

    def metadata(self) -> Dict[str, Any]:
        """Everything the run manifest needs to describe this environment."""

        return {
            'scenario': self.scenario,
            'config_path': self.config_path,
            'name': str(self.unwrapped.name),
            'num_cameras': self.n_agents,
            'num_targets': self.n_targets,
            'num_obstacles': self.n_obstacles,
            'obs_dim': self.n_obs,
            'state_dim': self.state_dim,
            'num_actions': self.n_actions,
            'discrete_levels': self.discrete_levels,
            'max_episode_steps': self.max_time_steps,
            'reward': 'raw' if not self.reward_coefficients else dict(self.reward_coefficients),
            'target_agent': type(self.target_agent).__name__,
            'wrapper_stack': str(self.env),
        }

    def describe(self) -> str:
        """One-line ``[ENV]`` description, used by the console logger."""

        return (
            f'{self.scenario} cameras={self.n_agents} targets={self.n_targets} '
            f'obstacles={self.n_obstacles} obs_dim={self.n_obs} state_dim={self.state_dim} '
            f'action_space=Discrete({self.n_actions}) episode_limit={self.max_time_steps}'
        )

    # ------------------------------------------------------------- lifecycle --

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is not None:
            self._seed = int(seed)
        return self.env.seed(self._seed)

    def reset(self, seed: Optional[int] = None) -> MATEObservation:
        if seed is not None:
            self.seed(seed)
        joint_observation = self.env.reset()
        self.episode_step = 0
        self._last_infos = []
        return self._observe(np.asarray(joint_observation, dtype=np.float64), infos=[])

    def step(self, joint_action: Sequence[int]):
        """Apply one joint camera action.

        Returns ``(observation, reward, done, info)`` where ``reward`` is a
        ``(n_cameras,)`` array (MATE's camera-team reward, repeated per camera by
        ``RepeatedRewardIndividualDone``) and ``done`` is a scalar bool.
        """

        action = np.asarray(joint_action, dtype=np.int64).ravel()
        if action.shape != (self.n_agents,):
            raise ValueError(
                f'Expected a joint action of shape ({self.n_agents},), got {action.shape}.'
            )
        if not ((0 <= action) & (action < self.n_actions)).all():
            raise ValueError(
                f'Camera actions must lie in [0, {self.n_actions}), got {action.tolist()}.'
            )

        joint_observation, reward, done, infos = self.env.step(action)
        self.episode_step += 1
        self._last_infos = list(infos)

        reward = np.asarray(reward, dtype=np.float64).reshape(-1)
        if reward.size == 1:  # no RepeatedRewardIndividualDone in the stack
            reward = np.repeat(reward, self.n_agents)
        done_flag = bool(np.all(done))

        observation = self._observe(
            np.asarray(joint_observation, dtype=np.float64), infos=list(infos)
        )
        info = {
            'coverage_rate': float(infos[0]['coverage_rate']),
            'real_coverage_rate': float(infos[0]['real_coverage_rate']),
            'mean_transport_rate': float(infos[0]['mean_transport_rate']),
            'num_delivered_cargoes': int(infos[0]['num_delivered_cargoes']),
            'episode_step': self.episode_step,
            'truncated': done_flag and self.episode_step >= self.max_time_steps,
        }
        return observation, reward, done_flag, info

    def close(self) -> None:
        self.env.close()

    # --------------------------------------------------------------- forking --

    def fork(self) -> 'MATEEnv':
        """A deep copy of this environment, including its RNG state.

        Used by the oracle world model (brief section 28) to roll the *real*
        dynamics forward for a candidate action without disturbing the live
        episode.  MATE has no ``set_state``, so forking is the only way to get
        ground-truth one-step consequences.
        """

        return copy.deepcopy(self)

    # ------------------------------------------------------------ conversion --

    def _observe(self, joint_observation: np.ndarray, infos: List[dict]) -> MATEObservation:
        state_raw = np.asarray(self.unwrapped.state(), dtype=np.float64)
        return MATEObservation(
            obs_raw=joint_observation,
            obs=self.rescale_obs(joint_observation),
            state_raw=state_raw,
            state=self.rescale_state(state_raw),
            infos=infos,
            episode_step=self.episode_step,
        )

    def rescale_obs(self, observation: np.ndarray) -> np.ndarray:
        """Camera observation -> ``[-1, 1]`` using MATE's own rescaling.

        Identical to what :class:`mate.RescaledObservation` would produce; it is
        applied here instead of in the wrapper stack so that MATE's rule-based
        camera agents keep seeing MATE world units (see the module docstring).
        """

        return rescale_observation(np.asarray(observation, dtype=np.float64), **self._observation_kwargs)

    def unrescale_obs(self, rescaled: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`rescale_obs`: back from ``[-1, 1]`` to MATE world units."""

        return denormalize_observation(rescaled, self.camera_observation_space)

    def unrescale_state(self, rescaled: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`rescale_state`."""

        return denormalize_observation(rescaled, self.state_space)

    def rescale_state(self, state: np.ndarray) -> np.ndarray:
        """Global state -> ``[-1, 1]`` on the bounded dimensions.

        ``MultiAgentTracking.state_space`` is unbounded above on the preserved
        block and on the freight/bounty/cargo tail, so those entries are left in
        raw units; DIMA normalises the global state again with its own
        ``RunningMeanStd`` (``DreamerLearner.normalize_state``), which absorbs
        the remaining scale difference.
        """

        return normalize_observation(np.asarray(state, dtype=np.float64), self.state_space)

    def action_to_continuous(self, joint_action: Sequence[int]) -> np.ndarray:
        """Discrete camera indices -> the MATE ``(rotation, zoom)`` deltas they mean."""

        return self.action_grid[np.asarray(joint_action, dtype=np.int64).ravel()]

    def action_from_continuous(self, joint_action_continuous: np.ndarray) -> np.ndarray:
        """MATE ``(rotation, zoom)`` deltas -> nearest discrete camera indices.

        Delegates to :meth:`mate.DiscreteCamera.reverse_action`, so a continuous
        MATE agent (``GreedyCameraAgent``) can drive the discretised environment
        without reimplementing the projection.
        """

        discrete_env = self.env
        while not isinstance(discrete_env, mate.DiscreteCamera):
            discrete_env = discrete_env.env
        camera_actions, _ = discrete_env.reverse_action(
            (np.asarray(joint_action_continuous, dtype=np.float64), None)
        )
        return np.asarray(camera_actions, dtype=np.int64)
