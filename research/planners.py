"""Planners (brief sections 14, 15, 16).

A planner turns "what I see now" plus "what the world model thinks will happen"
into a joint camera action.  It never touches a DIMA tensor, a DIMA config, a
checkpoint, or a MATE wrapper at planning time -- :meth:`Planner.plan` receives
only a :class:`~research.env_adapter.MATEObservation`, a
:class:`~research.world_model.Prediction` and a :class:`PlanContext`.

Constructors are allowed to know more than :meth:`plan` does.  :class:`MATEAgentPlanner`
*is* a group of MATE ``CameraAgentBase`` agents, so it necessarily holds the
environment; :class:`ModelBasedGreedyPlanner` holds a world model.  That coupling
is at construction, which the run script does once, not in the planning loop that
the research questions live in.

Selection is configuration-driven (brief section 16) through :func:`build_planner`,
which resolves either a short registry name or a ``module:Attribute`` entry point.
The entry-point form reuses MATE's own ``mate.evaluate.load_entry`` -- the same
mechanism ``python -m mate.evaluate --camera-agent`` uses -- so a new planner is
selectable without editing anything here.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

import mate
from mate.wrappers.single_team import group_reset, group_step

try:
    from mate.evaluate import load_entry
except ImportError:  # pragma: no cover - setuptools >= 81 dropped pkg_resources
    # mate/evaluate.py imports pkg_resources at module level purely to compare
    # gym versions for video recording.  The research layer only needs its
    # entry-point resolver, so fall back to the same five lines
    # (mate/evaluate.py:76-82) rather than requiring setuptools < 81.
    import importlib

    def load_entry(entry_point: str):
        """Load a module attribute from an ``module:Attribute`` entry point."""

        mod_name, attr_name = entry_point.split(':')
        return getattr(importlib.import_module(mod_name), attr_name)

from research.env_adapter import MATEEnv, MATEObservation
from research.views import ObservationLayout, SceneView
from research.world_model import History, Prediction, WorldModel


__all__ = [
    'PlanContext',
    'PrivilegedView',
    'Planner',
    'MATEAgentPlanner',
    'RandomPlanner',
    'MATERandomPlanner',
    'ReactiveGreedyPlanner',
    'ModelBasedGreedyPlanner',
    'PLANNER_REGISTRY',
    'build_planner',
]


# --------------------------------------------------------------------------- #
# What a planner is given
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PrivilegedView:
    """Ground-truth access, handed only to planners that declare they need it.

    Brief section 9 asks for privileged information to be impossible to use by
    accident.  ``PlanContext.privileged`` is ``None`` unless the planner class
    sets ``USES_PRIVILEGED_STATE = True``, so a planner that claims to be
    decentralised cannot silently read the global state.
    """

    state: np.ndarray
    state_raw: np.ndarray


@dataclass(frozen=True)
class PlanContext:
    """Everything a planner may know beyond the current observation."""

    step: int
    episode: int
    num_cameras: int
    num_targets: int
    num_actions: int
    env_metadata: Dict[str, Any]
    layout: ObservationLayout
    last_info: Dict[str, Any] = field(default_factory=dict)
    history: Optional[History] = None
    privileged: Optional[PrivilegedView] = None


class Planner(abc.ABC):
    """Base class for every planner in the baseline matrix."""

    #: The runner only computes a prediction for planners that want one.
    USES_WORLD_MODEL: bool = False
    #: Gates ``PlanContext.privileged``.
    USES_PRIVILEGED_STATE: bool = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._diagnostics: Dict[str, Any] = {}

    def reset(self, observation: MATEObservation, context: PlanContext) -> None:  # noqa: B027
        """Called once per episode, before the first :meth:`plan`."""

    @abc.abstractmethod
    def plan(
        self,
        observation: MATEObservation,
        prediction: Optional[Prediction],
        context: PlanContext,
    ) -> np.ndarray:
        """Return a ``(num_cameras,)`` array of discrete camera actions."""

    def diagnostics(self) -> Dict[str, Any]:
        """Per-step planner diagnostics for the last :meth:`plan` (brief section 27)."""

        return dict(self._diagnostics)

    def describe(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'class': type(self).__name__,
            'uses_world_model': self.USES_WORLD_MODEL,
            'uses_privileged_state': self.USES_PRIVILEGED_STATE,
        }


# --------------------------------------------------------------------------- #
# Baselines that reuse MATE's own agents verbatim
# --------------------------------------------------------------------------- #


class MATEAgentPlanner(Planner):
    """Runs a group of MATE ``CameraAgentBase`` agents as a planner.

    This is how brief section 15's "match the existing MATE GreedyCameraAgent
    behaviour as closely as practical" is satisfied: the behaviour is not matched,
    it *is* ``GreedyCameraAgent``.  The full MATE agent protocol runs --
    ``observe``, request/response messaging, ``act`` -- via MATE's own
    :func:`mate.group_step`, so inter-camera communication behaves exactly as it
    does under ``python -m mate.evaluate``.

    MATE's rule-based agents always emit *continuous* ``(rotation, zoom)``
    actions, because ``AgentBase.reset`` builds their action space from
    ``CameraStatePrivate.action_space`` regardless of any ``DiscreteCamera``
    wrapper.  Projection onto the discrete grid is delegated to
    ``DiscreteCamera.reverse_action`` through
    :meth:`~research.env_adapter.MATEEnv.action_from_continuous`, so the
    discretisation is MATE's, not ours.
    """

    def __init__(
        self,
        env: MATEEnv,
        agent_class: Type[Any] = mate.GreedyCameraAgent,
        *,
        seed: int = 0,
        name: Optional[str] = None,
        agent_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name or agent_class.__name__)
        if not issubclass(agent_class, mate.CameraAgentBase):
            raise TypeError(f'{agent_class!r} is not a mate.CameraAgentBase subclass.')

        self._env = env
        self._prototype = agent_class(seed=seed, **(agent_kwargs or {}))
        self._agents: List[Any] = []
        self._last_continuous: Optional[np.ndarray] = None

    def reset(self, observation: MATEObservation, context: PlanContext) -> None:
        self._agents = self._prototype.spawn(context.num_cameras)
        group_reset(self._agents, observation.obs_raw)

    def plan(self, observation, prediction, context) -> np.ndarray:
        infos = observation.infos or None
        continuous = group_step(self._env.env, self._agents, observation.obs_raw, infos)
        continuous = np.asarray(continuous, dtype=np.float64)
        discrete = self._env.action_from_continuous(continuous)

        self._last_continuous = continuous
        self._diagnostics = {
            'action_continuous': continuous.tolist(),
            'discretisation_error': float(
                np.abs(self._env.action_to_continuous(discrete) - continuous).max()
            ),
        }
        return discrete


class MATERandomPlanner(MATEAgentPlanner):
    """MATE's own ``RandomCameraAgent`` (continuous sample, held for 20 steps)."""

    def __init__(self, env: MATEEnv, *, seed: int = 0, **kwargs) -> None:
        super().__init__(env, mate.RandomCameraAgent, seed=seed, name='mate_random', **kwargs)


class ReactiveGreedyPlanner(MATEAgentPlanner):
    """MATE's ``GreedyCameraAgent``: track the nearest remembered target.

    The non-world-model baseline of brief section 15, and the adapter's own
    validation test -- if this scores differently here than under
    ``python -m mate.evaluate``, the adapter is wrong, not the planner.
    """

    def __init__(self, env: MATEEnv, *, seed: int = 0, **kwargs) -> None:
        super().__init__(env, mate.GreedyCameraAgent, seed=seed, name='reactive_greedy', **kwargs)


class NaivePlanner(MATEAgentPlanner):
    """MATE's ``NaiveCameraAgent``: rotate anti-clockwise at maximum viewing angle."""

    def __init__(self, env: MATEEnv, *, seed: int = 0, **kwargs) -> None:
        super().__init__(env, mate.NaiveCameraAgent, seed=seed, name='naive', **kwargs)


class HeuristicPlanner(MATEAgentPlanner):
    """MATE's ``HeuristicCameraAgent``: the strongest rule-based camera team MATE ships."""

    def __init__(self, env: MATEEnv, *, seed: int = 0, **kwargs) -> None:
        super().__init__(env, mate.HeuristicCameraAgent, seed=seed, name='heuristic', **kwargs)


class RandomPlanner(Planner):
    """Uniform over the discretised camera action set.

    The clean lower bound for *this* action space.  ``mate_random`` (MATE's
    ``RandomCameraAgent``) samples the continuous box and then projects, which is
    not uniform over the grid -- both are provided so the comparison is explicit.
    """

    def __init__(self, env: MATEEnv, *, seed: int = 0, name: str = 'random') -> None:
        super().__init__(name)
        self._rng = np.random.default_rng(seed)
        self._num_actions = env.n_actions

    def plan(self, observation, prediction, context) -> np.ndarray:
        action = self._rng.integers(0, self._num_actions, size=context.num_cameras)
        self._diagnostics = {}
        return action.astype(np.int64)


# --------------------------------------------------------------------------- #
# The model-based planner
# --------------------------------------------------------------------------- #


def _coverage_utility(view: SceneView) -> float:
    return view.coverage_estimate()


def _soft_coverage_utility(view: SceneView) -> float:
    return view.soft_coverage_estimate()


UTILITIES: Dict[str, Callable[[SceneView], float]] = {
    'coverage': _coverage_utility,
    'soft_coverage': _soft_coverage_utility,
}


class ModelBasedGreedyPlanner(Planner):
    """One coordinate-descent sweep over cameras, scored by a world model.

    This single class is brief section 15's *Predictive Greedy* and *Oracle
    Planner*: the only difference between them is which
    :class:`~research.world_model.WorldModel` it is constructed with, which is
    exactly what brief section 28 requires ("the same planner must be usable with
    the DIMA world model and the oracle world model without modification").

    Search
    ------
    Evaluating all ``num_actions ** num_cameras`` joint actions is 390 625 for
    MATE-4v2-9 at ``levels=5``.  Instead the planner starts from a base joint
    action and, camera by camera, asks the world model for the consequence of
    each of that camera's actions while the others are held fixed -- ``C * A``
    queries per sweep, batched into one world-model call per camera.  The base
    action is the previous step's choice, so a sweep never scores worse than the
    status quo under the model.

    Utility
    -------
    The world model returns predicted *observations* in MATE units.  Those are
    turned into a :class:`~research.views.SceneView` and scored with
    ``soft_coverage`` by default (see
    :meth:`~research.views.SceneView.soft_coverage_estimate`).  ``coverage``
    (hard) and any callable are also accepted.  Predicted reward is recorded as a
    diagnostic but is not the default utility: DIMA's reward head is trained on
    MATE's raw team reward, which is zero on most steps and therefore a poor
    action discriminator early in training.
    """

    USES_WORLD_MODEL = True

    def __init__(
        self,
        env: MATEEnv,
        world_model: WorldModel,
        *,
        horizon: int = 1,
        sweeps: int = 1,
        utility: str = 'soft_coverage',
        discount: float = 1.0,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f'model_based_greedy({world_model.name})')
        self.world_model = world_model
        self.horizon = max(1, int(horizon))
        self.sweeps = max(1, int(sweeps))
        self.discount = float(discount)
        self.utility_name = utility if isinstance(utility, str) else getattr(utility, '__name__', 'custom')
        self._utility = UTILITIES[utility] if isinstance(utility, str) else utility

        self.USES_PRIVILEGED_STATE = world_model.uses_privileged_state

        self._num_actions = env.n_actions
        self._num_cameras = env.n_agents
        self._rng = np.random.default_rng(seed)
        self._base_action = np.full(self._num_cameras, env.discrete_levels**2 // 2, dtype=np.int64)

    def reset(self, observation: MATEObservation, context: PlanContext) -> None:
        self._base_action = np.full(context.num_cameras, self._num_actions // 2, dtype=np.int64)

    def plan(self, observation, prediction, context) -> np.ndarray:
        if context.history is None:
            raise ValueError('ModelBasedGreedyPlanner needs PlanContext.history.')

        joint = self._base_action.copy()
        scored: Dict[int, np.ndarray] = {}
        base_utility = None

        for _ in range(self.sweeps):
            for camera in range(context.num_cameras):
                candidates = np.tile(joint, (self._num_actions, 1))
                candidates[:, camera] = np.arange(self._num_actions)

                pred = self.world_model.predict(
                    context.history, candidates, horizon=self.horizon
                )
                utilities = self._score(pred, context)
                scored[camera] = utilities

                best = int(np.argmax(utilities))
                if base_utility is None:
                    base_utility = float(utilities[joint[camera]])
                joint[camera] = best

        self._base_action = joint.copy()

        per_camera_spread = {
            f'camera_{c}': float(np.ptp(u)) for c, u in scored.items()
        }
        final_utilities = scored.get(context.num_cameras - 1)
        self._diagnostics = {
            'predicted_utility': float(final_utilities.max()) if final_utilities is not None else float('nan'),
            'base_utility': base_utility if base_utility is not None else float('nan'),
            'utility_spread_per_camera': per_camera_spread,
            'utility': self.utility_name,
            'horizon': self.horizon,
        }
        return joint

    def _score(self, prediction: Prediction, context: PlanContext) -> np.ndarray:
        """Discounted sum of per-step utility over the prediction horizon."""

        num_candidates, horizon = prediction.observations.shape[:2]
        utilities = np.zeros(num_candidates, dtype=np.float64)
        for b in range(num_candidates):
            total = 0.0
            weight = 1.0
            for h in range(horizon):
                view = SceneView.from_joint_observation(
                    prediction.observations[b, h], context.layout
                )
                total += weight * self._utility(view)
                weight *= self.discount
            utilities[b] = total
        return utilities


# --------------------------------------------------------------------------- #
# Registry (brief section 16)
# --------------------------------------------------------------------------- #

#: Short names accepted by ``--planner``.  Anything not listed here is treated as
#: a ``module:Attribute`` entry point and resolved with MATE's ``load_entry``, so
#: new planners never require an edit to this table.
PLANNER_REGISTRY: Dict[str, Type[Planner]] = {
    'random': RandomPlanner,
    'mate_random': MATERandomPlanner,
    'naive': NaivePlanner,
    'reactive_greedy': ReactiveGreedyPlanner,
    'heuristic': HeuristicPlanner,
    'predictive_greedy': ModelBasedGreedyPlanner,
    'oracle': ModelBasedGreedyPlanner,
}

#: Planners that must be handed a world model.
NEEDS_WORLD_MODEL = frozenset({'predictive_greedy', 'oracle'})


def build_planner(
    spec: str,
    env: MATEEnv,
    *,
    world_model: Optional[WorldModel] = None,
    seed: int = 0,
    **kwargs: Any,
) -> Planner:
    """Resolve ``spec`` to a planner instance.

    ``spec`` is either a key of :data:`PLANNER_REGISTRY` or a
    ``module:Attribute`` entry point naming a :class:`Planner` subclass.
    """

    if spec in PLANNER_REGISTRY:
        cls = PLANNER_REGISTRY[spec]
        needs_wm = spec in NEEDS_WORLD_MODEL
    else:
        cls = load_entry(spec)
        if not (isinstance(cls, type) and issubclass(cls, Planner)):
            raise TypeError(f'Entry point {spec!r} does not name a research.planners.Planner.')
        needs_wm = cls.USES_WORLD_MODEL

    if needs_wm:
        if world_model is None:
            raise ValueError(f'Planner {spec!r} needs a world model; none was provided.')
        planner = cls(env, world_model, seed=seed, **kwargs)
        if spec in PLANNER_REGISTRY:
            planner.name = spec
        return planner

    return cls(env, seed=seed, **kwargs)
