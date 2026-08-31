"""The closed loop, and the trajectories it produces (brief sections 17, 19).

One function, :func:`run_episode`, is used for everything: collecting training
data with a heuristic policy, evaluating a baseline, and running DIMA in the
loop.  That is deliberate -- if data collection and evaluation used different
loops, a discrepancy between them would be invisible.

Trajectories are converted to DIMA's format by :func:`to_dima_rollout`, which
produces exactly the dictionary ``DreamerLearner.step`` consumes.  No new
episode or dataset class is introduced: DIMA's ``MamujocoEpisode`` and
``MultiAgentEpisodesDataset`` already hold what MATE produces (brief section 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.env_adapter import MATEEnv, MATEObservation
from research.planners import PlanContext, Planner, PrivilegedView
from research.views import ObservationLayout
from research.world_model import History, Prediction, WorldModel


__all__ = ['Transition', 'EpisodeResult', 'run_episode', 'to_dima_rollout']


@dataclass
class Transition:
    """One environment step, in the terms both DIMA and the diagnostics need."""

    obs: np.ndarray  # (C, obs_dim) rescaled -- decentralised
    obs_raw: np.ndarray  # (C, obs_dim) MATE units -- decentralised
    state: np.ndarray  # (state_dim,) rescaled -- PRIVILEGED
    action: np.ndarray  # (C,) discrete
    reward: np.ndarray  # (C,)
    next_obs: np.ndarray
    next_obs_raw: np.ndarray
    next_state: np.ndarray
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    transitions: List[Transition]
    metrics: Dict[str, float]
    planner_diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    predictions: List[Optional[Prediction]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.transitions)


def run_episode(
    env: MATEEnv,
    planner: Planner,
    *,
    world_model: Optional[WorldModel] = None,
    episode: int = 0,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
    keep_predictions: bool = False,
    reference_prediction: bool = True,
    on_step=None,
) -> EpisodeResult:
    """Run one closed-loop episode (brief section 19).

    ``reference_prediction`` computes one world-model prediction per step for the
    planner's *previous* action.  It is what gets handed to ``planner.plan`` and
    what the prediction-versus-actual diagnostics compare against; search
    planners issue their own extra queries on top of it.
    """

    layout = ObservationLayout.from_env_metadata(env.metadata())
    conditioning = world_model.conditioning_steps if world_model is not None else 1
    history = History(length=max(1, conditioning))

    noop = getattr(world_model, 'noop_action', env.discrete_levels**2 // 2)

    observation = env.reset(seed=seed)
    history.seed(observation, env.n_agents, noop)

    context = _make_context(env, planner, layout, observation, history, episode=episode, step=0)
    planner.reset(observation, context)
    if world_model is not None:
        world_model.reset(observation)

    limit = max_steps if max_steps is not None else env.max_time_steps
    transitions: List[Transition] = []
    diagnostics: List[Dict[str, Any]] = []
    predictions: List[Optional[Prediction]] = []

    coverage_rates: List[float] = []
    reward_total = 0.0
    last_action = np.full(env.n_agents, noop, dtype=np.int64)

    for step in range(limit):
        context = _make_context(
            env, planner, layout, observation, history, episode=episode, step=step
        )

        prediction: Optional[Prediction] = None
        if world_model is not None and reference_prediction:
            prediction = world_model.predict(history, last_action, horizon=1)

        action = np.asarray(planner.plan(observation, prediction, context), dtype=np.int64).ravel()
        next_observation, reward, done, info = env.step(action)

        history.push(next_observation, action)
        if world_model is not None:
            world_model.observe(next_observation, action)

        transitions.append(
            Transition(
                obs=observation.obs,
                obs_raw=observation.obs_raw,
                state=observation.state,
                action=action,
                reward=reward,
                next_obs=next_observation.obs,
                next_obs_raw=next_observation.obs_raw,
                next_state=next_observation.state,
                done=done,
                info=info,
            )
        )
        diagnostics.append(planner.diagnostics())
        predictions.append(prediction if keep_predictions else None)

        coverage_rates.append(info['coverage_rate'])
        reward_total += float(np.mean(reward))
        last_action = action

        if on_step is not None:
            on_step(step, transitions[-1], prediction, diagnostics[-1])

        observation = next_observation
        if done:
            break

    unwrapped = env.unwrapped
    metrics = {
        'episode_length': float(len(transitions)),
        'camera_team_return': reward_total,
        'mean_coverage_rate': float(np.mean(coverage_rates)) if coverage_rates else 0.0,
        'final_coverage_rate': float(coverage_rates[-1]) if coverage_rates else 0.0,
        'mean_transport_rate': float(transitions[-1].info['mean_transport_rate'])
        if transitions
        else 0.0,
        'num_delivered_cargoes': float(transitions[-1].info['num_delivered_cargoes'])
        if transitions
        else 0.0,
        # `mate.evaluate`'s headline number, computed the same way (evaluate.py:138).
        'normalized_target_episode_reward': float(
            unwrapped.target_team_episode_reward / unwrapped.max_target_team_episode_reward
        ),
    }

    return EpisodeResult(
        transitions=transitions,
        metrics=metrics,
        planner_diagnostics=diagnostics,
        predictions=predictions,
    )


def _make_context(
    env: MATEEnv,
    planner: Planner,
    layout: ObservationLayout,
    observation: MATEObservation,
    history: History,
    *,
    episode: int,
    step: int,
) -> PlanContext:
    privileged = None
    if planner.USES_PRIVILEGED_STATE:
        privileged = PrivilegedView(state=observation.state, state_raw=observation.state_raw)

    return PlanContext(
        step=step,
        episode=episode,
        num_cameras=env.n_agents,
        num_targets=env.n_targets,
        num_actions=env.n_actions,
        env_metadata=env.metadata(),
        layout=layout,
        last_info=observation.infos[0] if observation.infos else {},
        history=history,
        privileged=privileged,
    )


def to_dima_rollout(result: EpisodeResult, num_actions: int) -> Dict[str, np.ndarray]:
    """Convert an episode to the dictionary ``DreamerLearner.step`` consumes.

    The keys and shapes are those produced by DIMA's own
    ``DreamerController.dispatch_buffer`` (controllers/DreamerController.py:87-94)
    from what ``DreamerWorker.run`` buffers (workers/DreamerWorker.py:172-181).
    Every array is ``(T, num_cameras, ...)``:

    ``observation``       ``(T, C, obs_dim)``    -- rescaled local observations
    ``shared_obs``        ``(T, C, state_dim)``  -- global state, repeated per camera
    ``next_shared_obs``   ``(T, C, state_dim)``
    ``action``            ``(T, C, num_actions)``-- one-hot
    ``reward``            ``(T, C, 1)``
    ``done``              ``(T, C, 1)``
    ``fake``              ``(T, C, 1)``          -- absorbing-state flag, unused for MATE
    ``last``              ``(T, C, 1)``          -- 1 on the final step
    ``entropy``           ``(T, C)``             -- logging only

    ``shared_obs`` is repeated across the camera axis because that is how DIMA
    stores it; the learner recovers the single global state with ``.mean(2)``
    (DreamerLearner.py:321, 374, 423).
    """

    if not result.transitions:
        raise ValueError('Cannot convert an empty episode.')

    steps = len(result.transitions)
    num_agents = result.transitions[0].obs.shape[0]

    observation = np.stack([t.obs for t in result.transitions]).astype(np.float32)
    next_observation = np.stack([t.next_obs for t in result.transitions]).astype(np.float32)
    state = np.stack([t.state for t in result.transitions]).astype(np.float32)
    next_state = np.stack([t.next_state for t in result.transitions]).astype(np.float32)
    actions = np.stack([t.action for t in result.transitions]).astype(np.int64)
    reward = np.stack([t.reward for t in result.transitions]).astype(np.float32)
    done = np.array([t.done for t in result.transitions], dtype=np.float32)

    shared_obs = np.repeat(state[:, None, :], num_agents, axis=1)
    next_shared_obs = np.repeat(next_state[:, None, :], num_agents, axis=1)

    one_hot = np.zeros((steps, num_agents, num_actions), dtype=np.float32)
    np.put_along_axis(one_hot, actions[..., None], 1.0, axis=2)

    last = np.zeros((steps, num_agents, 1), dtype=np.float32)
    last[-1] = 1.0

    return {
        'observation': observation,
        'next_observation': next_observation,
        'shared_obs': shared_obs,
        'next_shared_obs': next_shared_obs,
        'action': one_hot,
        'reward': reward[..., None],
        'done': np.repeat(done[:, None, None], num_agents, axis=1),
        'fake': np.zeros((steps, num_agents, 1), dtype=np.float32),
        'last': last,
        'entropy': np.zeros((steps, num_agents), dtype=np.float32),
    }
