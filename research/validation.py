"""Held-out world-model validation, for watching training move (or not).

Training loss going down is a weak signal: DIMA's denoising loss can fall while
the model learns an action-*independent* prior, which is exactly the failure this
project measures. What has to improve, and improve *stably*, is:

* ``ade`` / ``mae`` at H=1 -- is the prediction getting closer to the truth;
* ``sensitivity_ratio`` -- does changing the action move the prediction more than
  re-sampling it does. Below 1.0 no planner can use the model at all.

:func:`validate` computes both on episodes the learner has never seen, directly
from the live learner, and is cheap enough to run every few training rounds.

Kept separate from ``diagnostics.py`` because that module reports on a *finished*
model, while this one runs inside the training loop and has to be fast, quiet and
side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.diagnostics import compute_action_sensitivity, compute_prediction_error
from research.env_adapter import MATEEnv
from research.views import ObservationLayout
from research.world_model import History


__all__ = ['ValidationEpisode', 'validate', 'format_trend']


@dataclass
class _Step:
    state: np.ndarray
    obs: np.ndarray
    action: np.ndarray
    next_obs_raw: np.ndarray


@dataclass
class ValidationEpisode:
    """A held-out rollout in the shape the diagnostics expect.

    Built straight from a collected ``.npz`` rollout, so validation data is the
    same data format training consumes -- there is no second pipeline to keep in
    sync.
    """

    transitions: List[_Step] = field(default_factory=list)

    @classmethod
    def from_rollout(cls, rollout: Dict[str, np.ndarray], env: MATEEnv) -> 'ValidationEpisode':
        """Build from one collected ``.npz`` rollout.

        The stored arrays hold ``observation[t]``; the successor is simply
        ``observation[t+1]``, so ``next_obs`` is derived rather than stored — the
        dataset stays exactly the set of arrays ``DreamerLearner.step`` consumes
        and does not double in size. The final step has no successor and is
        dropped, which is why the returned episode is one transition shorter.

        ``observation`` is rescaled to ``[-1, 1]``; the truth a prediction is
        compared against has to be in MATE world units, because that is what
        :class:`~research.world_model.Prediction` carries.
        """

        observation = rollout['observation'].astype(np.float64)
        actions = rollout['action'].argmax(-1).astype(np.int64)
        # shared_obs repeats the single global state across the camera axis.
        states = rollout['shared_obs'][:, 0, :].astype(np.float64)

        length = len(actions) - 1
        if length <= 0:
            return cls(transitions=[])

        next_obs_raw = env.unrescale_obs(observation[1 : length + 1])
        steps = [
            _Step(
                state=states[t],
                obs=observation[t],
                action=actions[t],
                next_obs_raw=next_obs_raw[t],
            )
            for t in range(length)
        ]
        return cls(transitions=steps)


def _history_at(episode: ValidationEpisode, t: int, conditioning: int) -> History:
    """Conditioning window ending at ``t``, following ``History``'s alignment.

    ``actions[i]`` is the action that produced ``states[i]``, so the action stored
    alongside step ``k`` is ``transitions[k-1].action`` -- see
    ``research/world_model.py:History``.
    """

    history = History(length=conditioning)
    for k in range(t - conditioning + 1, t + 1):
        history.states.append(episode.transitions[k].state)
        history.observations.append(episode.transitions[k].obs)
        history.actions.append(
            episode.transitions[k - 1].action if k > 0 else episode.transitions[k].action
        )
    return history


def validate(
    world_model,
    episodes: Sequence[ValidationEpisode],
    layout: ObservationLayout,
    *,
    num_agents: int,
    num_actions: int,
    horizons: Sequence[int] = (1,),
    states_per_episode: int = 4,
    sensitivity_samples: int = 4,
    sensitivity_actions: int = 9,
    sensitivity_states: int = 3,
    seed: int = 0,
) -> Dict[str, Any]:
    """Prediction error and action sensitivity on held-out data.

    Reproducibility is the whole point here. ``seed`` is deliberately *not* varied
    per call: every validation must score the **same** held-out states, or
    pass-to-pass differences measure which states got sampled rather than whether
    the model improved. Measured on a short CPU run, resampling the states made
    ADE wander by ±50 while the model barely moved.

    ``sensitivity_actions`` sub-samples the action set (evenly spaced) because the
    full sweep costs ``num_cameras * num_actions * sensitivity_samples`` diffusion
    samples; the ratio is stable under sub-sampling. ``sensitivity_states``
    averages the ratio over several states, since a single state is a noisy
    estimate of it.
    """

    from research.diagnostics import horizon_error_report

    if hasattr(world_model, 'eval_mode'):
        world_model.eval_mode()

    errors = horizon_error_report(
        world_model,
        episodes,
        layout,
        horizons=horizons,
        max_states_per_episode=states_per_episode,
        rng=np.random.default_rng(seed),
    )

    conditioning = max(1, world_model.conditioning_steps)
    usable = [ep for ep in episodes if len(ep.transitions) > conditioning + 2]
    subset = np.unique(
        np.linspace(0, num_actions - 1, min(sensitivity_actions, num_actions)).astype(int)
    )

    # A fresh generator, seeded identically on every call, so the sensitivity
    # states are also fixed across passes.
    rng = np.random.default_rng(seed + 1)
    measured: List[Any] = []
    for _ in range(max(1, sensitivity_states)):
        if not usable:
            break
        episode = usable[int(rng.integers(len(usable)))]
        t = int(rng.integers(conditioning - 1, len(episode.transitions) - 1))
        measured.append(
            compute_action_sensitivity(
                world_model,
                _history_at(episode, t, conditioning),
                episode.transitions[t].action,
                num_cameras=num_agents,
                num_actions=num_actions,
                num_samples=sensitivity_samples,
                action_subset=subset,
            )
        )

    row: Dict[str, Any] = {}
    for horizon, stat in errors.items():
        row[f'mae_h{horizon}'] = stat.mae
        row[f'rmse_h{horizon}'] = stat.rmse
        row[f'ade_h{horizon}'] = stat.ade
        row[f'coverage_mae_h{horizon}'] = stat.coverage_mae
        row[f'sighting_recall_h{horizon}'] = stat.sighting_recall

    if measured:
        row['sensitivity_states'] = len(measured)
        for field_name in ('between', 'within', 'ratio', 'asymmetry'):
            values = [
                getattr(s, field_name)
                for s in measured
                if getattr(s, field_name) is not None and np.isfinite(getattr(s, field_name))
            ]
            row[f'sensitivity_{field_name}'] = float(np.mean(values)) if values else None
        if row['sensitivity_ratio'] is None and any(
            s.ratio == float('inf') for s in measured
        ):
            # Every state was deterministic -- see diagnostics._sensitivity_ratio.
            row['sensitivity_ratio'] = float('inf')
    return row


_ARROWS = {1: '↑', -1: '↓', 0: '→'}


def format_trend(history: Sequence[Dict[str, Any]], keys: Sequence[str]) -> str:
    """One line summarising where each tracked metric stands and which way it moved.

    ``ade`` should fall and ``sensitivity_ratio`` should rise; the arrow says
    which way the last step went so a stalled or oscillating run is visible in
    the console without opening TensorBoard.
    """

    if not history:
        return 'no validation yet'

    latest = history[-1]
    previous = history[-2] if len(history) > 1 else None
    parts: List[str] = []
    for key in keys:
        value = latest.get(key)
        if value is None or not np.isfinite(value):
            parts.append(f'{key}=N/A')
            continue
        arrow = ''
        if previous is not None:
            before = previous.get(key)
            if before is not None and np.isfinite(before):
                delta = value - before
                scale = max(abs(before), 1e-9)
                direction = 0 if abs(delta) / scale < 1e-3 else (1 if delta > 0 else -1)
                arrow = _ARROWS[direction]
        parts.append(f'{key}={value:.4f}{arrow}')
    return '  '.join(parts)
