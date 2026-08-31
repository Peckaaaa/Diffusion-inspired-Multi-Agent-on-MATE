"""World-model and planner diagnostics (brief sections 22-27, 35).

The research hierarchy this module exists to keep separate (brief section 35)::

    1. prediction accuracy        <- compute_prediction_error / horizon_error_report
    2. action sensitivity         <- compute_action_sensitivity
    3. action ranking quality     <- compute_action_ranking
    4. planner performance        <- PlannerAccumulator
    5. final coverage             <- reported by MATE, never by us

Low prediction error does not imply good planning and a better world model does
not imply better coordination.  Nothing here aggregates these into a single
score, precisely so that the hypotheses stay testable.

Every metric that cannot be computed is reported as ``N/A`` rather than dropped
(brief section 22).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.views import ObservationLayout, SceneView
from research.world_model import History, Prediction, WorldModel


__all__ = [
    'PredictionErrorStats',
    'ActionSensitivityStats',
    'ActionRankingStats',
    'compute_prediction_error',
    'horizon_error_report',
    'compute_action_sensitivity',
    'compute_action_ranking',
    'prediction_validity',
    'format_world_model_report',
    'PlannerAccumulator',
]


_NA = 'N/A'


def _fmt(value: Optional[float], spec: str = '.4f') -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return _NA
    return format(value, spec)


# --------------------------------------------------------------------------- #
# 1. Prediction accuracy (brief section 22)
# --------------------------------------------------------------------------- #


@dataclass
class PredictionErrorStats:
    """Errors between predicted and actual observations.

    ``mae`` / ``rmse`` are over the whole observation vector in MATE world units,
    so they are dominated by the largest-magnitude fields (coordinates).  ``ade``
    / ``fde`` are average and final *displacement* errors of the target
    positions, in MATE distance units -- the quantity a tracking planner actually
    cares about, and the reason the raw MAE alone is not enough.
    """

    horizon: int
    count: int
    mae: Optional[float] = None
    rmse: Optional[float] = None
    ade: Optional[float] = None
    fde: Optional[float] = None
    coverage_mae: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'horizon': self.horizon,
            'count': self.count,
            'mae': self.mae,
            'rmse': self.rmse,
            'ade': self.ade,
            'fde': self.fde,
            'coverage_mae': self.coverage_mae,
        }


def compute_prediction_error(
    predicted: np.ndarray,
    actual: np.ndarray,
    layout: ObservationLayout,
) -> PredictionErrorStats:
    """Compare ``(N, H, C, obs_dim)`` predictions against the same-shaped truth."""

    predicted = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if predicted.shape != actual.shape:
        raise ValueError(f'Shape mismatch: predicted {predicted.shape}, actual {actual.shape}.')

    count, horizon = predicted.shape[:2]
    if count == 0:
        return PredictionErrorStats(horizon=horizon, count=0)

    residual = predicted - actual
    mae = float(np.abs(residual).mean())
    rmse = float(np.sqrt(np.square(residual).mean()))

    # Displacement errors are computed on the decoded target positions, using
    # only targets the ground truth actually sighted -- unsighted targets carry a
    # zeroed placeholder position and would otherwise dominate the average.
    displacements: List[np.ndarray] = []
    final_displacements: List[np.ndarray] = []
    coverage_errors: List[float] = []

    for n in range(count):
        per_step: List[float] = []
        for h in range(horizon):
            pred_view = SceneView.from_joint_observation(predicted[n, h], layout)
            true_view = SceneView.from_joint_observation(actual[n, h], layout)
            visible = true_view.target_sighted
            if visible.any():
                delta = pred_view.target_positions[visible] - true_view.target_positions[visible]
                per_step.append(float(np.linalg.norm(delta, axis=-1).mean()))
            coverage_errors.append(
                abs(pred_view.coverage_estimate() - true_view.coverage_estimate())
            )
        if per_step:
            displacements.append(np.mean(per_step))
            final_displacements.append(per_step[-1])

    return PredictionErrorStats(
        horizon=horizon,
        count=count,
        mae=mae,
        rmse=rmse,
        ade=float(np.mean(displacements)) if displacements else None,
        fde=float(np.mean(final_displacements)) if final_displacements else None,
        coverage_mae=float(np.mean(coverage_errors)) if coverage_errors else None,
    )


def horizon_error_report(
    world_model: WorldModel,
    episodes: Sequence[Any],
    layout: ObservationLayout,
    horizons: Sequence[int] = (1, 3, 5, 10),
    *,
    max_states_per_episode: int = 8,
    rng: Optional[np.random.Generator] = None,
) -> Dict[int, PredictionErrorStats]:
    """Roll the model forward from logged states and score it at each horizon.

    Each sampled timestep replays the *actual* action sequence that followed it,
    so this measures open-loop prediction quality, not planning quality.
    """

    rng = rng or np.random.default_rng(0)
    conditioning = max(1, world_model.conditioning_steps)
    results: Dict[int, PredictionErrorStats] = {}

    for horizon in horizons:
        predicted_batches: List[np.ndarray] = []
        actual_batches: List[np.ndarray] = []

        for episode in episodes:
            steps = len(episode.transitions)
            first = conditioning - 1
            last = steps - horizon
            if last <= first:
                continue
            choices = rng.choice(
                np.arange(first, last), size=min(max_states_per_episode, last - first), replace=False
            )
            for t in sorted(int(c) for c in choices):
                history = History(length=conditioning)
                for k in range(t - conditioning + 1, t + 1):
                    history.states.append(episode.transitions[k].state)
                    history.observations.append(episode.transitions[k].obs)
                    history.actions.append(
                        episode.transitions[k - 1].action
                        if k > 0
                        else np.zeros_like(episode.transitions[k].action)
                    )

                action_seq = np.stack(
                    [episode.transitions[t + h].action for h in range(horizon)]
                )[None]
                prediction = world_model.predict(history, action_seq, horizon=horizon)
                truth = np.stack(
                    [episode.transitions[t + h].next_obs_raw for h in range(horizon)]
                )[None]

                predicted_batches.append(prediction.observations)
                actual_batches.append(truth)

        if predicted_batches:
            results[horizon] = compute_prediction_error(
                np.concatenate(predicted_batches), np.concatenate(actual_batches), layout
            )
        else:
            results[horizon] = PredictionErrorStats(horizon=horizon, count=0)

    return results


# --------------------------------------------------------------------------- #
# 2. Action sensitivity (brief sections 23, 24)
# --------------------------------------------------------------------------- #


@dataclass
class ActionSensitivityStats:
    """How much a prediction moves when the action moves, above sampling noise.

    Definition
    ----------
    DIMA's denoiser is stochastic: :meth:`DiffusionSampler.sample` starts from
    ``randn * sigma_max`` and the action of one agent at a time is unmasked over
    the denoising steps (``diffusion_sampler.py:87-97``).  A bare
    ``||prediction(a_i) - prediction(a_j)||`` therefore measures sampling noise
    *plus* action effect, and would show a comfortable non-zero value even for a
    model that ignores actions entirely.

    So this metric draws ``num_samples`` predictions per candidate action from the
    *same* conditioning window and separates the two:

    ``between``
        mean pairwise L2 distance between the per-action **mean** predictions.
        As ``num_samples`` grows this converges to the action effect.
    ``within``
        mean L2 distance from a sample to its own action's mean, times
        ``sqrt(2)`` so that it is on the same scale as a pairwise distance.  This
        is the noise floor.
    ``ratio = between / within``
        ``~1`` means the model is action-blind at this state; ``>> 1`` means
        actions genuinely change the prediction.  This ratio, not ``between``
        alone, is the quantity to read.

    Distances are computed in the model's own normalised state space, which is
    the space the denoiser was trained in and where dimensions are comparable.
    """

    between: Optional[float] = None
    within: Optional[float] = None
    ratio: Optional[float] = None
    per_camera: Dict[int, float] = field(default_factory=dict)
    per_camera_ratio: Dict[int, float] = field(default_factory=dict)
    num_samples: int = 1
    num_actions: int = 0

    @property
    def asymmetry(self) -> Optional[float]:
        """max/min of the per-camera sensitivity (brief section 24).

        A large value means the denoiser's per-agent conditioning is unbalanced:
        some cameras' actions move the prediction far more than others'.
        """

        if len(self.per_camera) < 2:
            return None
        values = np.array(list(self.per_camera.values()), dtype=np.float64)
        smallest = float(values.min())
        if smallest <= 0.0:
            return float('inf')
        return float(values.max() / smallest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'between': self.between,
            'within': self.within,
            'ratio': self.ratio,
            'per_camera': self.per_camera,
            'per_camera_ratio': self.per_camera_ratio,
            'asymmetry': self.asymmetry,
            'num_samples': self.num_samples,
            'num_actions': self.num_actions,
        }


def _pairwise_mean_distance(vectors: np.ndarray) -> float:
    if vectors.shape[0] < 2:
        return float('nan')
    diff = vectors[:, None, :] - vectors[None, :, :]
    distance = np.linalg.norm(diff, axis=-1)
    upper = np.triu_indices(vectors.shape[0], k=1)
    return float(distance[upper].mean())


def compute_action_sensitivity(
    world_model: WorldModel,
    history: History,
    base_action: np.ndarray,
    *,
    num_cameras: int,
    num_actions: int,
    num_samples: int = 8,
    action_subset: Optional[Sequence[int]] = None,
) -> ActionSensitivityStats:
    """Per-camera action sensitivity at one state (brief sections 23, 24)."""

    candidates = np.asarray(
        action_subset if action_subset is not None else np.arange(num_actions), dtype=np.int64
    )
    stats = ActionSensitivityStats(num_samples=num_samples, num_actions=int(candidates.size))

    per_camera_between: Dict[int, float] = {}
    per_camera_within: Dict[int, float] = {}

    for camera in range(num_cameras):
        joint = np.tile(np.asarray(base_action, dtype=np.int64), (candidates.size, 1))
        joint[:, camera] = candidates

        means = []
        withins = []
        for row in joint:
            samples = np.stack(
                [
                    world_model.predict(history, row, horizon=1).states[0, 0]
                    for _ in range(num_samples)
                ]
            )
            mean = samples.mean(axis=0)
            means.append(mean)
            if num_samples > 1:
                withins.append(float(np.linalg.norm(samples - mean, axis=-1).mean()))

        means = np.stack(means)
        between = _pairwise_mean_distance(means)
        within = float(np.mean(withins)) * np.sqrt(2.0) if withins else float('nan')

        per_camera_between[camera] = between
        per_camera_within[camera] = within

    stats.per_camera = {c: v for c, v in per_camera_between.items()}
    stats.per_camera_ratio = {
        c: (per_camera_between[c] / per_camera_within[c])
        if np.isfinite(per_camera_within[c]) and per_camera_within[c] > 0
        else float('nan')
        for c in per_camera_between
    }
    finite_between = [v for v in per_camera_between.values() if np.isfinite(v)]
    finite_within = [v for v in per_camera_within.values() if np.isfinite(v)]
    stats.between = float(np.mean(finite_between)) if finite_between else None
    stats.within = float(np.mean(finite_within)) if finite_within else None
    stats.ratio = (
        stats.between / stats.within
        if stats.between is not None and stats.within not in (None, 0.0)
        else None
    )
    return stats


# --------------------------------------------------------------------------- #
# 3. Action ranking quality (brief section 26)
# --------------------------------------------------------------------------- #


@dataclass
class ActionRankingStats:
    """Does the model order actions the way the environment does?

    Planning depends on *ordering*, not on absolute error: a model with large but
    uniform bias plans perfectly, and a model with small but action-scrambling
    error plans badly.  That is why this is reported separately from
    :class:`PredictionErrorStats` (brief sections 26, 35).
    """

    count: int = 0
    top1_agreement: Optional[float] = None
    top3_overlap: Optional[float] = None
    spearman: Optional[float] = None
    kendall: Optional[float] = None
    direction_agreement: Optional[float] = None
    predicted_utility: Optional[float] = None
    actual_utility: Optional[float] = None
    regret: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'count': self.count,
            'top1_agreement': self.top1_agreement,
            'top3_overlap': self.top3_overlap,
            'spearman': self.spearman,
            'kendall': self.kendall,
            'direction_agreement': self.direction_agreement,
            'predicted_utility': self.predicted_utility,
            'actual_utility': self.actual_utility,
            'regret': self.regret,
        }


def _utilities(prediction: Prediction, layout: ObservationLayout, utility) -> np.ndarray:
    scores = np.zeros(prediction.observations.shape[0], dtype=np.float64)
    for b in range(scores.size):
        view = SceneView.from_joint_observation(prediction.observations[b, -1], layout)
        scores[b] = utility(view)
    return scores


def compute_action_ranking(
    world_model: WorldModel,
    oracle: WorldModel,
    history: History,
    base_action: np.ndarray,
    layout: ObservationLayout,
    *,
    camera: int,
    num_actions: int,
    utility=SceneView.soft_coverage_estimate,
    horizon: int = 1,
) -> ActionRankingStats:
    """Rank one camera's actions under the model and under the environment."""

    candidates = np.tile(np.asarray(base_action, dtype=np.int64), (num_actions, 1))
    candidates[:, camera] = np.arange(num_actions)

    model_scores = _utilities(
        world_model.predict(history, candidates, horizon=horizon), layout, utility
    )
    truth_scores = _utilities(oracle.predict(history, candidates, horizon=horizon), layout, utility)

    stats = ActionRankingStats(count=int(num_actions))
    stats.top1_agreement = float(int(np.argmax(model_scores)) == int(np.argmax(truth_scores)))

    k = min(3, num_actions)
    model_top = set(np.argsort(-model_scores)[:k].tolist())
    truth_top = set(np.argsort(-truth_scores)[:k].tolist())
    stats.top3_overlap = len(model_top & truth_top) / float(k)

    try:
        from scipy import stats as scipy_stats

        if np.ptp(model_scores) > 0 and np.ptp(truth_scores) > 0:
            stats.spearman = float(scipy_stats.spearmanr(model_scores, truth_scores).statistic)
            stats.kendall = float(scipy_stats.kendalltau(model_scores, truth_scores).statistic)
    except ImportError:  # pragma: no cover - scipy is a MATE dependency
        pass

    upper = np.triu_indices(num_actions, k=1)
    model_sign = np.sign(model_scores[:, None] - model_scores[None, :])[upper]
    truth_sign = np.sign(truth_scores[:, None] - truth_scores[None, :])[upper]
    comparable = truth_sign != 0
    if comparable.any():
        stats.direction_agreement = float(
            (model_sign[comparable] == truth_sign[comparable]).mean()
        )

    chosen = int(np.argmax(model_scores))
    stats.predicted_utility = float(model_scores[chosen])
    stats.actual_utility = float(truth_scores[chosen])
    stats.regret = float(truth_scores.max() - truth_scores[chosen])
    return stats


# --------------------------------------------------------------------------- #
# Prediction validity (brief section 22)
# --------------------------------------------------------------------------- #


def prediction_validity(prediction: Prediction, layout: ObservationLayout) -> Dict[str, float]:
    """Fraction of predicted observations that are physically admissible.

    A diffusion model can emit anything; this checks the decoded quantities
    against MATE's own limits -- positions inside the 2000x2000 terrain, viewing
    angle in ``[0, 180]``, non-negative sight range -- so that "the world model is
    producing nonsense" is a measurable statement rather than an impression.
    """

    obs = prediction.observations
    finite = float(np.isfinite(obs).mean())

    in_terrain: List[float] = []
    angle_ok: List[float] = []
    range_ok: List[float] = []
    for b in range(obs.shape[0]):
        for h in range(obs.shape[1]):
            if not np.isfinite(obs[b, h]).all():
                continue
            view = SceneView.from_joint_observation(obs[b, h], layout)
            in_terrain.append(float((np.abs(view.camera_positions) <= 1000.0).all()))
            angle_ok.append(
                float(((view.camera_viewing_angles >= 0.0) & (view.camera_viewing_angles <= 180.0)).all())
            )
            range_ok.append(float((view.camera_sight_ranges >= 0.0).all()))

    return {
        'finite_fraction': finite,
        'camera_in_terrain': float(np.mean(in_terrain)) if in_terrain else float('nan'),
        'viewing_angle_valid': float(np.mean(angle_ok)) if angle_ok else float('nan'),
        'sight_range_valid': float(np.mean(range_ok)) if range_ok else float('nan'),
    }


# --------------------------------------------------------------------------- #
# Report formatting (brief section 22)
# --------------------------------------------------------------------------- #


def format_world_model_report(
    *,
    errors: Optional[Dict[int, PredictionErrorStats]] = None,
    sensitivity: Optional[ActionSensitivityStats] = None,
    ranking: Optional[Dict[int, ActionRankingStats]] = None,
    validity: Optional[Dict[str, float]] = None,
) -> str:
    """The section 22 block, with ``N/A`` wherever a metric is unavailable."""

    lines: List[str] = ['', 'WORLD MODEL DIAGNOSTICS', '=' * 60, '', 'Prediction error:']
    if errors:
        lines.append(f'  {"H":>4}  {"count":>6}  {"MAE":>10}  {"RMSE":>10}  {"ADE":>10}  {"FDE":>10}')
        for horizon in sorted(errors):
            stat = errors[horizon]
            lines.append(
                f'  {horizon:>4}  {stat.count:>6}  {_fmt(stat.mae):>10}  {_fmt(stat.rmse):>10}  '
                f'{_fmt(stat.ade):>10}  {_fmt(stat.fde):>10}'
            )
    else:
        lines.append(f'  {_NA}')

    lines += ['', 'Action sensitivity (between-action spread / sampling noise floor):']
    if sensitivity is not None:
        lines += [
            f'  samples per action : {sensitivity.num_samples}',
            f'  actions compared   : {sensitivity.num_actions}',
            f'  between            : {_fmt(sensitivity.between)}',
            f'  within (noise)     : {_fmt(sensitivity.within)}',
            f'  ratio              : {_fmt(sensitivity.ratio)}   (~1.0 means action-blind)',
            '',
            'Per-camera sensitivity (brief section 24):',
        ]
        for camera in sorted(sensitivity.per_camera):
            lines.append(
                f'  camera {camera}: between={_fmt(sensitivity.per_camera[camera])}  '
                f'ratio={_fmt(sensitivity.per_camera_ratio.get(camera))}'
            )
        lines.append(f'  max/min ratio      : {_fmt(sensitivity.asymmetry)}')
    else:
        lines.append(f'  {_NA}')

    lines += ['', 'Action ranking vs. the environment (brief section 26):']
    if ranking:
        lines.append(
            f'  {"camera":>7}  {"top1":>6}  {"top3":>6}  {"spearman":>9}  {"kendall":>8}  '
            f'{"direction":>9}  {"regret":>8}'
        )
        for camera in sorted(ranking):
            stat = ranking[camera]
            lines.append(
                f'  {camera:>7}  {_fmt(stat.top1_agreement, ".2f"):>6}  '
                f'{_fmt(stat.top3_overlap, ".2f"):>6}  {_fmt(stat.spearman, ".4f"):>9}  '
                f'{_fmt(stat.kendall, ".4f"):>8}  {_fmt(stat.direction_agreement, ".4f"):>9}  '
                f'{_fmt(stat.regret, ".4f"):>8}'
            )
    else:
        lines.append(f'  {_NA}')

    lines += ['', 'Prediction validity:']
    if validity:
        for key, value in validity.items():
            lines.append(f'  {key:<22}: {_fmt(value)}')
    else:
        lines.append(f'  {_NA}')

    lines.append('=' * 60)
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# 4. Planner diagnostics (brief section 27)
# --------------------------------------------------------------------------- #


class PlannerAccumulator:
    """Aggregates the planner-side metrics MATE can actually support.

    Deliberately omitted: "target assignment" and "assignment switch rate".
    MATE's camera action is a rotation/zoom delta, not an assignment, and none of
    the planners here maintain a camera-to-target assignment -- reporting one
    would be fabricating a metric (brief section 27).  ``camera_redundancy`` is
    reported instead, since it is well defined from MATE's own view masks.
    """

    def __init__(self, num_cameras: int, num_actions: int) -> None:
        self.num_cameras = num_cameras
        self.num_actions = num_actions
        self.action_histogram = np.zeros((num_cameras, num_actions), dtype=np.int64)
        self.action_changes = np.zeros(num_cameras, dtype=np.int64)
        self.steps = 0
        self._previous: Optional[np.ndarray] = None
        self._predicted_utility: List[float] = []
        self._actual_utility: List[float] = []
        self._redundancy: List[float] = []

    def update(
        self,
        action: np.ndarray,
        planner_diagnostics: Dict[str, Any],
        view: Optional[SceneView] = None,
        actual_utility: Optional[float] = None,
    ) -> None:
        action = np.asarray(action, dtype=np.int64).ravel()
        self.steps += 1
        for camera, index in enumerate(action):
            self.action_histogram[camera, index] += 1
        if self._previous is not None:
            self.action_changes += (action != self._previous).astype(np.int64)
        self._previous = action

        predicted = planner_diagnostics.get('predicted_utility')
        if predicted is not None and np.isfinite(predicted):
            self._predicted_utility.append(float(predicted))
        if actual_utility is not None and np.isfinite(actual_utility):
            self._actual_utility.append(float(actual_utility))
        if view is not None and view.num_targets:
            tracked = view.tracking_matrix().sum(axis=0)
            covered = tracked > 0
            self._redundancy.append(float(tracked[covered].mean()) if covered.any() else 0.0)

    def summary(self) -> Dict[str, Any]:
        histogram = self.action_histogram / max(1, self.steps)
        entropy = []
        for camera in range(self.num_cameras):
            probabilities = histogram[camera]
            positive = probabilities[probabilities > 0]
            entropy.append(float(-(positive * np.log(positive)).sum()))

        predicted = np.array(self._predicted_utility, dtype=np.float64)
        actual = np.array(self._actual_utility, dtype=np.float64)
        disagreement = None
        if predicted.size and actual.size and predicted.size == actual.size:
            disagreement = float(np.abs(predicted - actual).mean())

        return {
            'steps': self.steps,
            'action_histogram': self.action_histogram.tolist(),
            'action_entropy_per_camera': entropy,
            'action_switch_rate_per_camera': (
                self.action_changes / max(1, self.steps - 1)
            ).tolist(),
            'distinct_actions_per_camera': [
                int((self.action_histogram[c] > 0).sum()) for c in range(self.num_cameras)
            ],
            'camera_redundancy': float(np.mean(self._redundancy)) if self._redundancy else None,
            'predicted_utility_mean': float(predicted.mean()) if predicted.size else None,
            'actual_utility_mean': float(actual.mean()) if actual.size else None,
            'prediction_planner_disagreement': disagreement,
        }
