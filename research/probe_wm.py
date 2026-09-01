"""Component-wise probe of a trained world model, for when the headline
diagnostics say nothing is working.

``validate`` reports ``sighting_recall_h1 = 0`` and ``ade = N/A`` together, and
those two are not independent: ADE is computed only over targets *both* views
sight, so zero recall forces ADE to N/A. That leaves an ambiguity this module
exists to settle:

* the model really has collapsed onto "no target is ever sighted", or
* the model predicts the sighting mask reasonably but the value never crosses the
  ``> 0.5`` threshold ``SceneView`` applies, so a working model reads as broken.

The probe answers it by looking at the raw predicted numbers per observation
component instead of at the derived booleans:

``mask``          the sighting flag channel -- its predicted distribution against
                  the truth, and what recall would be at other thresholds
``target_pos``    predicted vs true target positions, over truth-sighted targets
                  only, ignoring the mask entirely
``camera_pos``    camera x, y -- these are near-deterministic given the action,
                  so a model that cannot get them right has learned nothing
``camera_orient`` camera sight vector
``reward``        the reward head
``continue``      the continuation head

The camera block is the control: it is the easiest part of the transition, so if
it is also wrong the problem is the model, not the sighting metric.

Read-only. Loads a checkpoint and a held-out dataset; touches neither.

    python -m research.probe_wm --checkpoint runs/<tag>/wm/ckpt/latest.pth \\
        --dataset runs/<tag>/dataset --transitions 300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from mate import constants as consts
from research.collect import load_dataset
from research.config import allow_dima_checkpoint_globals
from research.logging_utils import log
from research.train_wm import build_env_from_manifest
from research.validation import ValidationEpisode, _history_at
from research.views import ObservationLayout, SceneView
from research.world_model import DIMAWorldModel


__all__ = ['probe', 'main']


#: Thresholds the mask is scored at.  ``SceneView`` uses 0.5; the rest are here to
#: show whether that particular cut is what produces zero recall.
MASK_THRESHOLDS = (0.1, 0.25, 0.5, 0.75, 0.9)


def _weights_checkpoint(checkpoint: Path) -> Path:
    """A checkpoint in the weights-only shape ``DIMAWorldModel`` can load.

    ``ckpt/latest.pth`` is the resumable checkpoint written by
    ``DreamerLearner.save_full``: optimiser state, counters and RNG state wrapped
    around the weights. It is the freshest state a running job has on disk and so
    the natural thing to probe, but ``load_pretrained`` expects the flat
    ``params()`` layout. Rather than make the caller hunt for the newest
    ``model_ep*.pth``, the weights are extracted into a sibling file here.

    A checkpoint already in the flat shape is returned unchanged.
    """

    import torch

    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'learner' not in ckpt:
        return checkpoint

    state = ckpt['learner']
    modules = state['modules']
    rms_fields = state['state_rms']

    from agent.utils.running_mean_std import RunningMeanStd

    running_mean_std = RunningMeanStd(shape=np.asarray(rms_fields['mean']).shape)
    running_mean_std.mean = np.asarray(rms_fields['mean'], dtype=np.float64)
    running_mean_std.var = np.asarray(rms_fields['var'], dtype=np.float64)
    running_mean_std.count = float(rms_fields['count'])

    flat = {name: modules[name] for name in
            ('state_decoder', 'denoiser', 'rew_end_model', 'actor', 'critic')
            if name in modules}
    flat['running_mean_std'] = running_mean_std

    extracted = checkpoint.with_name(checkpoint.stem + '_weights.pth')
    torch.save(flat, extracted)
    log('PROBE', f'extracted weights from the resumable checkpoint -> {extracted}')
    return extracted


def _summarise(name: str, error: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Error of one component, next to the spread of the truth it is predicting.

    The truth's own standard deviation is the number that makes the error
    readable: an MAE below it means the model beats predicting the mean, and an
    MAE at or above it means it does not.
    """

    error = np.asarray(error, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    return {
        'component': name,
        'mae': float(np.abs(error).mean()) if error.size else float('nan'),
        'rmse': float(np.sqrt(np.square(error).mean())) if error.size else float('nan'),
        'truth_std': float(truth.std()) if truth.size else float('nan'),
        'truth_mean': float(truth.mean()) if truth.size else float('nan'),
        'count': int(error.size),
    }


def probe(
    checkpoint: Path,
    dataset_dir: Path,
    *,
    transitions: int = 300,
    device: Optional[str] = None,
    num_samples: int = 1,
    seed: int = 0,
) -> Dict[str, object]:
    """One-step predictions on held-out transitions, broken down by component."""

    manifest, rollouts = load_dataset(dataset_dir)
    env = build_env_from_manifest(manifest, seed=seed)
    layout = ObservationLayout.from_env_metadata(env.metadata())

    # DreamerLearner.load_pretrained calls torch.load with PyTorch's 2.6 default
    # of weights_only=True; DIMA checkpoints carry a live RunningMeanStd.
    allow_dima_checkpoint_globals()
    model = DIMAWorldModel(
        env, _weights_checkpoint(checkpoint), device=device, num_samples=num_samples
    )
    conditioning = model.conditioning_steps

    episodes = [ValidationEpisode.from_rollout(r, env) for r in rollouts]
    episodes = [e for e in episodes if len(e.transitions) > conditioning + 1]
    if not episodes:
        raise ValueError(
            f'No held-out episode is longer than the {conditioning}-step conditioning window.'
        )

    rng = np.random.default_rng(seed)

    predicted: List[np.ndarray] = []
    actual: List[np.ndarray] = []
    rewards_pred: List[float] = []
    continues_pred: List[float] = []

    while len(predicted) < transitions:
        episode = episodes[int(rng.integers(len(episodes)))]
        last = len(episode.transitions) - 1
        if last <= conditioning:
            continue
        t = int(rng.integers(conditioning - 1, last))

        history = _history_at(episode, t, conditioning)
        action = episode.transitions[t].action
        prediction = model.predict(history, np.asarray(action), horizon=1)

        predicted.append(np.asarray(prediction.observations[0, 0], dtype=np.float64))
        actual.append(np.asarray(episode.transitions[t].next_obs_raw, dtype=np.float64))
        if prediction.rewards is not None:
            rewards_pred.append(float(prediction.rewards[0, 0]))
        if prediction.continues is not None:
            continues_pred.append(float(prediction.continues[0, 0]))

    pred = np.stack(predicted)   # (N, C, obs_dim), MATE world units
    true = np.stack(actual)

    report: Dict[str, object] = {
        'checkpoint': str(checkpoint),
        'dataset': str(dataset_dir),
        'transitions': int(pred.shape[0]),
        'num_cameras': layout.num_cameras,
        'num_targets': layout.num_targets,
        'conditioning_steps': conditioning,
        'num_samples': num_samples,
    }

    # ---- the sighting mask, before any threshold is applied ---------------- #
    opp = slice(
        layout.slices['opponent_states_with_mask'].start,
        layout.slices['opponent_states_with_mask'].stop,
    )
    entry_dim = consts.TARGET_STATE_DIM_PUBLIC + 1
    pred_opp = pred[:, :, opp].reshape(-1, layout.num_cameras, layout.num_targets, entry_dim)
    true_opp = true[:, :, opp].reshape(-1, layout.num_cameras, layout.num_targets, entry_dim)

    pred_mask = pred_opp[..., consts.TARGET_STATE_DIM_PUBLIC]
    true_mask = true_opp[..., consts.TARGET_STATE_DIM_PUBLIC]
    truth_sighted = true_mask > 0.5

    report['mask'] = {
        'true_positive_rate': float(truth_sighted.mean()),
        'pred_min': float(pred_mask.min()),
        'pred_max': float(pred_mask.max()),
        'pred_mean': float(pred_mask.mean()),
        'pred_mean_where_truth_sighted': (
            float(pred_mask[truth_sighted].mean()) if truth_sighted.any() else None
        ),
        'pred_mean_where_truth_unsighted': (
            float(pred_mask[~truth_sighted].mean()) if (~truth_sighted).any() else None
        ),
        # Recall at each cut: if recall is zero at 0.5 but not at a lower one, the
        # model has the signal and only the threshold is wrong.
        'recall_at_threshold': {
            str(th): (
                float((pred_mask[truth_sighted] > th).mean()) if truth_sighted.any() else None
            )
            for th in MASK_THRESHOLDS
        },
        'false_positive_rate_at_threshold': {
            str(th): (
                float((pred_mask[~truth_sighted] > th).mean())
                if (~truth_sighted).any()
                else None
            )
            for th in MASK_THRESHOLDS
        },
    }

    # ---- per-component errors, mask ignored ------------------------------- #
    self_state = layout.slices['self_state']
    pred_self = pred[:, :, self_state]
    true_self = true[:, :, self_state]

    components: List[Dict[str, float]] = [
        _summarise('camera_pos_x', pred_self[..., 0] - true_self[..., 0], true_self[..., 0]),
        _summarise('camera_pos_y', pred_self[..., 1] - true_self[..., 1], true_self[..., 1]),
        _summarise('camera_sight_vec_x', pred_self[..., 3] - true_self[..., 3], true_self[..., 3]),
        _summarise('camera_sight_vec_y', pred_self[..., 4] - true_self[..., 4], true_self[..., 4]),
        _summarise('camera_viewing_angle', pred_self[..., 5] - true_self[..., 5], true_self[..., 5]),
        _summarise('camera_sight_range_max', pred_self[..., 6] - true_self[..., 6], true_self[..., 6]),
    ]

    # Target positions scored over truth-sighted entries only, deliberately
    # ignoring what the predicted mask says -- this is the number the headline
    # ADE cannot report while recall is zero.
    if truth_sighted.any():
        pos_err = pred_opp[..., 0:2][truth_sighted] - true_opp[..., 0:2][truth_sighted]
        components.append(
            _summarise('target_pos (truth-sighted)', pos_err, true_opp[..., 0:2][truth_sighted])
        )
        report['target_displacement_truth_sighted'] = float(
            np.linalg.norm(pos_err, axis=-1).mean()
        )

    report['components'] = components

    # ---- reward and continuation heads ------------------------------------ #
    if rewards_pred:
        report['reward'] = {
            'pred_mean': float(np.mean(rewards_pred)),
            'pred_std': float(np.std(rewards_pred)),
        }
    if continues_pred:
        report['continue'] = {
            'pred_mean': float(np.mean(continues_pred)),
            'pred_std': float(np.std(continues_pred)),
        }

    # ---- what SceneView itself reports, for the direct comparison ---------- #
    recalls: List[float] = []
    for n in range(pred.shape[0]):
        pred_view = SceneView.from_joint_observation(pred[n], layout)
        true_view = SceneView.from_joint_observation(true[n], layout)
        if true_view.target_sighted.any():
            agreed = np.logical_and(true_view.target_sighted, pred_view.target_sighted)
            recalls.append(float(agreed.sum()) / float(true_view.target_sighted.sum()))
    report['sceneview_sighting_recall'] = float(np.mean(recalls)) if recalls else None

    env.close()
    return report


def _print_report(report: Dict[str, object]) -> None:
    log('PROBE', f'checkpoint {report["checkpoint"]}')
    log(
        'PROBE',
        f'{report["transitions"]} held-out transitions, '
        f'{report["num_cameras"]} cameras x {report["num_targets"]} targets, '
        f'conditioning={report["conditioning_steps"]} samples={report["num_samples"]}',
    )

    mask = report['mask']
    log('PROBE', f'--- sighting mask (SceneView cuts at > 0.5) ---')
    log('PROBE', f'  truth sighted rate      {mask["true_positive_rate"]:.4f}')
    log(
        'PROBE',
        f'  predicted mask range    [{mask["pred_min"]:.4f}, {mask["pred_max"]:.4f}] '
        f'mean={mask["pred_mean"]:.4f}',
    )
    sighted_mean = mask['pred_mean_where_truth_sighted']
    unsighted_mean = mask['pred_mean_where_truth_unsighted']
    if sighted_mean is not None and unsighted_mean is not None:
        log(
            'PROBE',
            f'  mean where truth sighted {sighted_mean:.4f} vs unsighted {unsighted_mean:.4f} '
            f'(separation {sighted_mean - unsighted_mean:+.4f})',
        )
    for th in MASK_THRESHOLDS:
        recall = mask['recall_at_threshold'][str(th)]
        fpr = mask['false_positive_rate_at_threshold'][str(th)]
        if recall is None:
            continue
        log('PROBE', f'  threshold {th:<5} recall={recall:.4f}  false-positive={fpr:.4f}')

    log('PROBE', '--- per-component error (camera block is the control) ---')
    log('PROBE', f'  {"component":<28} {"mae":>12} {"rmse":>12} {"truth_std":>12}')
    for row in report['components']:
        log(
            'PROBE',
            f'  {row["component"]:<28} {row["mae"]:>12.4f} {row["rmse"]:>12.4f} '
            f'{row["truth_std"]:>12.4f}',
        )

    if 'target_displacement_truth_sighted' in report:
        log(
            'PROBE',
            f'  target displacement over truth-sighted targets: '
            f'{report["target_displacement_truth_sighted"]:.4f}',
        )

    for head in ('reward', 'continue'):
        if head in report:
            log(
                'PROBE',
                f'  {head}: mean={report[head]["pred_mean"]:.4f} '
                f'std={report[head]["pred_std"]:.4f}',
            )

    recall = report['sceneview_sighting_recall']
    log('PROBE', f'SceneView sighting recall (what validate reports): '
                 f'{"N/A" if recall is None else f"{recall:.4f}"}')

    log('PROBE', '--- reading this ---')
    log(
        'PROBE',
        '  recall 0 at 0.5 but > 0 at a lower threshold  -> threshold artefact, '
        'the model has the signal',
    )
    log(
        'PROBE',
        '  mask separation ~0 and camera mae >= truth_std -> the model has learned '
        'nothing action-dependent yet',
    )
    log(
        'PROBE',
        '  camera mae << truth_std but mask separation ~0 -> the model learned the '
        'easy block only; more training or a different sighting loss',
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.probe_wm', description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path,
                        help='a weights checkpoint (model_*.pth) or a resumable one '
                             '(ckpt/latest.pth); the latter has its weights extracted')
    parser.add_argument('--dataset', required=True, type=Path,
                        help='held-out episodes to probe on; any collected dataset works')
    parser.add_argument('--transitions', type=int, default=300)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-samples', type=int, default=1,
                        help='diffusion samples averaged per prediction')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=None, help='write the report as JSON')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    report = probe(
        args.checkpoint,
        args.dataset,
        transitions=args.transitions,
        device=args.device,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    _print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
        log('PROBE', f'report -> {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
