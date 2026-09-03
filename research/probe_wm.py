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
from research.config import allow_dima_checkpoint_globals, resolve_weights_checkpoint
from research.diagnostics import _pairwise_mean_distance
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
    """Kept as a thin alias so the probe reads the same checkpoints evaluation does."""

    resolved = resolve_weights_checkpoint(checkpoint)
    if Path(resolved) != Path(checkpoint):
        log('PROBE', f'extracted weights from the resumable checkpoint -> {resolved}')
    return Path(resolved)


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


def _decoder_reconstruction(model, episodes, layout, rng, count: int) -> Dict[str, object]:
    """What the state decoder alone produces from the *true* next state.

    An observation never comes out of the denoiser directly. The denoiser predicts
    the next global state, and ``state_decoder.encode_decode`` turns that state
    into the joint observation (``world_model.py:610``). Two very different
    failures therefore look identical in the headline metrics:

    * the denoiser predicts the wrong state, or
    * the decoder cannot express the sighting mask no matter which state it gets.

    Feeding the ground-truth next state through the decoder separates them. If the
    mask is still flat here, the diffusion model is not the thing to fix -- the
    state has to be decodable before predicting it can help.
    """

    import torch

    states: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    while len(states) < count:
        episode = episodes[int(rng.integers(len(episodes)))]
        last = len(episode.transitions) - 1
        if last <= 1:
            continue
        t = int(rng.integers(0, last))
        # transitions[t].next_obs_raw is the successor of transitions[t], whose
        # state is transitions[t + 1].state.
        states.append(episode.transitions[t + 1].state)
        truths.append(episode.transitions[t].next_obs_raw)

    state_batch = torch.as_tensor(np.stack(states), dtype=torch.float32, device=model.device)
    with torch.no_grad():
        normalised = model._normalize_state(state_batch.unsqueeze(1)).squeeze(1)
        flat = model.state_decoder.encode_decode(normalised)
    decoded = flat.reshape(len(states), model.num_agents, model.obs_dim).cpu().numpy()
    decoded_world = model._unrescale_obs(decoded.astype(np.float64))
    truth_world = np.stack(truths)

    entry_dim = consts.TARGET_STATE_DIM_PUBLIC + 1
    opp = layout.slices['opponent_states_with_mask']
    dec_mask = decoded_world[:, :, opp].reshape(
        -1, layout.num_cameras, layout.num_targets, entry_dim
    )[..., consts.TARGET_STATE_DIM_PUBLIC]
    true_mask = truth_world[:, :, opp].reshape(
        -1, layout.num_cameras, layout.num_targets, entry_dim
    )[..., consts.TARGET_STATE_DIM_PUBLIC]
    sighted = true_mask > 0.5

    self_state = layout.slices['self_state']
    camera_err = decoded_world[:, :, self_state][..., 0:2] - truth_world[:, :, self_state][..., 0:2]

    return {
        'count': len(states),
        'mask_pred_min': float(dec_mask.min()),
        'mask_pred_max': float(dec_mask.max()),
        'mask_mean_where_sighted': float(dec_mask[sighted].mean()) if sighted.any() else None,
        'mask_mean_where_unsighted': (
            float(dec_mask[~sighted].mean()) if (~sighted).any() else None
        ),
        'recall_at_threshold': {
            str(th): (float((dec_mask[sighted] > th).mean()) if sighted.any() else None)
            for th in MASK_THRESHOLDS
        },
        'camera_pos_mae': float(np.abs(camera_err).mean()),
        'camera_pos_truth_std': float(truth_world[:, :, self_state][..., 0:2].std()),
    }


def _seeded_predict(model, history, action, seed: int) -> np.ndarray:
    """One prediction with the diffusion noise pinned to ``seed``.

    ``sensitivity_ratio`` compares actions through 8-sample means, so anything
    smaller than the sampling error of those means -- 1/sqrt(8) of the noise --
    is invisible to it. Pinning the seed removes the noise from the comparison
    entirely: two predictions that differ only in the action see identical noise,
    so whatever separates them is the action's doing, however small.
    """

    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    prediction = model.predict(history, np.asarray(action), horizon=1)
    state = np.asarray(prediction.states[0, 0], dtype=np.float64)
    reward = (
        float(prediction.rewards[0, 0]) if prediction.rewards is not None else float('nan')
    )
    return state, reward


def action_effect(
    model,
    episodes,
    rng,
    *,
    num_agents: int,
    num_actions: int,
    states: int = 20,
    candidates: int = 6,
    noise_seeds: int = 6,
) -> Dict[str, object]:
    """How much the action moves the predicted next state, noise held fixed.

    Three quantities per sampled state:

    ``effect``       spread of predictions across candidate actions for one camera,
                     all sharing one noise seed
    ``noise``        spread across noise seeds for one fixed action -- the scale
                     ``effect`` has to be read against
    ``extreme``      distance between the two most different joint actions
                     (all-zero vs all-max), same seed: the largest effect the
                     model can show at this state

    ``identical_pairs`` counts candidate pairs whose predictions match to
    floating-point exactness. A model that merely learned to ignore the action
    still produces slightly different numbers; bitwise-identical output means the
    action never reached the network at all, which is a wiring bug rather than a
    training outcome.
    """

    conditioning = model.conditioning_steps
    usable = [e for e in episodes if len(e.transitions) > conditioning + 1]

    effects, noises, extremes = [], [], []
    reward_spreads, reward_noises, reward_extremes = [], [], []
    per_camera: List[List[float]] = []
    identical_pairs = 0
    compared_pairs = 0

    action_grid = np.linspace(0, num_actions - 1, candidates).astype(np.int64)

    for i in range(states):
        episode = usable[int(rng.integers(len(usable)))]
        t = int(rng.integers(conditioning - 1, len(episode.transitions) - 1))
        history = _history_at(episode, t, conditioning)
        base = np.asarray(episode.transitions[t].action, dtype=np.int64).reshape(-1)
        seed = 10_000 + i

        # One camera at a time, everything else pinned.  Which camera matters:
        # sample_agent_order reverses the agent index, so camera n-1 conditions the
        # first denoising step (highest sigma, coarse structure) and camera 0 the
        # last (lowest sigma, final refinement).  Measuring only one of them would
        # not say how much the action moves the prediction in general.
        per_camera_now = []
        for camera in range(num_agents):
            preds, rew = [], []
            for a in action_grid:
                joint = base.copy()
                joint[camera] = a
                state, reward = _seeded_predict(model, history, joint, seed)
                preds.append(state)
                rew.append(reward)
            preds = np.stack(preds)
            per_camera_now.append(_pairwise_mean_distance(preds))
            reward_spreads.append(float(np.nanstd(rew)))

            for x in range(len(preds)):
                for y in range(x + 1, len(preds)):
                    compared_pairs += 1
                    if np.array_equal(preds[x], preds[y]):
                        identical_pairs += 1

        per_camera.append(per_camera_now)
        effects.append(float(np.mean(per_camera_now)))

        # Same action, different noise: the scale to read `effect` against.
        noise_pairs = [
            _seeded_predict(model, history, base, seed + 1000 * k) for k in range(noise_seeds)
        ]
        noise_preds = np.stack([p for p, _ in noise_pairs])
        noises.append(_pairwise_mean_distance(noise_preds))
        reward_noises.append(float(np.nanstd([r for _, r in noise_pairs])))

        # The two furthest-apart joint actions, noise held fixed.
        low, low_rew = _seeded_predict(model, history, np.zeros(num_agents, dtype=np.int64), seed)
        high, high_rew = _seeded_predict(
            model, history, np.full(num_agents, num_actions - 1, dtype=np.int64), seed
        )
        extremes.append(float(np.linalg.norm(high - low)))
        reward_extremes.append(abs(high_rew - low_rew))

    effect = float(np.mean(effects))
    noise = float(np.mean(noises))
    return {
        'states': states,
        'num_agents': int(num_agents),
        'candidates': int(action_grid.size),
        'effect': effect,
        'noise': noise,
        'effect_over_noise': effect / noise if noise else float('inf'),
        'extreme_effect': float(np.mean(extremes)),
        'extreme_over_noise': float(np.mean(extremes)) / noise if noise else float('inf'),
        'per_camera_effect': np.asarray(per_camera, dtype=float).mean(axis=0).tolist(),
        'per_camera_over_noise': (
            (np.asarray(per_camera, dtype=float).mean(axis=0) / noise).tolist()
            if noise else None
        ),
        'identical_pairs': identical_pairs,
        'compared_pairs': compared_pairs,
        # The reward head, measured the same way.  This is the quantity the actor's
        # lambda-return is built from, so it -- not the state -- is what decides
        # whether learning in imagination can teach a policy anything at all.
        'reward_effect': float(np.nanmean(reward_spreads)),
        'reward_noise': float(np.nanmean(reward_noises)),
        'reward_effect_over_noise': (
            float(np.nanmean(reward_spreads) / np.nanmean(reward_noises))
            if np.nanmean(reward_noises) else float('inf')
        ),
        'reward_extreme_effect': float(np.nanmean(reward_extremes)),
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

    # Decoder in isolation: the same question asked of the true next state.
    report['decoder_reconstruction'] = _decoder_reconstruction(
        model, episodes, layout, np.random.default_rng(seed + 1), min(transitions, 200)
    )

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

    dec = report['decoder_reconstruction']
    log('PROBE', '--- state decoder alone, fed the TRUE next state ---')
    log(
        'PROBE',
        f'  decoded mask range      [{dec["mask_pred_min"]:.4f}, {dec["mask_pred_max"]:.4f}]',
    )
    if dec['mask_mean_where_sighted'] is not None:
        log(
            'PROBE',
            f'  mean where sighted      {dec["mask_mean_where_sighted"]:.4f} vs unsighted '
            f'{dec["mask_mean_where_unsighted"]:.4f} '
            f'(separation {dec["mask_mean_where_sighted"] - dec["mask_mean_where_unsighted"]:+.4f})',
        )
    for th in MASK_THRESHOLDS:
        recall = dec['recall_at_threshold'][str(th)]
        if recall is not None:
            log('PROBE', f'  threshold {th:<5} recall={recall:.4f}')
    log(
        'PROBE',
        f'  camera pos mae          {dec["camera_pos_mae"]:.4f} '
        f'(truth std {dec["camera_pos_truth_std"]:.4f})',
    )

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
    log(
        'PROBE',
        '  decoder mask flat on the TRUE state            -> the decoder is the '
        'bottleneck, not the denoiser: predicting states better cannot help',
    )
    log(
        'PROBE',
        '  decoder mask separated but prediction flat     -> the decoder works, the '
        'denoiser is not predicting usable states yet',
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
    parser.add_argument('--num-steps-denoising', type=int, default=None,
                        help='override the sampler schedule; inference-only, no retraining needed')
    parser.add_argument('--agent-order', default=None,
                        choices=['default', 'reverse', 'random', 'tiled'],
                        help='override the order actions condition denoising steps in')
    parser.add_argument('--action-effect', action='store_true',
                        help='measure how much the action moves the predicted state with the '
                             'diffusion noise pinned, instead of the full reconstruction probe')
    parser.add_argument('--states', type=int, default=20,
                        help='action-effect only: states sampled')
    parser.add_argument('--out', type=Path, default=None, help='write the report as JSON')
    return parser.parse_args(argv)


def _apply_sampler_overrides(model, args) -> None:
    """Retune the sampler on an already-loaded model.

    The EDM framework decouples inference-time sampling from training (the paper's
    section E.5), so the schedule and the conditioning order can be changed on a
    finished checkpoint without retraining anything.
    """

    cfg = model.sampler.cfg
    if args.num_steps_denoising is not None:
        cfg.num_steps_denoising = int(args.num_steps_denoising)
        from agent.world_models.diffusion.diffusion_sampler import build_sigmas

        model.sampler.sigmas = build_sigmas(
            cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho,
            model.sampler.denoiser.device,
        )
    if args.agent_order is not None:
        cfg.agent_order = str(args.agent_order)
    log('PROBE', f'sampler: num_steps_denoising={cfg.num_steps_denoising} '
                 f'agent_order={cfg.agent_order!r}')


def _run_action_effect(args) -> Dict[str, object]:
    manifest, rollouts = load_dataset(args.dataset)
    env = build_env_from_manifest(manifest, seed=args.seed)
    allow_dima_checkpoint_globals()
    model = DIMAWorldModel(
        env, _weights_checkpoint(args.checkpoint), device=args.device, num_samples=1
    )
    _apply_sampler_overrides(model, args)
    episodes = [ValidationEpisode.from_rollout(r, env) for r in rollouts]
    report = action_effect(
        model,
        episodes,
        np.random.default_rng(args.seed),
        num_agents=env.n_agents,
        num_actions=env.n_actions,
        states=args.states,
    )
    report['checkpoint'] = str(args.checkpoint)

    log('PROBE', f'checkpoint {args.checkpoint}')
    log('PROBE', f'{report["states"]} states x {report["candidates"]} candidate actions, '
                 f'diffusion noise pinned per state')
    log('PROBE', f'  action effect (mean over cameras) {report["effect"]:.4f}')
    for cam, (eff, ratio) in enumerate(
        zip(report['per_camera_effect'], report['per_camera_over_noise'] or [])
    ):
        log('PROBE', f'    camera {cam} (denoising step {report["num_agents"] - 1 - cam}) '
                     f'effect {eff:.4f}  = {ratio:.4f} x noise')
    log('PROBE', f'  noise floor   (vary seed)       {report["noise"]:.4f}')
    log('PROBE', f'  effect / noise                  {report["effect_over_noise"]:.4f}')
    log('PROBE', f'  extreme action pair             {report["extreme_effect"]:.4f} '
                 f'({report["extreme_over_noise"]:.4f} x noise)')
    log('PROBE', '--- reward head (what the actor actually learns from) ---')
    log('PROBE', f'  reward effect (vary action)     {report["reward_effect"]:.6f}')
    log('PROBE', f'  reward noise  (vary seed)       {report["reward_noise"]:.6f}')
    log('PROBE', f'  reward effect / noise           {report["reward_effect_over_noise"]:.4f}')
    log('PROBE', f'  extreme action pair             {report["reward_extreme_effect"]:.6f}')
    log('PROBE', f'  bitwise-identical pairs         {report["identical_pairs"]}'
                 f'/{report["compared_pairs"]}')
    log('PROBE', '--- reading this ---')
    log('PROBE', '  identical pairs > 0        -> the action never reaches the network: a wiring '
                 'bug, not a training outcome')
    log('PROBE', '  effect/noise ~0 but not 0  -> the action reaches it and the model ignores it')
    log('PROBE', '  effect/noise rising across checkpoints -> it is learning, just slowly')
    env.close()
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.action_effect:
        report = _run_action_effect(args)
        if args.out:
            args.out.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
            log('PROBE', f'report -> {args.out}')
        return 0
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
