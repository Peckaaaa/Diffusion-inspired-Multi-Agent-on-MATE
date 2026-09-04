"""Closed-loop evaluation and the baseline matrix (brief sections 19, 22-27, 30).

    python -m research.evaluate --planner reactive_greedy --episodes 20
    python -m research.evaluate --planner predictive_greedy --world-model dima \\
        --checkpoint runs/wm-4v2/ckpt/model_final.pth --episodes 20 --diagnostics
    python -m research.evaluate --baseline-matrix --checkpoint runs/wm-4v2/ckpt/model_final.pth

Every row of the baseline matrix runs with the same scenario, the same episode
limit, the same discretisation and the same seed list, so the rows are comparable
by construction (brief section 30).  Coverage and transport rate always come from
MATE's own counters, never from this project's estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.diagnostics import (
    PlannerAccumulator,
    compute_action_ranking,
    compute_action_sensitivity,
    format_world_model_report,
    horizon_error_report,
    prediction_validity,
)
from research.config import configure_torch, default_device
from research.env_adapter import DEFAULT_SCENARIO, MATEEnv
from research.logging_utils import RunLogger, log
from research.planners import CEMPlanner, build_planner, planner_class
from research.rollout import run_episode
from research.views import ObservationLayout, SceneView
from research.world_model import (
    AlphaOracleWorldModel,
    DIMAWorldModel,
    History,
    OracleWorldModel,
    WorldModel,
)


#: Brief section 30's matrix.  Extra rows are added by naming a planner, not by
#: editing an ``if`` chain anywhere in the loop.
BASELINE_MATRIX = (
    ('random', 'none'),
    ('mate_random', 'none'),
    ('reactive_greedy', 'none'),
    ('predictive_greedy', 'dima'),
    ('cem', 'dima'),
    ('dima_actor', 'dima'),
    ('oracle', 'oracle'),
)

#: ``cem_oracle`` answers the planner-ceiling question -- does joint-space search
#: beat coordinate descent when the model is perfect? -- but it is not in the
#: matrix above because it cannot be afforded there: ``OracleWorldModel`` forks
#: MATE once per candidate, so one step costs ``cem_samples * cem_iterations``
#: environment steps (384 at the defaults) against the greedy sweep's 100.  Run it
#: on its own, on a few short episodes:
#:
#:     python -m research.evaluate --planner cem_oracle --world-model oracle #:         --episodes 3 --max-episode-steps 50


def build_world_model(
    spec: str,
    env: MATEEnv,
    *,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    seed: int = 0,
    num_samples: int = 1,
) -> Optional[WorldModel]:
    """``'none' | 'dima' | 'oracle' | 'alpha_oracle:<alpha>' | module:Attribute``."""

    if spec in ('none', '', None):
        return None
    if spec == 'dima':
        if checkpoint is None:
            raise ValueError('--world-model dima needs --checkpoint.')
        return DIMAWorldModel(env, checkpoint, device=device, num_samples=num_samples)
    if spec == 'oracle':
        return OracleWorldModel(env)
    if spec.startswith('alpha_oracle'):
        _, _, alpha = spec.partition(':')
        return AlphaOracleWorldModel(env, alpha=float(alpha or 1.0), seed=seed)

    from research.planners import load_entry

    factory = load_entry(spec)
    return factory(env)


def evaluate(
    *,
    planner_spec: str,
    world_model_spec: str = 'none',
    scenario: str = DEFAULT_SCENARIO,
    episodes: int = 20,
    seed: int = 0,
    max_episode_steps: int = 200,
    discrete_levels: int = 5,
    checkpoint: Optional[str] = None,
    device: Optional[str] = None,
    horizon: int = 1,
    num_samples: int = 1,
    logger: Optional[RunLogger] = None,
    diagnostics: bool = False,
    diagnostic_states: int = 4,
    sensitivity_samples: int = 8,
    cem_samples: int = 128,
    cem_iterations: int = 3,
    cem_elite_frac: float = 0.1,
) -> Dict[str, Any]:
    """Run one row of the baseline matrix."""

    env = MATEEnv(
        scenario=scenario,
        seed=seed,
        discrete_levels=discrete_levels,
        max_episode_steps=max_episode_steps,
    )
    layout = ObservationLayout.from_env_metadata(env.metadata())

    world_model = build_world_model(
        world_model_spec,
        env,
        checkpoint=checkpoint,
        device=device,
        seed=seed,
        num_samples=num_samples,
    )
    # ``horizon`` is a search depth, so it goes only to planners that actually
    # search with the world model.  Some planners are handed a world model for
    # another reason -- ``dima_actor`` takes its policy network from it -- and
    # would reject the argument.
    cls = planner_class(planner_spec)
    searches = cls.USES_WORLD_MODEL
    planner_kwargs: Dict[str, Any] = {}
    if searches and world_model is not None:
        planner_kwargs['horizon'] = horizon
        # Search-shape arguments only mean something to the planner that has that
        # shape of search; the greedy sweep would reject them.
        if issubclass(cls, CEMPlanner):
            planner_kwargs.update(
                samples=cem_samples, iterations=cem_iterations, elite_frac=cem_elite_frac
            )
    planner = build_planner(
        planner_spec,
        env,
        world_model=world_model,
        seed=seed,
        **planner_kwargs,
    )

    log('ENV', env.describe())
    log('PLANNER', f'{planner.name}  {planner.describe()}')
    if world_model is not None:
        log('WM', str(world_model.describe()))
        if world_model.uses_privileged_state:
            log(
                'WARN',
                'this world model conditions on MATE\'s privileged global state; '
                'closed-loop results are NOT decentralised execution (see README).',
            )

    accumulator = PlannerAccumulator(env.n_agents, env.n_actions)
    per_episode: List[Dict[str, float]] = []
    collected: List[Any] = []
    started = time.time()

    for index in range(episodes):
        episode_seed = seed * 1000 + index
        result = run_episode(
            env,
            planner,
            world_model=world_model,
            episode=index,
            seed=episode_seed,
            max_steps=max_episode_steps,
            reference_prediction=world_model is not None,
        )
        collected.append(result)

        for transition, planner_diag in zip(result.transitions, result.planner_diagnostics):
            view = SceneView.from_joint_observation(transition.next_obs_raw, layout)
            accumulator.update(
                transition.action,
                planner_diag,
                view=view,
                actual_utility=view.soft_coverage_estimate(),
            )

        row = dict(result.metrics, episode=index, seed=episode_seed)
        per_episode.append(row)
        log(
            'EVAL',
            f'episode={index + 1}/{episodes} len={int(row["episode_length"])} '
            f'coverage={row["mean_coverage_rate"]:.4f} '
            f'transport={row["mean_transport_rate"]:.4f} '
            f'return={row["camera_team_return"]:+.3f}',
        )
        if logger is not None:
            logger.records(f'episodes_{planner.name}', [row])

    keys = [k for k in per_episode[0] if k not in ('episode', 'seed')]
    summary = {
        f'{key}_mean': float(np.mean([row[key] for row in per_episode])) for key in keys
    }
    summary.update(
        {f'{key}_std': float(np.std([row[key] for row in per_episode])) for key in keys}
    )
    summary['episodes'] = episodes
    summary['wall_seconds'] = round(time.time() - started, 2)
    summary['planner'] = planner.name
    summary['world_model'] = world_model.name if world_model is not None else 'none'
    summary['planner_diagnostics'] = accumulator.summary()

    log(
        'EVAL',
        f'{planner.name} / {summary["world_model"]}: '
        f'coverage={summary["mean_coverage_rate_mean"]:.4f}'
        f'+-{summary["mean_coverage_rate_std"]:.4f}  '
        f'transport={summary["mean_transport_rate_mean"]:.4f}  '
        f'normalized_target_reward={summary["normalized_target_episode_reward_mean"]:+.5f}',
    )
    log('PLANNER-DIAG', json.dumps(accumulator.summary(), default=str))

    if diagnostics and world_model is not None:
        summary['world_model_diagnostics'] = run_world_model_diagnostics(
            env,
            world_model,
            collected,
            layout,
            logger=logger,
            num_states=diagnostic_states,
            sensitivity_samples=sensitivity_samples,
            seed=seed,
        )

    env.close()
    return summary


def run_world_model_diagnostics(
    env: MATEEnv,
    world_model: WorldModel,
    episodes: Sequence[Any],
    layout: ObservationLayout,
    *,
    logger: Optional[RunLogger] = None,
    num_states: int = 4,
    sensitivity_samples: int = 8,
    seed: int = 0,
    horizons: Sequence[int] = (1, 3, 5, 10),
) -> Dict[str, Any]:
    """Brief sections 22-26, reported together and saved as structured records."""

    rng = np.random.default_rng(seed)

    errors = horizon_error_report(world_model, episodes, layout, horizons=horizons, rng=rng)

    # Sensitivity and ranking are measured at a handful of real states rather
    # than at every step: each state costs num_actions * num_samples world-model
    # calls per camera.
    conditioning = max(1, world_model.conditioning_steps)
    oracle = OracleWorldModel(env)
    sensitivity = None
    ranking: Dict[int, Any] = {}
    validity = None
    records: List[Dict[str, Any]] = []

    usable = [ep for ep in episodes if len(ep.transitions) > conditioning + 2]
    for _ in range(num_states):
        if not usable:
            break
        episode = usable[int(rng.integers(len(usable)))]
        t = int(rng.integers(conditioning - 1, len(episode.transitions) - 1))

        history = History(length=conditioning)
        for k in range(t - conditioning + 1, t + 1):
            history.states.append(episode.transitions[k].state)
            history.observations.append(episode.transitions[k].obs)
            history.actions.append(
                episode.transitions[k - 1].action if k > 0 else episode.transitions[k].action
            )

        base_action = episode.transitions[t].action

        if sensitivity is None:
            sensitivity = compute_action_sensitivity(
                world_model,
                history,
                base_action,
                num_cameras=env.n_agents,
                num_actions=env.n_actions,
                num_samples=sensitivity_samples,
            )

        # The oracle needs the *live* environment at this state; forking mid-episode
        # from a logged transition is not possible, so ranking is measured only for
        # oracle-capable world models or skipped with an explicit note.
        if isinstance(world_model, (OracleWorldModel, AlphaOracleWorldModel)):
            for camera in range(env.n_agents):
                ranking[camera] = compute_action_ranking(
                    world_model,
                    oracle,
                    history,
                    base_action,
                    layout,
                    camera=camera,
                    num_actions=env.n_actions,
                )

        prediction = world_model.predict(history, base_action, horizon=1)
        if validity is None:
            validity = prediction_validity(prediction, layout)

        # Brief section 25: save the comparison, not the tensors.
        predicted_view = SceneView.from_joint_observation(prediction.observations[0, 0], layout)
        actual_view = SceneView.from_joint_observation(episode.transitions[t].next_obs_raw, layout)
        records.append(
            {
                't': t,
                'action': base_action.tolist(),
                'predicted_target_positions': predicted_view.target_positions.tolist(),
                'actual_target_positions': actual_view.target_positions.tolist(),
                'predicted_camera_orientations': predicted_view.camera_orientations.tolist(),
                'actual_camera_orientations': actual_view.camera_orientations.tolist(),
                'predicted_coverage': predicted_view.coverage_estimate(),
                'actual_coverage': actual_view.coverage_estimate(),
                'position_error': float(
                    np.linalg.norm(
                        predicted_view.target_positions - actual_view.target_positions, axis=-1
                    ).mean()
                ),
            }
        )

    if logger is not None and records:
        logger.records(f'prediction_vs_actual_{world_model.name}', records)

    report = format_world_model_report(
        errors=errors, sensitivity=sensitivity, ranking=ranking, validity=validity
    )
    print(report)
    if not ranking:
        log(
            'WM-DIAG',
            'action ranking vs. the environment: N/A -- the oracle cannot be forked from a '
            'logged state; run with --world-model oracle or --world-model alpha_oracle:<a>.',
        )

    return {
        'errors': {h: stat.to_dict() for h, stat in errors.items()},
        'sensitivity': sensitivity.to_dict() if sensitivity is not None else None,
        'ranking': {c: stat.to_dict() for c, stat in ranking.items()},
        'validity': validity,
        'report': report,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.evaluate', description=__doc__)
    parser.add_argument('--planner', default='reactive_greedy')
    parser.add_argument(
        '--world-model',
        default='none',
        help="'none' | 'dima' | 'oracle' | 'alpha_oracle:<alpha>' | module:Attribute",
    )
    parser.add_argument('--baseline-matrix', action='store_true')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--device', default=None)
    parser.add_argument('--threads', type=int, default=None, help='torch CPU thread count')
    parser.add_argument('--horizon', type=int, default=1, help='planning horizon')
    parser.add_argument('--num-samples', type=int, default=1, help='diffusion samples per query')
    parser.add_argument('--diagnostics', action='store_true')
    parser.add_argument('--diagnostic-states', type=int, default=4)
    parser.add_argument('--sensitivity-samples', type=int, default=8)
    parser.add_argument('--cem-samples', type=int, default=128,
                        help='joint actions drawn per CEM iteration')
    parser.add_argument('--cem-iterations', type=int, default=3,
                        help='CEM refit iterations per environment step')
    parser.add_argument('--cem-elite-frac', type=float, default=0.1,
                        help='fraction of each CEM draw kept as the elite set')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--wandb-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--tensorboard', action='store_true')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir or Path('runs') / f'eval-{time.strftime("%m%d-%H%M%S")}'

    rows = (
        list(BASELINE_MATRIX)
        if args.baseline_matrix
        else [(args.planner, args.world_model)]
    )

    device = args.device or default_device()
    torch_setup = configure_torch(device, detect_anomaly=False, threads=args.threads, seed=args.seed)

    with RunLogger(
        run_dir,
        name=f'eval-{args.scenario}-s{args.seed}',
        config=vars(args),
        wandb_mode=args.wandb_mode,
        tensorboard=args.tensorboard,
        group=f'eval-{args.scenario}',
    ) as logger:
        logger.update_manifest(
            phase='evaluation', rows=[list(row) for row in rows], torch_setup=torch_setup
        )

        summaries: List[Dict[str, Any]] = []
        for planner_spec, world_model_spec in rows:
            if world_model_spec == 'dima' and args.checkpoint is None:
                log('WARN', f'skipping {planner_spec}/dima: no --checkpoint given.')
                continue
            print()
            log('RUN', f'--- {planner_spec} / {world_model_spec} ---')
            summary = evaluate(
                planner_spec=planner_spec,
                world_model_spec=world_model_spec,
                scenario=args.scenario,
                episodes=args.episodes,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
                discrete_levels=args.discrete_levels,
                checkpoint=args.checkpoint,
                device=device,
                horizon=args.horizon,
                num_samples=args.num_samples,
                logger=logger,
                diagnostics=args.diagnostics,
                diagnostic_states=args.diagnostic_states,
                sensitivity_samples=args.sensitivity_samples,
                cem_samples=args.cem_samples,
                cem_iterations=args.cem_iterations,
                cem_elite_frac=args.cem_elite_frac,
            )
            summaries.append(summary)
            logger.records('summaries', [summary])
            logger.scalars(
                {
                    'coverage': summary['mean_coverage_rate_mean'],
                    'transport': summary['mean_transport_rate_mean'],
                    'normalized_target_reward': summary['normalized_target_episode_reward_mean'],
                },
                step=len(summaries),
                prefix=f'{planner_spec}/{world_model_spec}/',
            )

        logger.update_manifest(summaries=summaries)
        _print_matrix(summaries)

    log('RUN', f'results written to {run_dir}')
    return 0


def _print_matrix(summaries: List[Dict[str, Any]]) -> None:
    if not summaries:
        return
    print()
    log('EVAL', 'baseline matrix (identical scenario, seeds, episode limit and discretisation)')
    header = (
        f'  {"planner":<20}{"world model":<16}{"coverage":>12}{"transport":>12}'
        f'{"norm target R":>15}{"return":>10}'
    )
    print(header)
    print('  ' + '-' * (len(header) - 2))
    for summary in summaries:
        print(
            f'  {summary["planner"]:<20}{summary["world_model"]:<16}'
            f'{summary["mean_coverage_rate_mean"] * 100:>11.2f}%'
            f'{summary["mean_transport_rate_mean"] * 100:>11.2f}%'
            f'{summary["normalized_target_episode_reward_mean"]:>+15.5f}'
            f'{summary["camera_team_return_mean"]:>+10.2f}'
        )


if __name__ == '__main__':
    sys.exit(main())
