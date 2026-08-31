"""Controlled action-information experiment (brief section 29).

How much action-dependent signal must a world model carry before model-based
planning is worth anything?

The same planner is run against :class:`~research.world_model.AlphaOracleWorldModel`
at a sweep of ``alpha``.  That model takes the *true* consequence of each candidate
action and attenuates the part that depends on the action::

    prediction(a) = baseline + alpha * (oracle(a) - baseline) + noise

with ``baseline`` the mean oracle outcome over the candidates being compared.  At
``alpha = 1`` the planner sees perfect information; at ``alpha = 0`` every action
looks identical and the planner degenerates to its tie-break.  The resulting curve
is the yardstick a real world model's action sensitivity has to be read against.

This is **diagnostic only**.  It forks the live environment and never produces
training data (brief section 29).

    python -m research.alpha_experiment --episodes 3 --max-episode-steps 60

Cost warning: every planner step issues ``num_cameras * num_actions`` real MATE
forks (100 for MATE-4v2-9 at levels=5), so keep the episode budget small.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.env_adapter import DEFAULT_SCENARIO, MATEEnv
from research.logging_utils import RunLogger, log
from research.planners import build_planner
from research.rollout import run_episode
from research.world_model import AlphaOracleWorldModel


DEFAULT_ALPHAS = (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00)


def run_sweep(
    *,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    scenario: str = DEFAULT_SCENARIO,
    episodes: int = 3,
    max_episode_steps: int = 60,
    discrete_levels: int = 5,
    seed: int = 0,
    noise_scale: float = 0.0,
    logger: Optional[RunLogger] = None,
    include_reference_baselines: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    specs: List[tuple] = [('alpha', a) for a in alphas]
    if include_reference_baselines:
        specs = [('baseline', 'random'), ('baseline', 'reactive_greedy')] + specs

    for kind, value in specs:
        started = time.time()
        coverages: List[float] = []
        transports: List[float] = []

        for index in range(episodes):
            env = MATEEnv(
                scenario=scenario,
                seed=seed + index,
                discrete_levels=discrete_levels,
                max_episode_steps=max_episode_steps,
            )
            if kind == 'baseline':
                world_model = None
                planner = build_planner(value, env, seed=seed + index)
            else:
                world_model = AlphaOracleWorldModel(
                    env, alpha=float(value), noise_scale=noise_scale, seed=seed + index
                )
                planner = build_planner(
                    'predictive_greedy', env, world_model=world_model, seed=seed + index, horizon=1
                )

            result = run_episode(
                env,
                planner,
                world_model=world_model,
                episode=index,
                seed=seed * 1000 + index,
                max_steps=max_episode_steps,
                reference_prediction=False,
            )
            coverages.append(result.metrics['mean_coverage_rate'])
            transports.append(result.metrics['mean_transport_rate'])
            env.close()

        label = value if kind == 'baseline' else f'alpha={float(value):.2f}'
        row = {
            'kind': kind,
            'label': label,
            'alpha': None if kind == 'baseline' else float(value),
            'coverage_mean': float(np.mean(coverages)),
            'coverage_std': float(np.std(coverages)),
            'transport_mean': float(np.mean(transports)),
            'episodes': episodes,
            'wall_seconds': round(time.time() - started, 1),
        }
        rows.append(row)
        log(
            'WM-DIAG',
            f'{label:<18} coverage={row["coverage_mean"] * 100:6.2f}%'
            f' +-{row["coverage_std"] * 100:5.2f}  transport={row["transport_mean"] * 100:6.2f}%'
            f'  ({row["wall_seconds"]:.0f}s)',
        )
        if logger is not None:
            logger.records('alpha_sweep', [row])

    return rows


def print_table(rows: Sequence[Dict[str, Any]]) -> None:
    print()
    log('EVAL', 'controlled action-information sweep (brief section 29)')
    print(f'  {"model":<18}{"coverage":>12}{"+-std":>9}{"transport":>12}')
    print('  ' + '-' * 49)
    for row in rows:
        print(
            f'  {row["label"]:<18}{row["coverage_mean"] * 100:>11.2f}%'
            f'{row["coverage_std"] * 100:>9.2f}{row["transport_mean"] * 100:>11.2f}%'
        )

    alpha_rows = [r for r in rows if r['alpha'] is not None]
    reactive = next((r for r in rows if r['label'] == 'reactive_greedy'), None)
    if alpha_rows and reactive is not None:
        better = [r for r in alpha_rows if r['coverage_mean'] >= reactive['coverage_mean']]
        print()
        if better:
            threshold = min(r['alpha'] for r in better)
            log(
                'WM-DIAG',
                f'model-based planning reaches reactive_greedy at alpha >= {threshold:.2f} '
                f'(coverage {reactive["coverage_mean"] * 100:.2f}%)',
            )
        else:
            log(
                'WM-DIAG',
                'no alpha in this sweep reached reactive_greedy; the action signal a world '
                'model would need is above the sweep range, or the episode budget is too small.',
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m research.alpha_experiment', description=__doc__
    )
    parser.add_argument(
        '--alphas',
        default=','.join(str(a) for a in DEFAULT_ALPHAS),
        help=f'comma-separated (default: {",".join(str(a) for a in DEFAULT_ALPHAS)})',
    )
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--episodes', type=int, default=3)
    parser.add_argument('--max-episode-steps', type=int, default=60)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--noise-scale', type=float, default=0.0)
    parser.add_argument('--no-baselines', action='store_true')
    parser.add_argument('--run-dir', type=Path, default=None)
    parser.add_argument('--wandb-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    alphas = [float(a) for a in args.alphas.split(',') if a.strip()]
    run_dir = args.run_dir or Path('runs') / f'alpha-{time.strftime("%m%d-%H%M%S")}'

    with RunLogger(
        run_dir,
        name=f'alpha-{args.scenario}-s{args.seed}',
        config=vars(args),
        wandb_mode=args.wandb_mode,
        group=f'alpha-{args.scenario}',
    ) as logger:
        logger.update_manifest(phase='alpha_experiment', alphas=alphas, diagnostic_only=True)
        rows = run_sweep(
            alphas=alphas,
            scenario=args.scenario,
            episodes=args.episodes,
            max_episode_steps=args.max_episode_steps,
            discrete_levels=args.discrete_levels,
            seed=args.seed,
            noise_scale=args.noise_scale,
            logger=logger,
            include_reference_baselines=not args.no_baselines,
        )
        logger.update_manifest(rows=rows)
        print_table(rows)

    log('RUN', f'results written to {run_dir}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
