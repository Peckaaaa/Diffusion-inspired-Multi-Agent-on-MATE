"""Collect MATE trajectories for DIMA (brief section 17).

Brief section 17 asks for more than random data.  The default policy mix is
therefore MATE's own rule-based camera agents plus uniform random, so the world
model sees both purposeful tracking behaviour and broad action coverage:

    reactive_greedy 0.40   MATE GreedyCameraAgent  -- purposeful, on-distribution
    heuristic       0.20   MATE HeuristicCameraAgent
    random          0.25   uniform over Discrete(levels**2) -- action coverage
    mate_random     0.15   MATE RandomCameraAgent  -- held actions, smooth motion

Each episode is stored as one ``.npz`` holding exactly the arrays
``DreamerLearner.step`` consumes, so training does no conversion at all.

    python -m research.collect --episodes 200 --out datasets/mate4v2-mixed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.env_adapter import DEFAULT_SCENARIO, MATEEnv
from research.logging_utils import dependency_versions, git_provenance, log
from research.planners import build_planner
from research.rollout import run_episode, to_dima_rollout


DEFAULT_MIX = 'reactive_greedy:0.40,heuristic:0.20,random:0.25,mate_random:0.15'

ROLLOUT_KEYS = (
    'observation',
    'shared_obs',
    'next_shared_obs',
    'action',
    'reward',
    'done',
    'fake',
    'last',
    'entropy',
)


def parse_mix(spec: str) -> List[Tuple[str, float]]:
    """``'a:0.5,b:0.5'`` -> ``[('a', 0.5), ('b', 0.5)]``, normalised."""

    entries: List[Tuple[str, float]] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, weight = chunk.partition(':')
        entries.append((name.strip(), float(weight) if weight else 1.0))
    if not entries:
        raise ValueError(f'Empty policy mix: {spec!r}')
    total = sum(w for _, w in entries)
    return [(name, weight / total) for name, weight in entries]


def collect(
    out_dir: Path,
    *,
    scenario: str = DEFAULT_SCENARIO,
    episodes: int = 100,
    max_episode_steps: int = 200,
    discrete_levels: int = 5,
    seed: int = 0,
    mix: str = DEFAULT_MIX,
    reward_coefficients: Optional[Dict[str, float]] = None,
) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    policies = parse_mix(mix)
    names = [name for name, _ in policies]
    weights = np.array([weight for _, weight in policies], dtype=np.float64)
    rng = np.random.default_rng(seed)

    env = MATEEnv(
        scenario=scenario,
        seed=seed,
        discrete_levels=discrete_levels,
        max_episode_steps=max_episode_steps,
        reward_coefficients=reward_coefficients,
    )
    log('ENV', env.describe())
    log('DATA', f'policy mix: {", ".join(f"{n}={w:.2f}" for n, w in policies)}')

    planners = {name: build_planner(name, env, seed=seed) for name in names}

    counts: Counter = Counter()
    per_policy: Dict[str, List[float]] = defaultdict(list)
    total_steps = 0
    started = time.time()

    for index in range(episodes):
        name = names[int(rng.choice(len(names), p=weights))]
        episode_seed = seed * 100_000 + index
        result = run_episode(
            env,
            planners[name],
            episode=index,
            seed=episode_seed,
            max_steps=max_episode_steps,
        )
        rollout = to_dima_rollout(result, env.n_actions)

        np.savez_compressed(
            out_dir / f'episode_{index:05d}.npz',
            policy=np.array(name),
            seed=np.array(episode_seed),
            **{key: rollout[key] for key in ROLLOUT_KEYS},
        )

        counts[name] += 1
        per_policy[name].append(result.metrics['mean_coverage_rate'])
        total_steps += len(result)

        if (index + 1) % max(1, episodes // 10) == 0 or index == episodes - 1:
            elapsed = time.time() - started
            log(
                'DATA',
                f'episode {index + 1}/{episodes}  steps={total_steps}  '
                f'{total_steps / max(elapsed, 1e-9):.0f} steps/s  last={name} '
                f'coverage={result.metrics["mean_coverage_rate"]:.3f}',
            )

    manifest = {
        'scenario': scenario,
        'environment': env.metadata(),
        'episodes': episodes,
        'total_steps': total_steps,
        'max_episode_steps': max_episode_steps,
        'discrete_levels': discrete_levels,
        'seed': seed,
        'policy_mix': dict(policies),
        'episodes_per_policy': dict(counts),
        'mean_coverage_per_policy': {k: float(np.mean(v)) for k, v in per_policy.items()},
        'rollout_keys': list(ROLLOUT_KEYS),
        'provenance': git_provenance(),
        'dependencies': dependency_versions(),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    log('DATA', f'wrote {episodes} episodes / {total_steps} steps to {out_dir}')
    for name in names:
        if counts[name]:
            log(
                'DATA',
                f'  {name:<16} episodes={counts[name]:<4} '
                f'mean_coverage={np.mean(per_policy[name]):.3f}',
            )
    env.close()
    return manifest


def load_dataset(path: Path) -> Tuple[Dict, List[Dict[str, np.ndarray]]]:
    """Read a collected dataset back into ``DreamerLearner.step`` rollouts."""

    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    rollouts: List[Dict[str, np.ndarray]] = []
    for file in sorted(path.glob('episode_*.npz')):
        with np.load(file) as data:
            rollouts.append({key: data[key] for key in manifest['rollout_keys']})
    return manifest, rollouts


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.collect', description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='output dataset directory')
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mix', default=DEFAULT_MIX, help=f'default: {DEFAULT_MIX}')
    parser.add_argument(
        '--reward-coefficients',
        default=None,
        help=(
            'JSON dict activating MATE\'s AuxiliaryCameraRewards, e.g. '
            '\'{"coverage_rate": 1.0}\'. Omit to keep MATE\'s raw camera-team reward.'
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    coefficients = json.loads(args.reward_coefficients) if args.reward_coefficients else None
    collect(
        args.out,
        scenario=args.scenario,
        episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
        discrete_levels=args.discrete_levels,
        seed=args.seed,
        mix=args.mix,
        reward_coefficients=coefficients,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
