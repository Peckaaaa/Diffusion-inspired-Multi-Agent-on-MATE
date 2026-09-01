"""One command that runs the whole DIMA x MATE experiment on a server.

    python -m research.pipeline --preset server

That single call does, in order, into one timestamped run directory:

    collect  ->  research.collect.collect      (MATE trajectories, mixed policies)
    train    ->  research.train_wm.train       (DIMA's DreamerLearner, DIMA's schedule)
    evaluate ->  research.evaluate.main        (the full baseline matrix + diagnostics)

Nothing new is implemented here.  This module owns *only* the ordering, the
directory layout and the preset sizes; every stage is the existing entry point,
called with the same arguments the README documents, so a stage run by hand and
the same stage run by the pipeline are the same code path.

Layout
------
::

    <out-root>/<tag>/
        pipeline.json          what was asked for, what ran, how long, what it produced
        dataset/               episode_*.npz + manifest.json   (research.collect)
        wm/                    ckpt/, wm_validation.jsonl, dima_scalars.jsonl, manifest.json
        eval/                  summaries.jsonl, diagnostics, manifest.json

Resuming
--------
A stage whose output already exists is skipped, so re-running the same command
after a preemption continues instead of starting over.  ``--force`` re-runs
everything; ``--stages`` runs a subset.

Presets
-------
``--preset`` sets sizes only, and every one of them is still overridable by the
matching flag.

* ``smoke``  -- minutes, on a laptop.  Proves the three stages wire together.
* ``laptop`` -- the README's quick start: one pass, shortened DIMA schedule.
* ``server`` -- DIMA's own unshortened schedule, 20 passes, GPU expected.

The scenario defaults to ``MATE-4v8-9`` -- 4 cameras, 8 targets, 9 obstacles
(``obs_dim=126``, ``state_dim=220``, ``|A|=25``).  That is deliberately *not*
``research.env_adapter.DEFAULT_SCENARIO`` (``MATE-4v2-9``), which stays where it
is because the verified numbers in the README were measured on it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import research  # noqa: F401 - installs sys.path + compat shims

from research import collect as collect_module
from research import evaluate as evaluate_module
from research import train_wm
from research.collect import DEFAULT_MIX
from research.logging_utils import dependency_versions, git_provenance, log


__all__ = ['PIPELINE_SCENARIO', 'PRESETS', 'STAGES', 'run_pipeline', 'main']


#: The scenario this experiment runs on.
PIPELINE_SCENARIO = 'MATE-4v8-9'

STAGES = ('collect', 'train', 'evaluate')

#: Preset sizes.  ``None`` means "leave DIMA's own default alone" -- that is what
#: makes ``server`` a faithful run rather than a shortened one.
PRESETS: Dict[str, Dict[str, Any]] = {
    'smoke': dict(
        episodes=12,
        max_episode_steps=100,
        passes=1,
        horizon=5,
        val_episodes=2,
        n_samples=200,
        wm_epochs=4,
        denoiser_steps_first_epoch=4,
        remodel_steps=4,
        sensitivity_samples=2,
        # Kept tiny on purpose: the matrix's `oracle` row forks the real MATE
        # environment once per camera per candidate action, so its cost is
        # episodes x steps x 4 x 25 environment steps and it dominates this stage.
        eval_episodes=1,
        eval_max_episode_steps=25,
        diagnostic_states=2,
    ),
    'laptop': dict(
        episodes=300,
        max_episode_steps=200,
        passes=1,
        horizon=5,
        val_episodes=20,
        n_samples=2000,
        wm_epochs=40,
        denoiser_steps_first_epoch=40,
        remodel_steps=20,
        sensitivity_samples=4,
        eval_episodes=10,
        eval_max_episode_steps=150,
        diagnostic_states=4,
    ),
    'server': dict(
        episodes=400,
        max_episode_steps=200,
        passes=20,
        horizon=5,
        val_episodes=20,
        # None -> DIMA's DreamerLearnerConfig values: N_SAMPLES=100, WM_EPOCHS=200,
        # denoiser_steps_first_epoch=200, remodel_steps=60.  Nothing is shortened.
        n_samples=None,
        wm_epochs=None,
        denoiser_steps_first_epoch=None,
        remodel_steps=None,
        sensitivity_samples=8,
        # 10 x 150 is the protocol the README's published baseline matrix used, so
        # a server run stays comparable to it.  It is also as much as the `oracle`
        # row can afford: that row forks the real MATE environment 4 x 25 times per
        # step, on the CPU, and no GPU makes it faster.  Raise it only if you are
        # prepared for the evaluate stage to outlast the training stage.
        eval_episodes=10,
        eval_max_episode_steps=150,
        diagnostic_states=4,
    ),
}

#: Preset keys a CLI flag of the same name may override.
_OVERRIDE_FIELDS = (
    'episodes',
    'max_episode_steps',
    'passes',
    'horizon',
    'val_episodes',
    'min_buffer_size',
    'n_samples',
    'wm_epochs',
    'denoiser_steps_first_epoch',
    'remodel_steps',
    'sensitivity_samples',
    'eval_episodes',
    'eval_max_episode_steps',
    'diagnostic_states',
)


#: Peak VRAM one training process needs, in GB.  Measured shape rather than a
#: guess: on MATE-4v8-9 at horizon 5 the three trained modules are 23.2 M
#: parameters, and DreamerLearner.step's batches are fixed at 256x1 / 64x6 /
#: 128xhorizon, so weights + gradients + Adam state + activations stay in a few
#: hundred MB.  4 GB leaves room for the CUDA context and fragmentation.
VRAM_PER_JOB_GB = 4.0


def _preflight(device: Optional[str], preset: str) -> Dict[str, Any]:
    """Report the device before anything expensive starts.

    A server run that silently lands on CPU is the failure this catches: it does
    not crash, it just takes weeks.  The two GPU facts worth knowing up front are
    whether TF32 is actually available -- it needs compute capability 8.0, so
    Ampere or later; on an RTX card that is the 30-series onwards -- and how many
    of these runs fit on the card at once, because one does not fill it.
    """

    import torch

    from research.config import default_device

    resolved = device or default_device()
    info: Dict[str, Any] = {'device': resolved, 'cuda_available': torch.cuda.is_available()}
    if resolved.startswith('cuda') and torch.cuda.is_available():
        index = torch.cuda.current_device() if ':' not in resolved else int(resolved.split(':')[1])
        properties = torch.cuda.get_device_properties(index)
        major, minor = torch.cuda.get_device_capability(index)
        memory_gb = round(properties.total_memory / 1024**3, 2)
        concurrent_jobs = max(1, int(memory_gb // VRAM_PER_JOB_GB))
        info.update(
            gpu_index=index,
            gpu_name=torch.cuda.get_device_name(index),
            gpu_count=torch.cuda.device_count(),
            gpu_capability=f'{major}.{minor}',
            gpu_total_memory_gb=memory_gb,
            tf32_supported=major >= 8,
            concurrent_jobs_that_fit=concurrent_jobs,
        )
        log(
            'RUN',
            f'device={resolved} {info["gpu_name"]} x{info["gpu_count"]} '
            f'cc{info["gpu_capability"]} {memory_gb} GB',
        )
        if major < 8:
            log(
                'WARN',
                f'compute capability {major}.{minor} is pre-Ampere, so TF32 is unavailable and '
                f'configure_torch\'s tf32 flags do nothing. An RTX 30-series or newer card is '
                f'where the free matmul speedup starts.',
            )
        if concurrent_jobs > 1:
            log(
                'RUN',
                f'one training process needs about {VRAM_PER_JOB_GB:.0f} GB, so roughly '
                f'{concurrent_jobs} of these fit on this card at once -- vary --seed and --tag, '
                f'and set --threads so they do not fight over CPU cores.',
            )
    else:
        log('RUN', f'device={resolved}')
        if preset == 'server':
            log(
                'WARN',
                'preset "server" selected but no CUDA device is visible -- this run will be '
                'orders of magnitude slower. Check CUDA_VISIBLE_DEVICES and the torch build.',
            )
    return info


def _free_gpu() -> None:
    """Hand a finished stage's allocator blocks back before the next one starts."""

    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_pipeline(
    *,
    out_root: Path,
    tag: Optional[str] = None,
    preset: str = 'server',
    stages: Sequence[str] = STAGES,
    scenario: str = PIPELINE_SCENARIO,
    seed: int = 0,
    discrete_levels: int = 5,
    mix: str = DEFAULT_MIX,
    device: Optional[str] = None,
    threads: Optional[int] = None,
    max_hours: Optional[float] = None,
    wandb_mode: str = 'disabled',
    tensorboard: bool = False,
    force: bool = False,
    dry_run: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if preset not in PRESETS:
        raise ValueError(f'Unknown preset {preset!r}. Known: {", ".join(PRESETS)}')
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f'Unknown stage(s) {unknown}. Known: {", ".join(STAGES)}')

    size = dict(PRESETS[preset])
    size.setdefault('min_buffer_size', None)
    size.update({k: v for k, v in (overrides or {}).items() if v is not None})

    tag = tag or f'{scenario}-{preset}-s{seed}-{time.strftime("%Y%m%d-%H%M%S")}'
    run_root = Path(out_root) / tag
    dataset_dir = run_root / 'dataset'
    wm_dir = run_root / 'wm'
    eval_dir = run_root / 'eval'
    checkpoint = wm_dir / 'ckpt' / 'model_final.pth'

    # DreamerLearner.step will not train at all until the buffer holds this many
    # steps, so a preset small enough to fall under it must lower MIN_BUFFER_SIZE
    # rather than fail deep inside DIMA.
    required_buffer = train_wm.minimum_buffer_steps(size['horizon'])
    train_steps = (size['episodes'] - size['val_episodes']) * size['max_episode_steps']
    min_buffer_size = size['min_buffer_size']
    if min_buffer_size is None and train_steps < 5000:
        min_buffer_size = required_buffer

    log('RUN', f'pipeline preset={preset} scenario={scenario} seed={seed} -> {run_root}')
    log(
        'RUN',
        f'plan: collect {size["episodes"]}x{size["max_episode_steps"]} steps '
        f'({size["val_episodes"]} held out) | train {size["passes"]} passes '
        f'horizon={size["horizon"]} | evaluate {size["eval_episodes"]} episodes '
        f'x{size["eval_max_episode_steps"]} steps',
    )
    if train_steps < required_buffer:
        raise ValueError(
            f'{train_steps} training steps is below the {required_buffer} '
            f'DreamerLearner.step needs at horizon={size["horizon"]}. '
            f'Raise --episodes or --max-episode-steps, or lower --horizon.'
        )

    device_info = _preflight(device, preset)

    record: Dict[str, Any] = {
        'tag': tag,
        'preset': preset,
        'scenario': scenario,
        'seed': seed,
        'discrete_levels': discrete_levels,
        'policy_mix': mix,
        'stages_requested': list(stages),
        'sizes': size,
        'min_buffer_size': min_buffer_size,
        'device': device_info,
        'threads': threads,
        'max_hours': max_hours,
        'paths': {
            'dataset': str(dataset_dir),
            'world_model': str(wm_dir),
            'evaluation': str(eval_dir),
            'checkpoint': str(checkpoint),
        },
        'provenance': git_provenance(),
        'dependencies': dependency_versions(),
        'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'stages': {},
    }

    if dry_run:
        log('RUN', 'dry run: nothing executed.')
        print(json.dumps(record, indent=2))
        return record

    run_root.mkdir(parents=True, exist_ok=True)

    def write_record() -> None:
        (run_root / 'pipeline.json').write_text(json.dumps(record, indent=2), encoding='utf-8')

    write_record()

    def stage(name: str, already_done: bool, run) -> None:
        if name not in stages:
            record['stages'][name] = {'status': 'not requested'}
            write_record()
            return
        if already_done and not force:
            log('RUN', f'stage {name}: output already present, skipping (--force to re-run)')
            record['stages'][name] = {'status': 'skipped'}
            write_record()
            return
        log('RUN', f'--- stage {name} ---')
        started = time.time()
        result = run()
        record['stages'][name] = {
            'status': 'ok',
            'wall_seconds': round(time.time() - started, 1),
            'result': result,
        }
        write_record()
        _free_gpu()

    stage(
        'collect',
        (dataset_dir / 'manifest.json').is_file(),
        lambda: collect_module.collect(
            dataset_dir,
            scenario=scenario,
            episodes=size['episodes'],
            max_episode_steps=size['max_episode_steps'],
            discrete_levels=discrete_levels,
            seed=seed,
            mix=mix,
        ),
    )

    stage(
        'train',
        checkpoint.is_file(),
        lambda: str(
            train_wm.train(
                dataset_dir,
                wm_dir,
                passes=size['passes'],
                seed=seed,
                device=device,
                horizon=size['horizon'],
                min_buffer_size=min_buffer_size,
                n_samples=size['n_samples'],
                wm_epochs=size['wm_epochs'],
                denoiser_steps_first_epoch=size['denoiser_steps_first_epoch'],
                remodel_steps=size['remodel_steps'],
                val_episodes=size['val_episodes'],
                sensitivity_samples=size['sensitivity_samples'],
                threads=threads,
                max_hours=max_hours,
                wandb_mode=wandb_mode,
                tensorboard=tensorboard,
            )
        ),
    )

    def run_evaluate() -> int:
        argv: List[str] = [
            '--baseline-matrix',
            '--scenario', scenario,
            '--episodes', str(size['eval_episodes']),
            '--seed', str(seed),
            '--max-episode-steps', str(size['eval_max_episode_steps']),
            '--discrete-levels', str(discrete_levels),
            '--diagnostics',
            '--diagnostic-states', str(size['diagnostic_states']),
            '--sensitivity-samples', str(size['sensitivity_samples']),
            '--run-dir', str(eval_dir),
            '--wandb-mode', wandb_mode,
        ]
        if checkpoint.is_file():
            argv += ['--checkpoint', str(checkpoint)]
        else:
            # evaluate.py already skips the dima row without a checkpoint, but say
            # so here as well: a matrix quietly missing its only model-based row
            # is easy to misread as a result.
            log('WARN', f'no checkpoint at {checkpoint}; the dima row will be skipped.')
        if device is not None:
            argv += ['--device', device]
        if threads is not None:
            argv += ['--threads', str(threads)]
        if tensorboard:
            argv += ['--tensorboard']
        return evaluate_module.main(argv)

    stage('evaluate', (eval_dir / 'manifest.json').is_file(), run_evaluate)

    record['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    write_record()
    log('RUN', f'pipeline complete -> {run_root / "pipeline.json"}')
    return record


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='python -m research.pipeline',
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--preset', default='server', choices=sorted(PRESETS))
    parser.add_argument('--out-root', type=Path, default=Path('runs'))
    parser.add_argument('--tag', default=None, help='run directory name (default: timestamped)')
    parser.add_argument(
        '--stages',
        default=','.join(STAGES),
        help='comma-separated subset of ' + ','.join(STAGES),
    )
    parser.add_argument('--scenario', default=PIPELINE_SCENARIO)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--mix', default=DEFAULT_MIX, help='default: ' + DEFAULT_MIX)
    parser.add_argument('--device', default=None,
                        help='cuda, cuda:1, cpu; default: detect')
    parser.add_argument('--threads', type=int, default=None, help='torch CPU thread count')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='wall-clock budget for the training stage; it stops cleanly and saves')
    parser.add_argument('--wandb-mode', default='disabled',
                        choices=['disabled', 'offline', 'online'])
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--force', action='store_true', help='re-run stages whose output exists')
    parser.add_argument('--dry-run', action='store_true', help='print the resolved plan and exit')

    sizes = parser.add_argument_group('preset overrides (each defaults to the preset value)')
    sizes.add_argument('--episodes', type=int, default=None)
    sizes.add_argument('--max-episode-steps', type=int, default=None)
    sizes.add_argument('--passes', type=int, default=None)
    sizes.add_argument('--horizon', type=int, default=None)
    sizes.add_argument('--val-episodes', type=int, default=None)
    sizes.add_argument('--min-buffer-size', type=int, default=None)
    sizes.add_argument('--n-samples', type=int, default=None)
    sizes.add_argument('--wm-epochs', type=int, default=None)
    sizes.add_argument('--denoiser-steps-first-epoch', type=int, default=None)
    sizes.add_argument('--remodel-steps', type=int, default=None)
    sizes.add_argument('--sensitivity-samples', type=int, default=None)
    sizes.add_argument('--eval-episodes', type=int, default=None)
    sizes.add_argument('--eval-max-episode-steps', type=int, default=None)
    sizes.add_argument('--diagnostic-states', type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_pipeline(
        out_root=args.out_root,
        tag=args.tag,
        preset=args.preset,
        stages=[s.strip() for s in args.stages.split(',') if s.strip()],
        scenario=args.scenario,
        seed=args.seed,
        discrete_levels=args.discrete_levels,
        mix=args.mix,
        device=args.device,
        threads=args.threads,
        max_hours=args.max_hours,
        wandb_mode=args.wandb_mode,
        tensorboard=args.tensorboard,
        force=args.force,
        dry_run=args.dry_run,
        overrides={field: getattr(args, field) for field in _OVERRIDE_FIELDS},
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
