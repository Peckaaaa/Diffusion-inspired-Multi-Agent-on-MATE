"""Online DIMA training on MATE: collect with the policy being trained.

    MATE episode -> DIMA rollout dict -> DreamerLearner.step -> better policy -> repeat

The offline path (:mod:`research.train_wm`) replays a dataset collected once by
fixed policies, so from the second pass onward nothing new enters the learner and
the policy never influences its own data.  This module closes that loop: every
episode after the warmup is collected by :class:`~research.planners.DIMAActorPlanner`
acting with the learner's live actor, and there is no episode budget -- it runs
until ``--max-hours``, ``--max-episodes`` or Ctrl-C.

Nothing about DIMA's control flow is changed to do this.  DIMA's own online runner
(``DreamerRunner`` + Ray ``DreamerWorker``) branches on ``Env.STARCRAFT`` /
``PETTINGZOO`` / ``GRF`` / ``MAMUJOCO`` over dict-keyed states and has no MATE
branch, whereas ``research.rollout`` already produces exactly the rollout dict
``DreamerLearner.step`` consumes.  So the loop here is the one in ``train_wm``
with the dataset replaced by a live episode.

    python -m research.train_online --run-dir runs/online-4v8 --wandb-mode online
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.collect import DEFAULT_MIX, parse_mix
from research.config import (
    CHECKPOINT_CONFIG_FILENAME,
    allow_dima_checkpoint_globals,
    build_learner_config,
    configure_torch,
    export_checkpoint_config,
)
from research.env_adapter import DEFAULT_SCENARIO, MATEEnv
from research.logging_utils import (
    RESUME_CHECKPOINT_FILENAME,
    RESUME_META_FILENAME,
    RunLogger,
    log,
    read_resume_meta,
    read_wandb_run_id,
    resolve_resume,
)
from research.planners import DIMAActorPlanner, build_planner
from research.train_wm import minimum_buffer_steps
from research.rollout import run_episode, to_dima_rollout
from research.validation import ValidationEpisode, format_trend, validate
from research.views import ObservationLayout


def _collect_episode(env, planner, *, episode: int, seed: int, max_steps: int):
    """One closed-loop episode, as both a DIMA rollout and its metrics."""

    result = run_episode(env, planner, episode=episode, seed=seed, max_steps=max_steps)
    return to_dima_rollout(result, env.n_actions), result.metrics


def train(
    run_dir: Path,
    *,
    scenario: str = DEFAULT_SCENARIO,
    seed: int = 0,
    device: Optional[str] = None,
    horizon: Optional[int] = None,
    max_episode_steps: int = 200,
    discrete_levels: int = 5,
    warmup_mix: str = DEFAULT_MIX,
    temperature: float = 1.0,
    min_buffer_size: Optional[int] = None,
    n_samples: Optional[int] = None,
    wm_epochs: Optional[int] = None,
    denoiser_steps_first_epoch: Optional[int] = None,
    remodel_steps: Optional[int] = None,
    state_decoder_batch_size: Optional[int] = None,
    denoiser_batch_size: Optional[int] = None,
    rew_end_batch_size: Optional[int] = None,
    obs_binary_loss_weight: Optional[float] = None,
    nums_obs_token: Optional[int] = None,
    obs_vocab_size: Optional[int] = None,
    num_steps_denoising: Optional[int] = None,
    agent_order: Optional[str] = None,
    save_every: int = 25,
    val_episodes: int = 20,
    eval_every: int = 50,
    sensitivity_samples: int = 4,
    detect_anomaly: bool = False,
    threads: Optional[int] = None,
    max_hours: Optional[float] = None,
    max_episodes: Optional[int] = None,
    wandb_mode: str = 'disabled',
    tensorboard: bool = False,
    resume: Optional[str] = None,
    save_buffer: bool = False,
) -> Path:
    run_dir = Path(run_dir)
    env = MATEEnv(
        scenario=scenario,
        seed=seed,
        discrete_levels=discrete_levels,
        max_episode_steps=max_episode_steps,
    )

    overrides: Dict[str, object] = {}
    if min_buffer_size is not None:
        overrides['MIN_BUFFER_SIZE'] = int(min_buffer_size)
    if n_samples is not None:
        overrides['N_SAMPLES'] = int(n_samples)
    if wm_epochs is not None:
        overrides['WM_EPOCHS'] = int(wm_epochs)
    if denoiser_steps_first_epoch is not None:
        overrides['denoiser_steps_first_epoch'] = int(denoiser_steps_first_epoch)
    if remodel_steps is not None:
        overrides['remodel_steps'] = int(remodel_steps)
        overrides['remodel_steps_first_epoch'] = int(remodel_steps)
    if state_decoder_batch_size is not None:
        overrides['state_decoder_batch_size'] = int(state_decoder_batch_size)
    if denoiser_batch_size is not None:
        overrides['denoiser_batch_size'] = int(denoiser_batch_size)
    if rew_end_batch_size is not None:
        overrides['rew_end_batch_size'] = int(rew_end_batch_size)
    if obs_binary_loss_weight is not None:
        overrides['obs_binary_loss_weight'] = float(obs_binary_loss_weight)
    if nums_obs_token is not None:
        overrides['nums_obs_token'] = int(nums_obs_token)
    if obs_vocab_size is not None:
        overrides['OBS_VOCAB_SIZE'] = int(obs_vocab_size)
    if num_steps_denoising is not None:
        overrides['num_steps_denoising'] = int(num_steps_denoising)
    if agent_order is not None:
        overrides['agent_order'] = str(agent_order)

    config = build_learner_config(
        env,
        seed=seed,
        run_dir=str(run_dir),
        device=device,
        horizon=horizon,
        # Not a flag.  Turning actor-critic training off sets EPOCHS and
        # ac_steps_first_epoch to 0 (research/config.py), and DIMA gates the
        # actor-critic loop on train_count > 9 (DreamerLearner.py:564).  Either
        # way the actor would never learn, and an "online" run would collect with
        # a permanently random policy while looking perfectly healthy.
        train_actor_critic=True,
        overrides=overrides,
    )
    # DreamerMemory.sample_indices needs batch_size * sequence_length steps, which
    # for a raised rew_end_batch_size can exceed MIN_BUFFER_SIZE.  Warmup ends on
    # MIN_BUFFER_SIZE, so without this the first training round raises
    # 'Not enough data in buffer' and kills the run.  train_wm.py:207 does the same.
    required = minimum_buffer_steps(
        config.horizon,
        state_decoder_batch_size=config.state_decoder_batch_size,
        rew_end_batch_size=config.rew_end_batch_size,
    )
    if config.MIN_BUFFER_SIZE < required:
        log(
            'WARN',
            f'MIN_BUFFER_SIZE={config.MIN_BUFFER_SIZE} is below the {required} steps '
            f'DreamerLearner.step needs at horizon={config.horizon}; raising it to {required}.',
        )
        config.MIN_BUFFER_SIZE = required

    allow_dima_checkpoint_globals()
    torch_setup = configure_torch(
        config.DEVICE, detect_anomaly=detect_anomaly, threads=threads, seed=seed
    )

    log('ENV', env.describe())
    if config.DEVICE.startswith('cuda'):
        log(
            'ONLINE',
            f'GPU {torch_setup.get("gpu_name")} x{torch_setup.get("gpu_count")} '
            f'cc{torch_setup.get("gpu_capability")} {torch_setup.get("gpu_total_memory_gb")} GB',
        )
    else:
        log('WARN', f'running on {config.DEVICE}; this path is for testing, not for real training')

    resume_checkpoint = resolve_resume(resume, run_dir)
    if resume_checkpoint is not None:
        resume_run_id = read_resume_meta(resume_checkpoint).get('wandb_run_id')
    else:
        resume_run_id = read_wandb_run_id(run_dir)

    with RunLogger(
        run_dir,
        name=f'online-{scenario}-s{seed}',
        config=config.to_dict(),
        wandb_mode=wandb_mode,
        tensorboard=tensorboard,
        group=f'online-{scenario}',
        resume_run_id=resume_run_id if resume_checkpoint is not None else None,
    ) as logger:
        logger.update_manifest(
            phase='online_training',
            environment=env.metadata(),
            warmup_mix=warmup_mix,
            torch_setup=torch_setup,
            max_episodes=max_episodes,
            max_hours=max_hours,
        )
        logger.tee_dima_scalars()

        from agent.learners.DreamerLearner import DreamerLearner

        learner = DreamerLearner(config)
        learner.tqdm_vis = sys.stdout.isatty()
        configure_torch(config.DEVICE, detect_anomaly=detect_anomaly, threads=threads)
        log(
            'ONLINE',
            f'MIN_BUFFER_SIZE={config.MIN_BUFFER_SIZE} N_SAMPLES={config.N_SAMPLES} '
            f'WM_EPOCHS={config.WM_EPOCHS} horizon={config.horizon}',
        )
        log(
            'ONLINE',
            f'batch sizes: state_decoder={config.state_decoder_batch_size} '
            f'denoiser={config.denoiser_batch_size} rew_end={config.rew_end_batch_size}',
        )

        checkpoint = run_dir / 'ckpt' / 'model_final.pth'
        resume_path = run_dir / 'ckpt' / RESUME_CHECKPOINT_FILENAME
        resume_meta_path = run_dir / 'ckpt' / RESUME_META_FILENAME
        sidecar = export_checkpoint_config(config, run_dir / 'ckpt' / CHECKPOINT_CONFIG_FILENAME)

        rng = np.random.default_rng(seed)
        started = time.time()
        episode_index = 0
        fed_steps = 0
        elapsed_before_resume = 0.0
        validation_history: List[Dict[str, object]] = []
        coverage_history: List[float] = []

        if resume_checkpoint is not None:
            loop_state = learner.load_full(str(resume_checkpoint))
            if loop_state is not None:
                episode_index = int(loop_state['next_episode'])
                fed_steps = int(loop_state['fed_steps'])
                elapsed_before_resume = float(loop_state['elapsed_seconds'])
                validation_history = list(loop_state['validation_history'])
                coverage_history = list(loop_state['coverage_history'])
                rng.bit_generator.state = loop_state['rng_state']
                log(
                    'ONLINE',
                    f'resumed from {resume_checkpoint}: episode {episode_index}, '
                    f'fed_steps={fed_steps}, train_count={learner.train_count}, '
                    f'buffer={learner.replay_buffer.num_steps} steps',
                )
                if learner.replay_buffer.num_steps < config.MIN_BUFFER_SIZE:
                    log(
                        'WARN',
                        'the checkpoint carries no replay buffer, so the warmup mix collects '
                        'again before the actor takes back over. Pass --save-buffer to avoid this.',
                    )

        # ---- held-out validation set -------------------------------------- #
        # Collected once with a fixed policy and fixed seeds and never trained on,
        # so the same states are scored every time and the trend measures the
        # model rather than the sampling (see research/validation.py).
        layout = ObservationLayout.from_env_metadata(env.metadata())
        validation_set: List[ValidationEpisode] = []
        if val_episodes > 0:
            log('ONLINE', f'collecting {val_episodes} held-out validation episodes')
            val_planner = build_planner('reactive_greedy', env, seed=seed + 7919)
            for i in range(val_episodes):
                rollout, _ = _collect_episode(
                    env, val_planner, episode=i, seed=seed + 500_000 + i, max_steps=max_episode_steps
                )
                validation_set.append(ValidationEpisode.from_rollout(rollout, env))

        world_model = None

        def run_validation(tag: int) -> None:
            nonlocal world_model
            if not validation_set or learner.train_count == 0:
                return
            if world_model is None:
                from research.world_model import DIMAWorldModel

                world_model = DIMAWorldModel.from_learner(learner, env, config)

            row = validate(
                world_model,
                validation_set,
                layout,
                num_agents=env.n_agents,
                num_actions=env.n_actions,
                sensitivity_samples=sensitivity_samples,
                seed=seed,
            )
            row['episode'] = tag
            row['train_count'] = learner.train_count
            row['fed_steps'] = fed_steps
            validation_history.append(row)
            logger.records('wm_validation', [row])
            logger.scalars(
                {k: v for k, v in row.items() if isinstance(v, (int, float)) and v is not None},
                step=tag,
                prefix='val/',
            )
            log(
                'ONLINE-DIAG',
                format_trend(
                    validation_history,
                    ('ade_h1', 'sighting_recall_h1', 'mae_h1', 'sensitivity_ratio'),
                ),
            )

        def save_resumable(next_episode: int) -> None:
            learner.save_full(
                resume_path,
                runner_state={
                    'next_episode': next_episode,
                    'fed_steps': fed_steps,
                    'elapsed_seconds': elapsed_before_resume + time.time() - started,
                    'validation_history': validation_history,
                    'coverage_history': coverage_history,
                    'rng_state': rng.bit_generator.state,
                },
                save_buffer=save_buffer,
            )
            resume_meta_path.write_text(
                json.dumps(
                    {
                        'wandb_run_id': getattr(logger.wandb_run, 'id', None),
                        'next_episode': next_episode,
                    }
                ),
                encoding='utf-8',
            )
            logger.log_checkpoint(resume_path, resume_meta_path, sidecar, aliases=['latest'])

        # Ctrl-C on a rented box must not be data loss: stop after the episode in
        # flight, then fall through to the same save the loop exit uses.
        interrupted = {'flag': False}

        def _on_sigint(signum, frame):  # noqa: ARG001
            if interrupted['flag']:
                raise KeyboardInterrupt
            interrupted['flag'] = True
            log('WARN', 'interrupt received; finishing this episode then saving. Ctrl-C again to abort.')

        previous_sigint = signal.signal(signal.SIGINT, _on_sigint)

        # ---- warmup + online loop ----------------------------------------- #
        policies = parse_mix(warmup_mix)
        warmup_names = [name for name, _ in policies]
        warmup_weights = np.array([weight for _, weight in policies], dtype=np.float64)
        warmup_planners = {name: build_planner(name, env, seed=seed) for name in warmup_names}
        actor_planner = DIMAActorPlanner(env, learner.actor, seed=seed, temperature=temperature)

        deadline = started + max_hours * 3600.0 - elapsed_before_resume if max_hours else None
        stop_reason = 'max-episodes'

        try:
            while True:
                if max_episodes is not None and episode_index >= max_episodes:
                    break
                if deadline is not None and time.time() > deadline:
                    stop_reason = 'max-hours'
                    log('WARN', f'--max-hours {max_hours} reached; stopping cleanly and saving.')
                    break
                if interrupted['flag']:
                    stop_reason = 'interrupt'
                    break

                # The buffer has to hold MIN_BUFFER_SIZE steps before DreamerLearner
                # trains at all, and an untrained actor would fill it with noise;
                # the warmup mix is the same one the offline dataset is built from.
                warming_up = learner.replay_buffer.num_steps < config.MIN_BUFFER_SIZE
                if warming_up:
                    name = warmup_names[int(rng.choice(len(warmup_names), p=warmup_weights))]
                    planner = warmup_planners[name]
                else:
                    name = actor_planner.name
                    planner = actor_planner

                rollout, metrics = _collect_episode(
                    env,
                    planner,
                    episode=episode_index,
                    seed=seed * 100_000 + episode_index,
                    max_steps=max_episode_steps,
                )
                learner.step(rollout)

                fed_steps += len(rollout['done'])
                coverage = float(metrics['mean_coverage_rate'])
                coverage_history.append(coverage)
                episode_index += 1

                logger.scalars(
                    {
                        'coverage_rate': coverage,
                        'final_coverage_rate': float(metrics['final_coverage_rate']),
                        'fed_steps': fed_steps,
                        'buffer_steps': learner.replay_buffer.num_steps,
                        'train_count': learner.train_count,
                        'warmup': float(warming_up),
                    },
                    step=episode_index,
                    prefix='online/',
                )

                if episode_index % 10 == 0 or warming_up:
                    recent = coverage_history[-25:]
                    log(
                        'ONLINE',
                        f'episode {episode_index} policy={name} '
                        f'coverage={coverage:.3f} (last{len(recent)}={np.mean(recent):.3f}) '
                        f'fed_steps={fed_steps} buffer={learner.replay_buffer.num_steps} '
                        f'train_count={learner.train_count} '
                        f'elapsed={elapsed_before_resume + time.time() - started:.0f}s',
                    )

                if eval_every and episode_index % eval_every == 0:
                    run_validation(episode_index)

                if save_every and episode_index % save_every == 0:
                    path = run_dir / 'ckpt' / f'model_ep{episode_index:06d}.pth'
                    learner.save(str(path))
                    logger.log_checkpoint(path, sidecar, aliases=[f'ep{episode_index:06d}'])
                    save_resumable(episode_index)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)

        learner.save(str(checkpoint))
        save_resumable(episode_index)
        logger.log_checkpoint(checkpoint, sidecar, aliases=['final'])
        log('ONLINE', f'stopped after {episode_index} episodes ({stop_reason}); {checkpoint}')

        logger.update_manifest(
            checkpoint=str(checkpoint),
            episodes=episode_index,
            fed_steps=fed_steps,
            train_count=learner.train_count,
            buffer_steps=learner.replay_buffer.num_steps,
            validation=validation_history,
            mean_coverage_last_50=float(np.mean(coverage_history[-50:])) if coverage_history else None,
            stop_reason=stop_reason,
            wall_seconds=round(elapsed_before_resume + time.time() - started, 1),
        )

    env.close()
    return checkpoint


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.train_online', description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--scenario', default=DEFAULT_SCENARIO)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--max-episode-steps', type=int, default=200)
    parser.add_argument('--discrete-levels', type=int, default=5)
    parser.add_argument('--warmup-mix', default=DEFAULT_MIX,
                        help='policies that fill the buffer before the actor takes over; '
                             f'default: {DEFAULT_MIX}')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='softmax temperature of the acting policy; >1 explores more')
    parser.add_argument('--min-buffer-size', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--wm-epochs', type=int, default=None)
    parser.add_argument('--denoiser-steps-first-epoch', type=int, default=None)
    parser.add_argument('--remodel-steps', type=int, default=None)
    parser.add_argument('--state-decoder-batch-size', type=int, default=None)
    parser.add_argument('--denoiser-batch-size', type=int, default=None)
    parser.add_argument('--rew-end-batch-size', type=int, default=None)
    parser.add_argument('--save-every', type=int, default=25, help='episodes between checkpoints')
    parser.add_argument('--val-episodes', type=int, default=20,
                        help='held-out episodes collected once for validation (0 disables)')
    parser.add_argument('--eval-every', type=int, default=50, help='validate every N episodes')
    parser.add_argument('--sensitivity-samples', type=int, default=4)
    parser.add_argument('--detect-anomaly', action='store_true')
    parser.add_argument('--threads', type=int, default=None)
    parser.add_argument('--max-hours', type=float, default=None,
                        help='stop cleanly and save before this wall-clock budget runs out')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='stop after this many episodes; omit to run until stopped')
    parser.add_argument('--wandb-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--obs-binary-loss-weight', type=float, default=None,
                        help='weight on the 0/1 flag block of the state decoder loss '
                             '(default 1.0)')
    parser.add_argument('--nums-obs-token', type=int, default=None,
                        help='VQ tokens the state is compressed into (default 12)')
    parser.add_argument('--obs-vocab-size', type=int, default=None,
                        help='VQ codebook size (default 128)')
    parser.add_argument('--num-steps-denoising', type=int, default=None,
                        help='diffusion steps per transition; must be a multiple of the agent '
                             'count (default: one step per agent)')
    parser.add_argument('--agent-order', default=None,
                        choices=['default', 'reverse', 'random', 'tiled'],
                        help="order actions condition the denoising steps in; 'tiled' gives every "
                             'agent a low-sigma slot instead of one sigma band each')
    parser.add_argument('--resume', default=None,
                        help='continue from ckpt/latest.pth: the file, its run directory, '
                             'or wandb://entity/project/artifact:alias')
    parser.add_argument('--save-buffer', action='store_true',
                        help='include the replay buffer in resumable checkpoints')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    train(
        args.run_dir,
        scenario=args.scenario,
        seed=args.seed,
        device=args.device,
        horizon=args.horizon,
        max_episode_steps=args.max_episode_steps,
        discrete_levels=args.discrete_levels,
        warmup_mix=args.warmup_mix,
        temperature=args.temperature,
        min_buffer_size=args.min_buffer_size,
        n_samples=args.n_samples,
        wm_epochs=args.wm_epochs,
        denoiser_steps_first_epoch=args.denoiser_steps_first_epoch,
        remodel_steps=args.remodel_steps,
        state_decoder_batch_size=args.state_decoder_batch_size,
        denoiser_batch_size=args.denoiser_batch_size,
        rew_end_batch_size=args.rew_end_batch_size,
        obs_binary_loss_weight=args.obs_binary_loss_weight,
        nums_obs_token=args.nums_obs_token,
        obs_vocab_size=args.obs_vocab_size,
        num_steps_denoising=args.num_steps_denoising,
        agent_order=args.agent_order,
        save_every=args.save_every,
        val_episodes=args.val_episodes,
        eval_every=args.eval_every,
        sensitivity_samples=args.sensitivity_samples,
        detect_anomaly=args.detect_anomaly,
        threads=args.threads,
        max_hours=args.max_hours,
        max_episodes=args.max_episodes,
        wandb_mode=args.wandb_mode,
        tensorboard=args.tensorboard,
        resume=args.resume,
        save_buffer=args.save_buffer,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
