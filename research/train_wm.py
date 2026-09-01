"""Train DIMA's world model on MATE trajectories (brief section 18).

The DIMA architecture is not touched.  This module is a *runner*: it replays
collected MATE episodes through ``DreamerLearner.step``, which is the same entry
point ``DreamerRunner`` uses online, so the state decoder, the diffusion denoiser
and the reward/termination transformer are trained by DIMA's own code, with
DIMA's own schedule, losses, optimisers and logging.

    MATE trajectories -> DIMA rollout dicts -> DreamerLearner.step -> checkpoint

Why a runner instead of ``DIMA/train.py``
-----------------------------------------
``train.py`` imports ``configs.EnvConfigs``, whose module-level imports require
SMAC, SMACv2, PettingZoo+SuperSuit, Google Research Football *and* MuJoCo to be
installed, and it drives collection through Ray workers that would each need a
MATE environment.  Replaying pre-collected episodes needs none of that, keeps the
data source explicit (brief section 17 asks for more than random trajectories),
and makes training reproducible from a fixed dataset.

Actor-critic training is off by default: the research pipeline replaces DIMA's
actor with a planner.  It is disabled through configuration
(``EPOCHS = ac_steps_first_epoch = 0``), not by editing DIMA -- pass
``--train-actor-critic`` to restore DIMA's full behaviour.

    python -m research.train_wm --dataset datasets/mate4v2-mixed \\
        --run-dir runs/wm-4v2 --passes 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.collect import load_dataset
from research.config import (
    CHECKPOINT_CONFIG_FILENAME,
    MATEDreamerLearnerConfig,
    allow_dima_checkpoint_globals,
    build_learner_config,
    configure_torch,
    export_checkpoint_config,
)
from research.env_adapter import MATEEnv
from research.logging_utils import (
    RESUME_CHECKPOINT_FILENAME,
    RESUME_META_FILENAME,
    RunLogger,
    log,
    read_resume_meta,
    read_wandb_run_id,
    resolve_resume,
)
from research.validation import ValidationEpisode, format_trend, validate
from research.views import ObservationLayout


def build_env_from_manifest(manifest: Dict, *, seed: int = 0) -> MATEEnv:
    """Recreate the environment the dataset was collected in.

    Only its *metadata* is used -- dimensions, action count, agent count -- so
    that DIMA's configuration is sized from the environment rather than by hand
    (brief section 4).
    """

    return MATEEnv(
        scenario=manifest['scenario'],
        seed=seed,
        discrete_levels=manifest['discrete_levels'],
        max_episode_steps=manifest['max_episode_steps'],
    )


def minimum_buffer_steps(
    horizon: int,
    *,
    state_decoder_batch_size: Optional[int] = None,
    rew_end_batch_size: Optional[int] = None,
) -> int:
    """Smallest replay buffer ``DreamerLearner.step`` can actually train from.

    ``DreamerMemory.sample_indices`` raises a bare ``ValueError('Not enough data
    in buffer')`` when ``batch_size * sequence_length`` exceeds what the buffer
    holds (DreamerMemory.py:263).  ``DreamerLearner.step`` draws two such batches:

    * state decoder -- ``state_decoder_batch_size``, ``sl=1``
    * reward/end model -- ``rew_end_batch_size``, ``sl=horizon`` for the
      transformer variant

    The second usually dominates: ``bs * horizon <= size - horizon + 1``.  Computing
    the bound here turns a confusing crash deep inside DIMA into an up-front, named
    constraint.

    The batch sizes default to the config values rather than being hard-coded, so
    raising them raises this bound with them.  Takes the horizon rather than a
    config object so ``research.pipeline`` can size a preset before any DIMA
    config exists.
    """

    defaults = MATEDreamerLearnerConfig()
    state_bs = int(state_decoder_batch_size or defaults.state_decoder_batch_size)
    rew_end_bs = int(rew_end_batch_size or defaults.rew_end_batch_size)
    return max(state_bs, (rew_end_bs + 1) * int(horizon) - 1)


def check_dataset(manifest: Dict, rollouts, env: MATEEnv) -> None:
    if not rollouts:
        raise ValueError('Dataset is empty.')
    sample = rollouts[0]
    expected = {
        'observation': env.n_obs,
        'shared_obs': env.state_dim,
        'next_shared_obs': env.state_dim,
        'action': env.n_actions,
    }
    for key, dim in expected.items():
        if sample[key].shape[-1] != dim:
            raise ValueError(
                f'Dataset {key} has last dimension {sample[key].shape[-1]}, '
                f'environment says {dim}. Was the dataset collected with a different scenario?'
            )
        if sample[key].shape[1] != env.n_agents:
            raise ValueError(
                f'Dataset {key} has {sample[key].shape[1]} agents, environment has {env.n_agents}.'
            )


def train(
    dataset_dir: Path,
    run_dir: Path,
    *,
    passes: int = 10,
    seed: int = 0,
    device: Optional[str] = None,
    horizon: Optional[int] = None,
    min_buffer_size: Optional[int] = None,
    n_samples: Optional[int] = None,
    wm_epochs: Optional[int] = None,
    denoiser_steps_first_epoch: Optional[int] = None,
    remodel_steps: Optional[int] = None,
    state_decoder_batch_size: Optional[int] = None,
    denoiser_batch_size: Optional[int] = None,
    rew_end_batch_size: Optional[int] = None,
    train_actor_critic: bool = False,
    save_every: int = 1,
    val_episodes: int = 20,
    eval_every: int = 1,
    sensitivity_samples: int = 4,
    detect_anomaly: bool = False,
    threads: Optional[int] = None,
    max_hours: Optional[float] = None,
    wandb_mode: str = 'disabled',
    tensorboard: bool = False,
    resume: Optional[str] = None,
    save_buffer: bool = False,
) -> Path:
    manifest, rollouts = load_dataset(dataset_dir)
    env = build_env_from_manifest(manifest, seed=seed)
    check_dataset(manifest, rollouts, env)

    # Held-out split, taken deterministically from the tail so that re-running
    # with the same dataset validates on the same episodes.
    val_episodes = max(0, min(int(val_episodes), max(0, len(rollouts) - 1)))
    validation_rollouts = rollouts[len(rollouts) - val_episodes :] if val_episodes else []
    rollouts = rollouts[: len(rollouts) - val_episodes] if val_episodes else rollouts

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

    config = build_learner_config(
        env,
        seed=seed,
        run_dir=str(run_dir),
        device=device,
        horizon=horizon,
        train_actor_critic=train_actor_critic,
        overrides=overrides,
    )
    allow_dima_checkpoint_globals()
    torch_setup = configure_torch(
        config.DEVICE, detect_anomaly=detect_anomaly, threads=threads, seed=seed
    )

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

    total_steps = sum(len(r['done']) for r in rollouts)
    if total_steps < config.MIN_BUFFER_SIZE:
        raise ValueError(
            f'Dataset holds {total_steps} steps but training needs at least '
            f'{config.MIN_BUFFER_SIZE} before DreamerLearner.step will train '
            f'(horizon={config.horizon}). Collect more episodes or lower --horizon.'
        )

    log('ENV', env.describe())
    log(
        'DATA',
        f'{len(rollouts)} train episodes / {total_steps} steps from {dataset_dir} '
        f'(mix: {manifest["policy_mix"]})',
    )
    log('DATA', f'{len(validation_rollouts)} held-out episodes for validation')
    if config.DEVICE.startswith('cuda'):
        log(
            'WM',
            f'GPU {torch_setup.get("gpu_name")} x{torch_setup.get("gpu_count")} '
            f'cc{torch_setup.get("gpu_capability")} {torch_setup.get("gpu_total_memory_gb")} GB · '
            f'tf32={torch_setup.get("tf32")} cudnn_benchmark={torch_setup.get("cudnn_benchmark")}',
        )
    else:
        log('WARN', f'running on {config.DEVICE}; this path is for testing, not for real training')
    if detect_anomaly:
        log(
            'WARN',
            'autograd anomaly detection is ON -- it makes training several times slower. '
            'Use it only to chase a NaN.',
        )

    resume_checkpoint = resolve_resume(resume, run_dir)
    if resume_checkpoint is not None:
        # A wandb-sourced checkpoint carries its run id in the sidecar beside it,
        # so even a run whose directory is gone continues on its own curves.
        resume_run_id = read_resume_meta(resume_checkpoint).get('wandb_run_id')
    else:
        resume_run_id = read_wandb_run_id(run_dir)

    with RunLogger(
        run_dir,
        name=f'wm-{manifest["scenario"]}-s{seed}',
        config=config.to_dict(),
        wandb_mode=wandb_mode,
        tensorboard=tensorboard,
        group=f'wm-{manifest["scenario"]}',
        resume_run_id=resume_run_id if resume_checkpoint is not None else None,
    ) as logger:
        logger.update_manifest(
            phase='world_model_training',
            dataset=str(dataset_dir),
            dataset_manifest=manifest,
            environment=env.metadata(),
            passes=passes,
            train_actor_critic=train_actor_critic,
            torch_setup=torch_setup,
            train_episodes=len(rollouts),
            validation_episodes=len(validation_rollouts),
        )
        # Capture DIMA's own loss curves without touching DIMA.
        logger.tee_dima_scalars()

        from agent.learners.DreamerLearner import DreamerLearner

        learner = DreamerLearner(config)
        # DIMA draws a tqdm bar per inner training step (DreamerLearner.py:315, 358,
        # 417) -- several hundred per episode.  On a terminal that is the progress
        # display; redirected to a server log file it is hundreds of megabytes of
        # carriage returns, so it is kept only when stdout is a terminal.
        learner.tqdm_vis = sys.stdout.isatty()
        # DreamerLearner.__init__ re-enables anomaly detection (line 83); undo it
        # unless it was explicitly asked for.
        configure_torch(config.DEVICE, detect_anomaly=detect_anomaly, threads=threads)
        log(
            'WM',
            f'device={config.DEVICE} horizon={config.horizon} '
            f'denoising_steps={config.diffusion_sampler_cfg.num_steps_denoising} '
            f'vq={config.vq_type} rew_end={config.rew_end_model_type}',
        )
        log(
            'WM',
            f'MIN_BUFFER_SIZE={config.MIN_BUFFER_SIZE} N_SAMPLES={config.N_SAMPLES} '
            f'WM_EPOCHS={config.WM_EPOCHS} actor_critic={"on" if train_actor_critic else "off"}',
        )
        log(
            'WM',
            f'batch sizes: state_decoder={config.state_decoder_batch_size} '
            f'denoiser={config.denoiser_batch_size} rew_end={config.rew_end_batch_size}',
        )

        rng = np.random.default_rng(seed)
        checkpoint = Path(run_dir) / 'ckpt' / 'model_final.pth'
        resume_path = Path(run_dir) / 'ckpt' / RESUME_CHECKPOINT_FILENAME
        resume_meta_path = Path(run_dir) / 'ckpt' / RESUME_META_FILENAME
        # Written once, up front, so every checkpoint in this directory -- including
        # the intermediate ones -- is self-describing even if the run is killed.
        sidecar = export_checkpoint_config(
            config, Path(run_dir) / 'ckpt' / CHECKPOINT_CONFIG_FILENAME
        )
        log('WM', f'checkpoint config -> {sidecar}')
        started = time.time()
        fed_steps = 0
        first_pass = 0
        elapsed_before_resume = 0.0
        validation_history_resumed: List[Dict[str, object]] = []

        if resume_checkpoint is not None:
            loop_state = learner.load_full(str(resume_checkpoint))
            if loop_state is not None:
                first_pass = int(loop_state['next_pass'])
                fed_steps = int(loop_state['fed_steps'])
                elapsed_before_resume = float(loop_state['elapsed_seconds'])
                validation_history_resumed = list(loop_state['validation_history'])
                rng.bit_generator.state = loop_state['rng_state']
                log(
                    'WM',
                    f'resumed from {resume_checkpoint}: starting at pass {first_pass + 1}/{passes}, '
                    f'fed_steps={fed_steps}, train_count={learner.train_count}, '
                    f'buffer={learner.replay_buffer.num_steps} steps',
                )
                if learner.replay_buffer.num_steps < config.MIN_BUFFER_SIZE:
                    log(
                        'WARN',
                        'the checkpoint carries no replay buffer, so training pauses until the '
                        'buffer refills from the dataset. Pass --save-buffer to avoid this.',
                    )
            if first_pass >= passes:
                log('WARN', f'checkpoint is already past --passes {passes}; nothing left to run.')

        layout = ObservationLayout.from_env_metadata(env.metadata())
        validation_set = [
            ValidationEpisode.from_rollout(r, env) for r in validation_rollouts
        ]
        validation_history: List[Dict[str, object]] = list(validation_history_resumed)
        world_model = None

        def run_validation(tag: int) -> None:
            """Measure the live model on held-out data (see research/validation.py)."""

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
                # Deliberately NOT varied per pass: the same held-out states must
                # be scored every time, or the trend measures sampling, not the
                # model.  See research/validation.py.
                seed=seed,
            )
            row['pass'] = tag
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
                'WM-DIAG',
                format_trend(
                    validation_history,
                    ('ade_h1', 'sighting_recall_h1', 'mae_h1', 'sensitivity_ratio'),
                ),
            )
            if row.get('sensitivity_ratio') is not None and row['sensitivity_ratio'] < 1.0:
                log(
                    'WARN',
                    f'action sensitivity ratio {row["sensitivity_ratio"]:.3f} < 1.0 -- changing the '
                    f'action still moves the prediction less than re-sampling it does, so no '
                    f'planner can use this model yet.',
                )

        # The budget covers the run as a whole, not each restart of it.
        deadline = started + max_hours * 3600.0 - elapsed_before_resume if max_hours else None
        stopped_early = False

        def save_resumable(next_pass: int) -> None:
            """Write the checkpoint a restart continues from, and push it to wandb.

            The wandb copy is what survives the instance, so the sidecar naming the
            run travels in the same artifact version as the weights.
            """

            learner.save_full(
                resume_path,
                runner_state={
                    'next_pass': next_pass,
                    'fed_steps': fed_steps,
                    'elapsed_seconds': elapsed_before_resume + time.time() - started,
                    'validation_history': validation_history,
                    'rng_state': rng.bit_generator.state,
                },
                save_buffer=save_buffer,
            )
            resume_meta_path.write_text(
                json.dumps({'wandb_run_id': getattr(logger.wandb_run, 'id', None),
                            'next_pass': next_pass,
                            'passes': passes}),
                encoding='utf-8',
            )
            logger.log_checkpoint(resume_path, resume_meta_path, sidecar, aliases=['latest'])

        for pass_index in range(first_pass, passes):
            order = rng.permutation(len(rollouts))
            for position, episode_index in enumerate(order):
                if deadline is not None and time.time() > deadline:
                    # Break out of the episode loop rather than the process, so the
                    # checkpoint below is still written: a preempted server run
                    # should leave a usable model, not nothing.
                    log(
                        'WARN',
                        f'--max-hours {max_hours} reached during pass {pass_index + 1}; '
                        f'stopping cleanly and saving.',
                    )
                    stopped_early = True
                    break
                rollout = rollouts[int(episode_index)]
                learner.step(rollout)
                fed_steps += len(rollout['done'])

                if position % 25 == 0:
                    log(
                        'WM',
                        f'pass {pass_index + 1}/{passes} episode {position + 1}/{len(order)} '
                        f'fed_steps={fed_steps} buffer={learner.replay_buffer.num_steps} '
                        f'train_count={learner.train_count} '
                        f'elapsed={time.time() - started:.0f}s',
                    )

            logger.scalars(
                {
                    'pass': pass_index + 1,
                    'fed_steps': fed_steps,
                    'buffer_steps': learner.replay_buffer.num_steps,
                    'train_count': learner.train_count,
                },
                step=pass_index + 1,
                prefix='wm/',
            )

            if eval_every and (pass_index + 1) % eval_every == 0:
                run_validation(pass_index + 1)

            # A pass cut short by --max-hours still counts as done: every pass is a
            # fresh permutation of the same dataset, so resuming into the next one
            # loses no data the run had not already replayed.
            if save_every and (pass_index + 1) % save_every == 0:
                path = Path(run_dir) / 'ckpt' / f'model_pass{pass_index + 1:03d}.pth'
                learner.save(str(path))
                log('WM', f'checkpoint -> {path}')
                logger.log_checkpoint(path, sidecar, aliases=[f'pass{pass_index + 1:03d}'])
                save_resumable(pass_index + 1)

            if stopped_early:
                break

        learner.save(str(checkpoint))
        log('WM', f'final checkpoint -> {checkpoint}')
        logger.log_checkpoint(checkpoint, sidecar, aliases=['final'])
        logger.update_manifest(
            checkpoint=str(checkpoint),
            fed_steps=fed_steps,
            train_count=learner.train_count,
            buffer_steps=learner.replay_buffer.num_steps,
            validation=validation_history,
            stopped_early=stopped_early,
            wall_seconds=round(time.time() - started, 1),
        )
        if validation_history:
            log('WM-DIAG', 'validation trend across passes:')
            for row in validation_history:
                log(
                    'WM-DIAG',
                    f'  pass {row["pass"]:>3}  train_count={row["train_count"]:>4}  '
                    + format_trend(
                        [row], ('ade_h1', 'sighting_recall_h1', 'mae_h1', 'sensitivity_ratio')
                    ),
                )

    env.close()
    return checkpoint


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog='python -m research.train_wm', description=__doc__)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--passes', type=int, default=10, help='sweeps over the dataset')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--min-buffer-size', type=int, default=None)
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--wm-epochs', type=int, default=None)
    parser.add_argument('--denoiser-steps-first-epoch', type=int, default=None)
    parser.add_argument('--remodel-steps', type=int, default=None)
    parser.add_argument('--state-decoder-batch-size', type=int, default=None,
                        help='DIMA used 256; the default here is larger. Pass the DIMA value '
                             'together with the matching epoch counts to reproduce it.')
    parser.add_argument('--denoiser-batch-size', type=int, default=None, help='DIMA used 64')
    parser.add_argument('--rew-end-batch-size', type=int, default=None, help='DIMA used 128')
    parser.add_argument('--train-actor-critic', action='store_true')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--val-episodes', type=int, default=20,
                        help='episodes held out of training for validation (0 disables)')
    parser.add_argument('--eval-every', type=int, default=1, help='validate every N passes')
    parser.add_argument('--sensitivity-samples', type=int, default=4)
    parser.add_argument('--detect-anomaly', action='store_true',
                        help='re-enable DIMA\'s autograd anomaly detection (several times slower)')
    parser.add_argument('--threads', type=int, default=None, help='torch CPU thread count')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='stop cleanly and save before this wall-clock budget runs out')
    parser.add_argument('--wandb-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--resume', default=None,
                        help='continue from ckpt/latest.pth: pass the file, its run directory, '
                             'or wandb://entity/project/artifact:alias')
    parser.add_argument('--save-buffer', action='store_true',
                        help='include the replay buffer in the resumable checkpoint, so a resumed '
                             'run trains immediately instead of refilling (much larger uploads)')
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    train(
        args.dataset,
        args.run_dir,
        passes=args.passes,
        seed=args.seed,
        device=args.device,
        horizon=args.horizon,
        min_buffer_size=args.min_buffer_size,
        n_samples=args.n_samples,
        wm_epochs=args.wm_epochs,
        denoiser_steps_first_epoch=args.denoiser_steps_first_epoch,
        remodel_steps=args.remodel_steps,
        state_decoder_batch_size=args.state_decoder_batch_size,
        denoiser_batch_size=args.denoiser_batch_size,
        rew_end_batch_size=args.rew_end_batch_size,
        train_actor_critic=args.train_actor_critic,
        save_every=args.save_every,
        val_episodes=args.val_episodes,
        eval_every=args.eval_every,
        sensitivity_samples=args.sensitivity_samples,
        detect_anomaly=args.detect_anomaly,
        resume=args.resume,
        save_buffer=args.save_buffer,
        threads=args.threads,
        max_hours=args.max_hours,
        wandb_mode=args.wandb_mode,
        tensorboard=args.tensorboard,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
