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
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from research.collect import load_dataset
from research.config import allow_dima_checkpoint_globals, build_learner_config
from research.env_adapter import MATEEnv
from research.logging_utils import RunLogger, log


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


def minimum_buffer_steps(config) -> int:
    """Smallest replay buffer ``DreamerLearner.step`` can actually train from.

    ``DreamerMemory.sample_indices`` raises a bare ``ValueError('Not enough data
    in buffer')`` when ``batch_size * sequence_length`` exceeds what the buffer
    holds (DreamerMemory.py:223-227).  ``DreamerLearner.step`` draws two such
    batches with hard-coded sizes:

    * state decoder -- ``bs=256``, ``sl=1``  (DreamerLearner.py:317)
    * reward/end model -- ``bs=128``, ``sl=horizon`` for the transformer variant
      (DreamerLearner.py:420)

    The second dominates: ``128 * horizon <= size - horizon + 1``.  Computing the
    bound here turns a confusing crash deep inside DIMA into an up-front, named
    constraint.
    """

    horizon = int(config.horizon)
    return max(256, 129 * horizon - 1)


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
    train_actor_critic: bool = False,
    save_every: int = 1,
    wandb_mode: str = 'disabled',
    tensorboard: bool = False,
) -> Path:
    manifest, rollouts = load_dataset(dataset_dir)
    env = build_env_from_manifest(manifest, seed=seed)
    check_dataset(manifest, rollouts, env)

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

    required = minimum_buffer_steps(config)
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
        f'{len(rollouts)} episodes / {total_steps} steps from {dataset_dir} '
        f'(mix: {manifest["policy_mix"]})',
    )

    with RunLogger(
        run_dir,
        name=f'wm-{manifest["scenario"]}-s{seed}',
        config=config.to_dict(),
        wandb_mode=wandb_mode,
        tensorboard=tensorboard,
        group=f'wm-{manifest["scenario"]}',
    ) as logger:
        logger.update_manifest(
            phase='world_model_training',
            dataset=str(dataset_dir),
            dataset_manifest=manifest,
            environment=env.metadata(),
            passes=passes,
            train_actor_critic=train_actor_critic,
        )

        from agent.learners.DreamerLearner import DreamerLearner

        learner = DreamerLearner(config)
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

        rng = np.random.default_rng(seed)
        checkpoint = Path(run_dir) / 'ckpt' / 'model_final.pth'
        started = time.time()
        fed_steps = 0

        for pass_index in range(passes):
            order = rng.permutation(len(rollouts))
            for position, episode_index in enumerate(order):
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

            if save_every and (pass_index + 1) % save_every == 0:
                path = Path(run_dir) / 'ckpt' / f'model_pass{pass_index + 1:03d}.pth'
                learner.save(str(path))
                log('WM', f'checkpoint -> {path}')

        learner.save(str(checkpoint))
        log('WM', f'final checkpoint -> {checkpoint}')
        logger.update_manifest(
            checkpoint=str(checkpoint),
            fed_steps=fed_steps,
            train_count=learner.train_count,
            buffer_steps=learner.replay_buffer.num_steps,
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
    parser.add_argument('--train-actor-critic', action='store_true')
    parser.add_argument('--save-every', type=int, default=1)
    parser.add_argument('--wandb-mode', default='disabled', choices=['disabled', 'offline', 'online'])
    parser.add_argument('--tensorboard', action='store_true')
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
        train_actor_critic=args.train_actor_critic,
        save_every=args.save_every,
        wandb_mode=args.wandb_mode,
        tensorboard=args.tensorboard,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
