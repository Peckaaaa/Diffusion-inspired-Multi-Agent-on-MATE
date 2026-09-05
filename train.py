"""End-to-end training entrypoint.

Four stages per iteration:

  1. act in the real MATE environment, fill the replay buffer
  2. update the world model (FSQ tokenizer, categorical diffusion, reward/done)
  3. imagine H steps through the categorical reverse process
  4. update communicative MAPPO on the imagined trajectories

``src/envs/config_resolver.py`` imports ``mate`` from the MATE-main checkout it
finds next to (or inside) the repository; ``MATE_ROOT`` overrides that search.

Run:
    python train.py --config src/configs/default.yaml --set train.device=cpu
"""

import argparse
import copy
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml

# The packages live under src/; the entrypoints stay at the repository root.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from algorithms.communicative_mappo import CommunicativeMAPPO
from algorithms.replay_buffer import ReplayBuffer
from algorithms.world_model_trainer import WorldModelTrainer
from envs.mate_wrapper import MATEEnv
from evaluate import evaluate_policy


# ------------------------------------------------------------------------ config

def load_config(path, overrides=()):
    with open(path, 'r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle)

    for override in overrides:
        key, _, raw = override.partition('=')
        if not _:
            raise ValueError(f'Override {override!r} is not of the form section.key=value')
        node = config
        *parents, leaf = key.split('.')
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise KeyError(f'Unknown config key {key!r}')
        node[leaf] = yaml.safe_load(raw)

    if config['train']['device'] == 'auto':
        config['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'configs', 'default.yaml'),
    )
    parser.add_argument(
        '--set',
        dest='overrides',
        action='append',
        default=[],
        metavar='section.key=value',
        help='override a config leaf, repeatable',
    )
    return parser.parse_args()


# --------------------------------------------------------------------- stage 1

class Collector:
    """Steps the real environment and writes transitions to the buffer.

    Keeps the running episode across calls so an iteration boundary does not
    truncate an episode.
    """

    def __init__(self, env, policy, buffer):
        self.env = env
        self.policy = policy
        self.buffer = buffer
        self.current = env.reset()
        self.episode_return = 0.0
        self.episode_coverage = []
        self.finished_episodes = []

    def collect(self, num_steps, random_actions=False):
        n = self.env.n_agents
        for _ in range(num_steps):
            if random_actions:
                actions = np.random.uniform(-1.0, 1.0, (n, self.env.action_dim)).astype(np.float32)
                emissions = np.random.uniform(-1.0, 1.0, (n, self.env.msg_dim)).astype(np.float32)
            else:
                actions, emissions = self.policy.act_numpy(
                    self.current['obs'], self.current['messages']
                )

            nxt, reward, done, info = self.env.step(actions, emissions)

            self.buffer.add(
                state=self.current['state'],
                obs=self.current['obs'],
                messages_in=self.current['messages'],
                messages_out=emissions,
                actions=actions,
                reward=reward,
                next_state=nxt['state'],
                next_messages_in=nxt['messages'],
                done=done,
            )

            self.episode_return += reward
            self.episode_coverage.append(info['coverage_rate'])
            self.current = nxt

            if done:
                self.finished_episodes.append(
                    {
                        'return': self.episode_return,
                        'coverage_rate': float(np.mean(self.episode_coverage)),
                    }
                )
                self.episode_return = 0.0
                self.episode_coverage = []
                self.current = self.env.reset()

    def drain_stats(self):
        if not self.finished_episodes:
            return {}
        stats = {
            'env/episode_return': float(np.mean([e['return'] for e in self.finished_episodes])),
            'env/coverage_rate': float(
                np.mean([e['coverage_rate'] for e in self.finished_episodes])
            ),
            'env/episodes': len(self.finished_episodes),
        }
        self.finished_episodes = []
        return stats


# ------------------------------------------------------------------------ setup

def build_env(config, seed_offset=0):
    env_config = config['env']
    return MATEEnv(
        scenario=env_config['scenario'],
        max_episode_steps=env_config['max_episode_steps'],
        msg_dim=env_config['msg_dim'],
        camera_comm=env_config['camera_comm'],
        reward_scale=env_config['reward_scale'],
        seed=env_config['seed'] + seed_offset,
    )


def env_spec(env):
    return {
        'n_agents': env.n_agents,
        'state_dim': env.state_dim,
        'obs_dim': env.obs_dim,
        'msg_dim': env.msg_dim,
        'action_dim': env.action_dim,
    }


def build_policy(spec, config):
    policy_config = config['policy']
    return CommunicativeMAPPO(
        obs_dim=spec['obs_dim'],
        msg_dim=spec['msg_dim'],
        action_dim=spec['action_dim'],
        state_dim=spec['state_dim'],
        hidden_dim=policy_config['hidden_dim'],
        actor_lr=policy_config['actor_lr'],
        critic_lr=policy_config['critic_lr'],
        clip_ratio=policy_config['clip_ratio'],
        entropy_coef=policy_config['entropy_coef'],
        value_coef=policy_config['value_coef'],
        max_grad_norm=policy_config['max_grad_norm'],
        gamma=policy_config['gamma'],
        lam=policy_config['lam'],
        ppo_epochs=policy_config['ppo_epochs'],
        num_minibatches=policy_config['num_minibatches'],
        device=config['train']['device'],
    )


# ------------------------------------------------------------------------- main

def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)

    train_config = config['train']
    torch.manual_seed(config['env']['seed'])
    np.random.seed(config['env']['seed'])
    os.makedirs(train_config['save_dir'], exist_ok=True)

    env = build_env(config)
    eval_env = build_env(config, seed_offset=10_000)
    # Evaluation must normalize with the statistics the policy was trained under.
    eval_env.obs_rms = env.obs_rms
    eval_env.state_rms = env.state_rms

    spec = env_spec(env)
    world_model = WorldModelTrainer(spec, config, train_config['device'])
    policy = build_policy(spec, config)
    buffer = ReplayBuffer(
        capacity=train_config['buffer_capacity'],
        n_agents=spec['n_agents'],
        state_dim=spec['state_dim'],
        obs_dim=spec['obs_dim'],
        msg_dim=spec['msg_dim'],
        action_dim=spec['action_dim'],
    )
    collector = Collector(env, policy, buffer)

    logger = None
    if train_config['wandb']:
        import wandb

        logger = wandb.init(project='categorical-diffusion-mate', config=config)

    # The device is printed because `auto` falls back to CPU silently when the
    # installed torch has no CUDA build -- on a GPU server that is worth seeing.
    device_name = train_config['device']
    if device_name.startswith('cuda'):
        device_name += f' ({torch.cuda.get_device_name(torch.device(device_name))})'
    print(
        f'{env.describe()} | codebook {world_model.autoencoder.codebook_size} x '
        f'{config["world_model"]["num_tokens"]} tokens | '
        f'{config["diffusion"]["noise_type"]} diffusion, T={config["diffusion"]["num_steps"]}, '
        f'{config["diffusion"]["sample_steps"]} sampling steps | device {device_name}'
    )

    collector.collect(train_config['seed_steps'], random_actions=True)
    env_steps = train_config['seed_steps']
    iteration = 0
    start = time.time()

    while env_steps < train_config['total_env_steps']:
        iteration += 1

        collector.collect(train_config['env_steps_per_iter'])
        env_steps += train_config['env_steps_per_iter']

        wm_metrics = defaultdict(float)
        for _ in range(train_config['wm_updates_per_iter']):
            step_metrics = world_model.update(buffer, train_config['wm_batch_size'])
            for key, value in step_metrics.items():
                wm_metrics[key] += value / train_config['wm_updates_per_iter']

        policy_metrics = defaultdict(float)
        for _ in range(train_config['policy_updates_per_iter']):
            rollout, imagine_stats = world_model.imagine(policy, buffer, config)
            for key, value in {**policy.update(rollout), **imagine_stats}.items():
                policy_metrics[key] += value / train_config['policy_updates_per_iter']

        metrics = {
            'env_steps': env_steps,
            'iteration': iteration,
            'sps': env_steps / (time.time() - start),
            **wm_metrics,
            **policy_metrics,
            **collector.drain_stats(),
        }

        if iteration % train_config['eval_every'] == 0:
            metrics.update(evaluate_policy(eval_env, policy, train_config['eval_episodes']))
            torch.save(
                {
                    'world_model': world_model.state_dict(),
                    'policy': policy.state_dict(),
                    'obs_rms': env.obs_rms.state_dict(),
                    'state_rms': env.state_rms.state_dict(),
                    'config': copy.deepcopy(config),
                },
                os.path.join(train_config['save_dir'], 'checkpoint.pt'),
            )

        print(
            ' | '.join(
                f'{k}={v:.4f}' if isinstance(v, float) else f'{k}={v}'
                for k, v in metrics.items()
            )
        )
        if logger is not None:
            logger.log(metrics, step=env_steps)

    env.close()
    eval_env.close()


if __name__ == '__main__':
    main()
