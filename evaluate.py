"""Coverage-rate evaluation, standalone or called from training.

    python evaluate.py --checkpoint runs/categorical_diffusion/checkpoint.pt --episodes 10
"""

import argparse

import numpy as np
import torch


def evaluate_policy(env, policy, episodes, deterministic=True, render=False):
    """Run whole episodes in the real environment.  No world model involved."""

    returns, coverages = [], []
    for _ in range(episodes):
        current = env.reset()
        done = False
        total, coverage = 0.0, []
        while not done:
            actions, emissions = policy.act_numpy(
                current['obs'], current['messages'], deterministic=deterministic
            )
            current, reward, done, info = env.step(actions, emissions)
            if render:
                env.render()
            total += reward
            coverage.append(info['coverage_rate'])
        returns.append(total)
        coverages.append(float(np.mean(coverage)))

    return {
        'eval/episode_return': float(np.mean(returns)),
        'eval/coverage_rate': float(np.mean(coverages)),
        'eval/coverage_rate_std': float(np.std(coverages)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--scenario', type=str, default=None, help='override the trained scenario')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--stochastic', action='store_true', help='sample actions instead of using the mean')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    return parser.parse_args()


def main():
    # Imported here, not at module scope: train.py imports evaluate_policy from
    # this module, and a top-level import back into train.py would be circular.
    from train import build_env, build_policy, env_spec

    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = checkpoint['config']
    config['train']['device'] = args.device
    config['env']['seed'] = args.seed
    if args.scenario is not None:
        config['env']['scenario'] = args.scenario

    env = build_env(config)
    env.obs_rms.load_state_dict(checkpoint['obs_rms'])
    env.state_rms.load_state_dict(checkpoint['state_rms'])

    policy = build_policy(env_spec(env), config)
    policy.load_state_dict(checkpoint['policy'])
    policy.actor.eval()
    policy.critic.eval()

    print(env.describe())
    metrics = evaluate_policy(
        env,
        policy,
        args.episodes,
        deterministic=not args.stochastic,
        render=args.render,
    )
    for key, value in metrics.items():
        print(f'{key}={value:.4f}')
    env.close()


if __name__ == '__main__':
    main()
