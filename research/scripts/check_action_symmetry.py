"""Per-camera action sensitivity of the diffusion sampler, untrained.

conv_out is zero-initialised, so a freshly built denoiser returns the same state
for every action and the probe would report a flat zero.  Its weights are
randomised here so the *wiring* can be measured without training: does an equal
change to each camera's action move the sampled state by an equal amount?
"""

import sys

import numpy as np
import torch

import research  # noqa: F401
from research.config import build_learner_config
from research.env_adapter import MATEEnv
from agent.world_models.diffusion import Denoiser, DiffusionSampler


def measure(action_cond, steps, seed=0):
    env = MATEEnv(scenario='MATE-4v8-9', seed=0, discrete_levels=5, max_episode_steps=10)
    cfg = build_learner_config(
        env, device='cpu',
        overrides={'action_cond': action_cond, 'num_steps_denoising': steps},
    )
    # DreamerLearner.__init__:92-93 does this; the config template ships -1.
    cfg.denoiser_cfg.inner_model.state_dim = cfg.STATE_DIM
    cfg.denoiser_cfg.inner_model.action_dim = cfg.ACTION_SIZE
    torch.manual_seed(seed)
    denoiser = Denoiser(
        cfg.denoiser_cfg, num_agents=cfg.NUM_AGENTS,
        clip_denoised=False, is_continuous_act=False,
    ).eval()
    # Every ResBlock's second conv and the output conv are zero-initialised, which
    # makes an untrained UNet the identity: the conditioning vector -- and so the
    # action -- cannot reach the output at all, and every candidate would tie at
    # exactly 0.  Randomise those weights so the *wiring* is what gets measured.
    with torch.no_grad():
        for module in denoiser.modules():
            if isinstance(module, torch.nn.Conv1d) and not module.weight.any():
                torch.nn.init.normal_(module.weight, std=0.05)
    sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler_cfg)

    sl = cfg.denoiser_cfg.inner_model.num_steps_conditioning
    n, a = cfg.NUM_AGENTS, cfg.ACTION_SIZE
    rng = np.random.default_rng(seed)

    base = torch.as_tensor(rng.integers(0, a, size=(1, sl, n)), dtype=torch.long)
    state = torch.randn(1, sl, cfg.STATE_DIM)
    x0 = torch.randn(1, 1, cfg.STATE_DIM) * sampler.sigmas[0]

    def sample(act):
        with torch.no_grad():
            out, _ = sampler.sample(state, act, x0=x0.clone())
        return out.reshape(-1).numpy()

    reference = sample(base)
    per_camera = []
    for camera in range(n):
        moved = []
        for value in range(0, a, max(1, a // 6)):
            act = base.clone()
            act[0, -1, camera] = value
            moved.append(np.linalg.norm(sample(act) - reference))
        per_camera.append(float(np.mean(moved)))
    env.close()
    return per_camera


SEEDS = 5
for cond, steps in (('sequential', 4), ('joint', 4), ('joint', 7)):
    runs = np.stack([measure(cond, steps, seed=s) for s in range(SEEDS)])
    effects = runs.mean(axis=0)
    spread = effects.max() / max(effects.min(), 1e-12)
    print(
        f'{cond:>10}  K={steps}  per-camera={[round(float(e), 4) for e in effects]}  '
        f'max/min={spread:.2f}  (mean of {SEEDS} seeds)'
    )
