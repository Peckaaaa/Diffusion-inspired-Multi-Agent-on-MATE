"""Chẩn đoán vì sao lambda-return trong imagination gần như không đổi giữa các trạng thái.

Chạy READ-ONLY: nạp checkpoint, thu vài episode thật để đổ đầy replay buffer (buffer
không nằm trong checkpoint), rồi đo 4 thứ. KHÔNG sửa, KHÔNG train, KHÔNG ghi checkpoint.

    python diagnose_imagination.py --resume_path <latest_resume.pth> [--env_name MATE-4v8-9]
"""
import argparse, os, sys, tempfile, warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from einops import rearrange, repeat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--resume_path', type=str, required=True)
    p.add_argument('--env_name', type=str, default='MATE-4v8-9')
    p.add_argument('--mate_levels', type=int, default=5)
    p.add_argument('--mate_episode_limit', type=int, default=200)
    p.add_argument('--collect_episodes', type=int, default=30,
                   help='Số episode thật thu để đổ buffer (cần >= MIN_BUFFER_SIZE/201).')
    p.add_argument('--n_workers', type=int, default=2)
    p.add_argument('--ac_batch_size', type=int, default=256,
                   help='Nhỏ hơn 600 của training để chạy nhanh; thống kê không đổi.')
    return p.parse_args()


def build_configs(args):
    """Lặp lại đúng trình tự train.py dựng config, nếu không learner sẽ thiếu thuộc tính."""
    from configs.EnvConfigs import MATEConfig
    from configs.dreamer.mate.MATELearnerConfig import MATEDreamerLearnerConfig
    from configs.dreamer.mate.MATEControllerConfig import MATEDreamerControllerConfig
    from environments import Env

    RANDOM_SEED = 23 + 1 * 100
    learner, controller = MATEDreamerLearnerConfig(), MATEDreamerControllerConfig()
    env_config = MATEConfig(args.env_name, RANDOM_SEED,
                            levels=args.mate_levels, max_episode_steps=args.mate_episode_limit)

    env = env_config.create_env()
    for c in (learner, controller):
        c.IN_DIM = env.n_obs
        c.STATE_DIM = env.state_dim
        c.ACTION_SIZE = env.n_actions
        c.NUM_AGENTS = env.n_agents
        c.CONTINUOUS_ACTION = not env.discrete
        c.ACTION_SPACE = getattr(env, 'individual_action_space', None)
    env.close()

    for c in (learner, controller, env_config):
        c.ENV_TYPE = Env.MATE
    for c in (learner, controller):
        c.policy_class = 'discrete'
        c.state_decoder_type = "s + id"
    learner.seed = RANDOM_SEED
    learner.use_ce_for_cont = False
    learner.compute_end_in_TD = False
    learner.sample_temperature = "inf"
    learner.load_pretrained = False
    learner.load_path = None
    learner.diffusion_sampler_cfg.num_steps_denoising = (
        learner.NUM_AGENTS if learner.NUM_AGENTS > 2 else learner.NUM_AGENTS * 2)
    learner.ac_batch_size = args.ac_batch_size
    learner.RUN_DIR = tempfile.mkdtemp()
    os.makedirs(learner.RUN_DIR + "/ckpt", exist_ok=True)
    learner.map_name = args.env_name
    return learner, controller, env_config


def collect_into_buffer(learner, controller_config, env_config, n_episodes, n_workers):
    """Đổ replay buffer bằng episode THẬT. WorldModelEnv lấy trạng thái khởi đầu từ buffer
    này, mà buffer không được lưu trong checkpoint -> phải thu lại."""
    import ray
    from agent.workers.DreamerWorker import DreamerWorker

    ray.init(num_cpus=1, _temp_dir="/tmp/ray_diag", ignore_reinit_error=True,
             logging_level="ERROR")
    workers = [DreamerWorker.remote(i, env_config, controller_config) for i in range(n_workers)]
    params = learner.params()
    tasks = [w.run.remote(params) for w in workers]

    done = 0
    while done < n_episodes:
        ready, tasks = ray.wait(tasks)
        rollout, info = ray.get(ready)[0]
        # Chỉ nạp buffer, KHÔNG gọi learner.step() (nó sẽ train).
        learner.add_experience_to_dataset(rollout)
        learner.mamba_replay_buffer.append(
            rollout['observation'], rollout['shared_obs'], rollout['next_shared_obs'],
            rollout['action'], rollout['reward'], rollout['done'],
            rollout['fake'], rollout['last'], rollout.get('avail_action'))
        learner.state_rms.update(rollout['shared_obs'].mean(1))
        done += 1
        tasks.append(workers[info['idx']].run.remote(params))
        print(f"\r  thu {done}/{n_episodes} episode "
              f"(buffer {learner.replay_buffer.num_steps} steps)", end="", flush=True)
    print()
    return learner


# ----------------------------------------------------------------------------
# TEST 1 -- world model có phụ thuộc hành động không?
# ----------------------------------------------------------------------------
def test_action_conditioning(learner, cfg):
    from agent.world_models.world_model_env import WorldModelEnv

    dev, H, N, A = cfg.DEVICE, cfg.horizon, cfg.NUM_AGENTS, cfg.ACTION_SIZE
    B = cfg.ac_batch_size

    wm = WorldModelEnv(
        running_mean_std=learner.state_rms, state_decoder=learner.state_decoder,
        denoiser=learner.denoiser, rew_end_model=learner.rew_end_model,
        dataset=learner.replay_buffer, num_envs=B, cfg=cfg.worldmodel_env_cfg,
        return_denoising_trajectory=True, mode='non-ensemble',
        use_stack_obs=cfg.use_stack, num_stack_obs=cfg.stack_obs_num,
        env_type=cfg.ENV_TYPE, state_decoder_type=cfg.state_decoder_type,
        should_reset_with_dead=cfg.compute_end_in_TD, device=dev)

    wm.reset()
    # Chụp lại trạng thái sau reset để MỌI chuỗi hành động xuất phát từ CÙNG một state.
    snap = (wm.state_buffer.clone(), wm.obs_buffer.clone(), wm.act_buffer.clone())
    tpb = wm.tokens_per_block

    def restore():
        wm.state_buffer, wm.obs_buffer, wm.act_buffer = (t.clone() for t in snap)
        wm.ep_len = torch.zeros(B, dtype=torch.long, device=dev)
        wm.keys_values_rew_end = wm.rew_end_model.transformer.generate_empty_keys_values(
            n=B, max_tokens=wm.rew_end_model.config.max_tokens)
        m = torch.tril(torch.ones(tpb, tpb, device=dev))[None, :, :].repeat(B, 1, 1)
        wm.flipped_attn_mask = m.flip(dims=[-1])

    @torch.no_grad()
    def rollout_fixed(action_idx_seq):
        """action_idx_seq: list dài H, mỗi phần tử là index hành động dùng cho MỌI agent."""
        restore()
        total = torch.zeros(B, device=dev)
        for t in range(H):
            idx = torch.full((B, N), action_idx_seq[t], dtype=torch.long, device=dev)
            act = torch.nn.functional.one_hot(idx, A).float()
            _, _, rew, _, _, _, _ = wm.step(act)
            total += rew.view(B, -1).mean(-1) if rew.dim() > 1 else rew.view(B)
        return total.mean().item()

    rng = np.random.RandomState(0)
    results = {}
    results['A: toàn hành động 0']  = rollout_fixed([0] * H)
    results['B: toàn hành động 24'] = rollout_fixed([24] * H)
    results['C: toàn hành động 12'] = rollout_fixed([12] * H)
    rand_vals = []
    for k in range(10):
        seq = rng.randint(0, A, size=H).tolist()
        v = rollout_fixed(seq)
        rand_vals.append(v)
        results[f'random #{k+1}'] = v

    vals = np.array(list(results.values()))
    spread = (vals.max() - vals.min()) / (abs(vals.mean()) + 1e-8) * 100
    del wm
    return results, spread, np.array(rand_vals)


# ----------------------------------------------------------------------------
# TEST 2 + 3 -- thống kê advantage & tỉ lệ hai thành phần loss
# ----------------------------------------------------------------------------
def test_advantage(learner, cfg):
    from agent.optim.loss import rollout_diffusion_world_models
    from agent.optim.utils import advantage as normalize_adv
    from agent.optim.utils import calculate_ppo_loss
    from agent.world_models.actor_critic import compute_lambda_returns_with_pcont_wo_end
    import torch.nn.functional as F

    with torch.no_grad():
        obs, shared_obs, act, rew, pcont, end, trunc, logits_act, val, val_bootstrap, av = \
            rollout_diffusion_world_models(
                learner.replay_buffer, learner.state_rms, learner.state_decoder,
                learner.denoiser, learner.rew_end_model, learner.actor, learner.critic,
                cfg, env_type=learner.env_type)

        vs = val_bootstrap.shape
        vb = learner.value_normalizer.denormalize(rearrange(val_bootstrap, 'b l 1 -> (b l) 1'))
        vb = rearrange(vb, '(b l) 1 -> b l 1', b=vs[0], l=vs[1])
        vu = learner.value_normalizer.denormalize(rearrange(val, 'b l 1 -> (b l) 1'))
        vu = rearrange(vu, '(b l) 1 -> b l 1', b=vs[0], l=vs[1])

        R = compute_lambda_returns_with_pcont_wo_end(
            rew.view(*rew.shape, 1), pcont.view(*pcont.shape, 1), vb,
            gamma=1.0 if cfg.contdisc else cfg.GAMMA, lmbda=cfg.DISCOUNT_LAMBDA)

        A_raw = (R - vu).detach()
        A_nrm = normalize_adv(A_raw.clone())

        # Tỉ lệ loss: epoch 0 của PPO có rho == 1 (cùng policy sinh ra action)
        A_rep = repeat(A_nrm, 'b h d -> b h n d', n=cfg.NUM_AGENTS)
        o = rearrange(obs, 'b h n d -> (b h) n d')
        a = rearrange(act, 'b h n d -> (b h) n d')
        lg = rearrange(logits_act, 'b h n d -> (b h) n d')
        Ar = rearrange(A_rep, 'b h n d -> (b h) n d')

        _, new_policy = learner.actor(o)
        ai = a.argmax(-1, keepdim=True)
        rho = (F.log_softmax(new_policy, -1).gather(2, ai)
               - F.log_softmax(lg, -1).gather(2, ai)).exp()
        pol, ent = calculate_ppo_loss(new_policy, rho, Ar, cfg.clip_param)

        # Toàn batch: rho==1 (cùng policy sinh action) và A đã chuẩn hoá zero-mean
        # -> mean(pol) = -mean(A) = 0 ĐÚNG BẰNG 0. Đây là con số giải thích
        # "Actor_loss dao động quanh đúng 0" mà KHÔNG phải lỗi.
        pol_full = pol.mean().abs().item()
        # Training thật chia minibatch step=2000 -> mean từng minibatch KHÁC 0,
        # dao động ~1/sqrt(2000). Đây mới là biên độ bạn thấy trên biểu đồ.
        step = 2000
        mb = [pol[i:i + step].mean().item() for i in range(0, pol.shape[0], step)]
        pol_m = float(np.mean(np.abs(mb)))
        ent_m = (ent.unsqueeze(-1) * cfg.ENTROPY).mean().abs().item()

    stats = {
        'R.mean': R.mean().item(), 'R.std': R.std().item(),
        'V_unnorm.mean': vu.mean().item(), 'V_unnorm.std': vu.std().item(),
        'V_logged(normalized).mean': val.mean().item(),
        'A_raw.mean': A_raw.mean().item(), 'A_raw.std': A_raw.std().item(),
        'A_raw.abs().max': A_raw.abs().max().item(), 'A_raw.min': A_raw.min().item(),
        'A_norm.std': A_nrm.std().item(),
        'rew.mean': rew.mean().item(), 'rew.std': rew.std().item(),
        'pcont.mean': pcont.float().mean().item(),
        '|policy_loss| toàn batch': pol_full,
        '|policy_loss| /minibatch2000': pol_m,
        '|entropy_loss*EC|': ent_m,
        'ent_ratio (dùng minibatch)': ent_m / (pol_m + 1e-12),
    }
    hist, edges = np.histogram(A_raw.cpu().numpy().ravel(), bins=10)
    return stats, hist, edges, R, vu


# ----------------------------------------------------------------------------
# TEST 4 -- reward model có khớp reward thật không?
# ----------------------------------------------------------------------------
def test_reward_calibration(learner, cfg, n_batches=20):
    try:
        preds, trues = [], []
        with torch.no_grad():
            for _ in range(n_batches):
                s = learner.mamba_replay_buffer.sample_batch(
                    bs=64, sl=cfg.horizon, mode='rew_end_model')
                s = learner._to_device(s)
                s['shared_obs'] = learner.normalize_state(s['shared_obs'].mean(2))
                s['next_shared_obs'] = learner.normalize_state(s['next_shared_obs'].mean(2))
                mask = learner.mamba_replay_buffer.generate_attn_mask(
                    s["done"], learner.rew_end_model.config.tokens_per_block).to(cfg.DEVICE)
                # Dựng token đúng như TransRewEndModel.compute_loss()
                m = learner.rew_end_model
                b, l, n = s['action'].shape[:3]
                act = rearrange(s['action'], 'b l n e -> (b l) n e')
                if m.is_discrete_action:
                    act = act.argmax(-1)
                act_cond = rearrange(m.act_emb(act), '(b l) e -> b l e', b=b, l=l)
                tok = torch.stack([s['shared_obs'].clone(),
                                   torch.empty(b, l, s['shared_obs'].size(-1),
                                               device=act_cond.device, dtype=act_cond.dtype)], dim=-2)
                tok = rearrange(tok, 'b l m e -> b (l m) e')
                out = m(tok, perattn_out=act_cond, attention_mask=mask)
                preds.append(out.pred_rewards.flatten().float().cpu().numpy())
                trues.append(s['reward'].mean(-2).flatten().float().cpu().numpy())
        p, t = np.concatenate(preds), np.concatenate(trues)
        n = min(len(p), len(t)); p, t = p[:n], t[:n]
        return {'MAE': float(np.abs(p - t).mean()),
                'Pearson r': float(np.corrcoef(p, t)[0, 1]),
                'pred.std': float(p.std()), 'true.std': float(t.std())}
    except Exception as e:
        return {'LỖI': f"{type(e).__name__}: {e}  (API rew_end_model khác dự kiến)"}


def main():
    args = parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    cfg, controller_cfg, env_cfg = build_configs(args)
    print(f"DEVICE={cfg.DEVICE}  STATE_DIM={cfg.STATE_DIM}  ACTION_SIZE={cfg.ACTION_SIZE}  "
          f"NUM_AGENTS={cfg.NUM_AGENTS}  horizon={cfg.horizon}")

    learner = cfg.create_learner()
    ck = torch.load(args.resume_path, map_location=cfg.DEVICE, weights_only=False)
    for name, mod in [('state_decoder', learner.state_decoder), ('denoiser', learner.denoiser),
                      ('rew_end_model', learner.rew_end_model), ('actor', learner.actor),
                      ('critic', learner.critic)]:
        if name in ck:
            mod.load_state_dict(ck[name]); mod.eval()
    if 'running_mean_std' in ck:
        learner.state_rms = ck['running_mean_std']
    if 'value_normalizer' in ck:
        learner.value_normalizer.load_state_dict(ck['value_normalizer'])
        print("  (checkpoint CÓ value_normalizer)")
    else:
        print("  (checkpoint KHÔNG có value_normalizer -> đang dùng normalizer khởi tạo mới; "
              "đây chính là nguồn nhảy bậc của Value sau resume)")
    print(f"nạp checkpoint: {args.resume_path}  keys={sorted(ck.keys())}")

    print("\nThu episode thật để đổ replay buffer...")
    collect_into_buffer(learner, controller_cfg, env_cfg, args.collect_episodes, args.n_workers)

    W = 78
    print("\n" + "=" * W + "\nTEST 1 — WORLD MODEL CÓ PHỤ THUỘC HÀNH ĐỘNG KHÔNG?\n" + "=" * W)
    res, spread, rand_vals = test_action_conditioning(learner, cfg)
    print(f"{'chuỗi hành động':<28}{'tổng reward tưởng tượng (H=15)':>34}")
    print("-" * W)
    for k, v in res.items():
        print(f"{k:<28}{v:>34.6f}")
    print("-" * W)
    print(f"{'biên độ (max-min)/|mean|':<28}{spread:>33.3f}%")
    print(f"{'std của 10 chuỗi ngẫu nhiên':<28}{rand_vals.std():>34.6f}")
    print(f"=> {'BỎ QUA ACTION (spread < 5%)' if spread < 5 else 'CÓ phụ thuộc hành động'}")

    print("\n" + "=" * W + "\nTEST 2+3 — THỐNG KÊ ADVANTAGE & TỈ LỆ LOSS\n" + "=" * W)
    stats, hist, edges, R, vu = test_advantage(learner, cfg)
    for k, v in stats.items():
        print(f"  {k:<28}{v:>18.6f}")
    print("\n  histogram A_raw (10 bins):")
    for i in range(len(hist)):
        bar = "#" * int(40 * hist[i] / max(hist.max(), 1))
        print(f"    [{edges[i]:>9.4f},{edges[i+1]:>9.4f})  {hist[i]:>7d} {bar}")

    print("\n" + "=" * W + "\nTEST 4 — CALIBRATION REWARD MODEL\n" + "=" * W)
    for k, v in test_reward_calibration(learner, cfg).items():
        print(f"  {k:<28}{v}")

    print("\n" + "=" * W + "\nKẾT LUẬN TỰ ĐỘNG\n" + "=" * W)
    if spread < 5:
        print("  NHÁNH 1: world model BỎ QUA action conditioning -> sửa đường đi action TRƯỚC.")
    elif stats['A_raw.std'] < 0.05:
        print("  NHÁNH 2: reward phụ thuộc action nhưng A_raw.std rất nhỏ")
        print("           -> lambda-return gần hằng số; advantage() chuẩn hoá NHIỄU lên std=1.")
    elif stats['ent_ratio'] > 1:
        print("  NHÁNH 3: ent_ratio > 1 -> entropy bonus át policy gradient.")
    else:
        print("  Không rơi vào nhánh nào rõ rệt -- xem bảng số ở trên.")
    print(f"  (R.std={stats['R.std']:.6f} là con số quyết định: nó đo lambda-return có")
    print("   thay đổi giữa các trạng thái hay không.)")


if __name__ == "__main__":
    main()
