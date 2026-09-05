#!/usr/bin/env bash
# Report everything needed to size this repository's training parameters to a host.
#
#   bash research/scripts/server_probe.sh              # system facts only
#   conda activate dima && bash research/scripts/server_probe.sh   # + torch and timing
#
# Run it from the repository root. Everything between the BEGIN/END markers is
# safe to paste back: no keys, no hostnames beyond the kernel string, no paths
# outside the repository.
#
# Why each block is here:
#   cgroup limits  -- a rented container reports the HOST's cores and RAM through
#                     nproc and /proc/meminfo while the cgroup caps it far lower.
#                     Tuning to the wrong one is the classic way to get OOM-killed.
#   arch list      -- a torch wheel without the card's compute capability imports
#                     fine and fails at the first kernel launch.
#   env-step rate  -- online collection is single-threaded Python; it bounds
#                     throughput no matter which GPU is fitted.
#   kernel timing  -- this world model is many small kernels, so launch overhead,
#                     not FLOPs, is what a faster card would have to beat.

set -uo pipefail

hr() { printf '%s\n' '--------------------------------------------------------------'; }

echo "===== BEGIN SERVER PROBE ====="
echo "probe_version 1  date $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

hr; echo "[OS]"
uname -srm
[ -r /etc/os-release ] && . /etc/os-release && echo "distro       $PRETTY_NAME"
echo "container    $( [ -f /.dockerenv ] && echo yes || echo 'no / unknown' )"

hr; echo "[CPU]"
if [ -r /proc/cpuinfo ]; then
  echo "model        $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "logical      $(nproc --all 2>/dev/null || echo '?')   available_to_process $(nproc 2>/dev/null || echo '?')"
  echo "sockets      $(lscpu 2>/dev/null | awk -F: '/^Socket\(s\)/{gsub(/ /,"",$2);print $2}')"
  echo "cores/socket $(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{gsub(/ /,"",$2);print $2}')"
  echo "mhz_max      $(lscpu 2>/dev/null | awk -F: '/^CPU max MHz/{gsub(/ /,"",$2);print $2}')"
  echo "mhz_now      $(grep -m1 'cpu MHz' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
  echo "governor     $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a')"
  echo "loadavg      $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)"
else
  echo "no /proc/cpuinfo (not Linux?)"
fi

hr; echo "[CGROUP LIMITS]  what this process may actually use"
if [ -f /sys/fs/cgroup/memory.max ]; then
  echo "cgroup       v2"
  echo "memory.max   $(cat /sys/fs/cgroup/memory.max)"
  echo "memory.high  $(cat /sys/fs/cgroup/memory.high 2>/dev/null || echo '-')"
  echo "cpu.max      $(cat /sys/fs/cgroup/cpu.max 2>/dev/null || echo '-')   # quota period, in us"
elif [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
  echo "cgroup       v1"
  echo "memory.limit $(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  echo "cpu.quota    $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null || echo '-')"
  echo "cpu.period   $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo '-')"
else
  echo "none found — the numbers above are the real limits"
fi

hr; echo "[MEMORY]"
free -g 2>/dev/null || echo "free(1) unavailable"
echo "swap_total   $(awk '/SwapTotal/{print $2/1048576" GiB"}' /proc/meminfo 2>/dev/null)"

hr; echo "[DISK]  where runs/ and checkpoints land"
df -h . 2>/dev/null | tail -2
ROOTDEV="$(df --output=source . 2>/dev/null | tail -1 | sed 's|/dev/||; s|[0-9]*$||')"
[ -n "${ROOTDEV:-}" ] && [ -r "/sys/block/$ROOTDEV/queue/rotational" ] && \
  echo "rotational   $(cat /sys/block/$ROOTDEV/queue/rotational)  (0 = SSD/NVMe, 1 = spinning)"

hr; echo "[GPU]"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total,memory.used,driver_version,clocks.max.sm,power.limit \
             --format=csv 2>/dev/null || nvidia-smi
  echo "-- processes already on the card --"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | head -10
  echo "CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-unset}'"
else
  echo "nvidia-smi not found"
fi

hr; echo "[PYTHON / TORCH]"
if command -v python >/dev/null 2>&1; then
python - <<'PY' 2>&1
import platform, sys
print(f"python       {platform.python_version()}  ({sys.executable})")
try:
    import torch
except Exception as exc:
    print(f"torch        NOT IMPORTABLE: {exc}")
    raise SystemExit
print(f"torch        {torch.__version__}   cuda_build={torch.version.cuda}")
print(f"cuda_avail   {torch.cuda.is_available()}")
# A wheel whose arch list lacks the card's sm_XX imports fine and dies at the
# first kernel launch with 'no kernel image is available'.
try:
    print(f"arch_list    {torch.cuda.get_arch_list()}")
except Exception as exc:
    print(f"arch_list    unavailable ({exc})")
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability(0)
    props = torch.cuda.get_device_properties(0)
    print(f"device0      {torch.cuda.get_device_name(0)}  cc{major}.{minor}  "
          f"{props.total_memory/1024**3:.1f} GiB  SMs={props.multi_processor_count}")
    print(f"tf32_usable  {major >= 8}   (needs compute capability 8.0+)")
    print(f"sm_in_wheel  {('sm_%d%d' % (major, minor)) in ''.join(torch.cuda.get_arch_list())}")
for name in ("numpy", "gym", "einops", "wandb", "ray"):
    try:
        mod = __import__(name)
        print(f"{name:<12} {getattr(mod, '__version__', '?')}")
    except Exception:
        print(f"{name:<12} missing")
PY
else
  echo "python not on PATH — activate the conda env and re-run for this block"
fi

hr; echo "[TIMING]  measured on the shapes this repository actually runs"
if python -c "import research" >/dev/null 2>&1; then
python - <<'PY' 2>&1
import time
import torch
import research  # noqa: F401
from research.config import build_learner_config, configure_torch
from research.env_adapter import MATEEnv
from agent.world_models.diffusion import Denoiser, DiffusionSampler

env = MATEEnv(scenario='MATE-4v8-9', seed=0, discrete_levels=5, max_episode_steps=200)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
cfg = build_learner_config(env, device=dev, horizon=5)
configure_torch(dev, detect_anomaly=False)
cfg.denoiser_cfg.inner_model.state_dim = cfg.STATE_DIM      # DreamerLearner.__init__:92-93
cfg.denoiser_cfg.inner_model.action_dim = cfg.ACTION_SIZE

den = Denoiser(cfg.denoiser_cfg, num_agents=cfg.NUM_AGENTS,
               clip_denoised=False, is_continuous_act=False).to(dev)
den.setup_training(cfg.sigma_distribution)
sampler = DiffusionSampler(den, cfg.diffusion_sampler_cfg)
sl, n, sd = cfg.denoiser_cfg.inner_model.num_steps_conditioning, cfg.NUM_AGENTS, cfg.STATE_DIM

def bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    if dev == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000.0

# 1. one sampler call at the imagination batch (DreamerLearnerConfig.ac_batch_size)
for batch, label in ((cfg.ac_batch_size, 'imagination'), (128, 'cem planner')):
    st = torch.randn(batch, sl, sd, device=dev)
    ac = torch.randint(0, cfg.ACTION_SIZE, (batch, sl, n), device=dev)
    ms = bench(lambda: sampler.sample(st, ac), 10)
    print(f"sampler      batch {batch:<4} K={cfg.diffusion_sampler_cfg.num_steps_denoising}  "
          f"{ms:8.2f} ms/call   ({label})")

# 2. one denoiser training step at the configured batch
b = cfg.denoiser_batch_size
opt = torch.optim.AdamW(den.parameters(), lr=1e-4)
obs = torch.randn(b, sl, sd, device=dev)
nxt = torch.randn(b, 1, sd, device=dev)
act = torch.randint(0, cfg.ACTION_SIZE, (b, sl, n), device=dev)
mask = torch.ones(b, sl, n, dtype=torch.long, device=dev)
sig = torch.full((b,), 1.0, device=dev)

def train_step():
    opt.zero_grad(set_to_none=True)
    cs = den.compute_conditioners(sig)
    noisy = den.apply_noise(nxt, sig, den.cfg.sigma_offset_noise)
    out = den.compute_model_output(noisy, obs, act, cs, mask)
    target = (nxt - cs.c_skip * noisy) / cs.c_out
    torch.nn.functional.mse_loss(out, target).backward()
    opt.step()

print(f"denoiser     batch {b:<4} fwd+bwd+step        {bench(train_step, 20):8.2f} ms/step")
if dev == 'cuda':
    print(f"peak_vram    {torch.cuda.max_memory_allocated()/1024**2:8.1f} MiB allocated so far")

# 3. the CPU floor: online collection is single-threaded Python, GPU idle
from research.planners import build_planner
from research.rollout import run_episode, to_dima_rollout
planner = build_planner('reactive_greedy', env, seed=0)
t = time.perf_counter()
for i in range(3):
    result = run_episode(env, planner, episode=i, seed=i, max_steps=200)
    to_dima_rollout(result, env.n_actions)
per_ep = (time.perf_counter() - t) / 3
print(f"env_step     single core                     {200/per_ep:8.0f} steps/s "
      f"({per_ep:.2f} s per 200-step episode)")
env.close()
PY
else
  echo "repository not importable here — run from the repo root inside the dima env"
fi

hr
echo "===== END SERVER PROBE ====="
