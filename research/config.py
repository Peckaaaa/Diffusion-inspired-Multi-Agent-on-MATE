"""DIMA configuration for MATE, built from the live environment.

DIMA configures itself with **Python config classes**, not YAML
(``DIMA/configs/Config.py``, ``DIMA/configs/dreamer/DreamerAgentConfig.py``).  This
module therefore does not introduce a competing configuration format: it
subclasses DIMA's own ``DreamerLearnerConfig`` / ``DreamerControllerConfig`` and
fills in the environment-dependent fields the same way DIMA's ``train.py`` does.

Brief section 4: no environment dimension is written by hand.  :func:`apply_env_info`
is the exact assignment block from ``DIMA/train.py:82-88``, applied to a
:class:`research.env_adapter.MATEEnv` instead of a SMAC/MPE/MAMuJoCo environment.

The three settings that MATE genuinely forces are documented at their assignment
site below; everything else is inherited from DIMA unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

import research  # noqa: F401 - installs sys.path + compat shims

from agent.world_models.vq import StateDecoderType
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env

from research.env_adapter import MATEEnv
from research.views import ObservationLayout


__all__ = [
    'MATEDreamerLearnerConfig',
    'MATEDreamerControllerConfig',
    'apply_env_info',
    'build_learner_config',
    'default_device',
    'configure_torch',
    'allow_dima_checkpoint_globals',
    'export_checkpoint_config',
    'apply_checkpoint_config',
    'CHECKPOINT_CONFIG_FIELDS',
    'CHECKPOINT_CONFIG_FILENAME',
]


def resolve_weights_checkpoint(checkpoint) -> Path:
    """A checkpoint path in the flat shape ``DreamerLearner.load_pretrained`` reads.

    Two different files get called "the checkpoint" in this project.
    ``model_*.pth`` holds ``DreamerLearner.params()`` -- a flat dict of module
    state -- while ``ckpt/latest.pth`` is the resumable checkpoint from
    ``save_full``, which wraps those modules together with optimiser state,
    counters and RNG state under a ``'learner'`` key.

    ``load_pretrained`` only understands the flat shape, and on PyTorch >= 2.6 it
    does not even get that far: it calls ``torch.load`` with the new
    ``weights_only=True`` default, which refuses the resumable file's numpy RNG
    state outright. Since the resumable one is the freshest state a running job
    has on disk, and therefore the natural thing to evaluate or probe, the weights
    are extracted here into a sibling file rather than being rejected.

    A checkpoint already in the flat shape is returned unchanged.
    """

    from agent.utils.running_mean_std import RunningMeanStd

    checkpoint = Path(checkpoint)
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'learner' not in ckpt:
        return checkpoint

    state = ckpt['learner']
    rms_fields = state['state_rms']
    running_mean_std = RunningMeanStd(shape=np.asarray(rms_fields['mean']).shape)
    running_mean_std.mean = np.asarray(rms_fields['mean'], dtype=np.float64)
    running_mean_std.var = np.asarray(rms_fields['var'], dtype=np.float64)
    running_mean_std.count = float(rms_fields['count'])

    flat = {
        name: state['modules'][name]
        for name in ('state_decoder', 'denoiser', 'rew_end_model', 'actor', 'critic')
        if name in state['modules']
    }
    flat['running_mean_std'] = running_mean_std

    extracted = checkpoint.with_name(checkpoint.stem + '_weights.pth')
    torch.save(flat, extracted)
    return extracted


def allow_dima_checkpoint_globals() -> None:
    """Let ``torch.load`` read DIMA checkpoints on PyTorch >= 2.6.

    ``DreamerLearner.params()`` stores the state normaliser as a live
    ``RunningMeanStd`` object, not a tensor (DreamerLearner.py:236).  PyTorch 2.6
    flipped ``torch.load``'s ``weights_only`` default to ``True``, which refuses
    to unpickle arbitrary classes, so ``DreamerLearner.load_pretrained`` -- which
    calls plain ``torch.load(load_path)`` -- raises ``UnpicklingError``.

    Allow-listing that one class is PyTorch's own recommended remedy and keeps
    ``load_pretrained`` untouched.  It is safe here because the only checkpoints
    this project loads are ones it wrote itself; the allow-list is deliberately
    limited to that single class rather than disabling ``weights_only``.
    """

    add_safe_globals = getattr(torch.serialization, 'add_safe_globals', None)
    if add_safe_globals is None:  # PyTorch < 2.4: weights_only defaulted to False
        return

    import numpy as np

    from agent.utils.running_mean_std import RunningMeanStd

    allowed = [RunningMeanStd, np.ndarray, np.dtype, np.float64, np.int64]
    # RunningMeanStd carries numpy arrays, whose unpickling needs the array
    # reconstructor and the dtype constructors as well.  Their module path moved
    # in NumPy 2, so both spellings are tried.
    for module_path, attribute in (
        ('numpy._core.multiarray', '_reconstruct'),
        ('numpy.core.multiarray', '_reconstruct'),
        ('numpy._core.multiarray', 'scalar'),
        ('numpy.core.multiarray', 'scalar'),
    ):
        try:
            module = __import__(module_path, fromlist=[attribute])
            allowed.append(getattr(module, attribute))
        except (ImportError, AttributeError):
            continue

    for dtype_name in ('Float64DType', 'Int64DType', 'BoolDType'):
        dtypes = getattr(np, 'dtypes', None)
        if dtypes is not None and hasattr(dtypes, dtype_name):
            allowed.append(getattr(dtypes, dtype_name))

    add_safe_globals(allowed)


def default_device() -> str:
    """DIMA's configs hardcode ``'cuda'``; this project must also run on laptops."""

    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        # MPS lacks kernels DIMA's diffusion path uses; CPU is the safe fallback.
        return 'cpu'
    return 'cpu'


def configure_torch(
    device: str,
    *,
    detect_anomaly: bool = False,
    tf32: bool = True,
    matmul_precision: str = 'high',
    threads: Optional[int] = None,
    cudnn_benchmark: bool = True,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Put PyTorch in the right mode before any DIMA module is built.

    Call this *before* constructing ``DreamerLearner``, and once more after (it is
    idempotent) because ``DreamerLearner.__init__`` re-enables anomaly detection.

    ``detect_anomaly`` is the one that matters most on a GPU. DIMA turns
    ``torch.autograd.set_detect_anomaly(True)`` on globally, in
    ``train.py`` *and* in ``DreamerLearner.__init__`` (line 83). It is a debugging
    aid: it records a traceback for every op in the graph and re-runs the backward
    pass to find NaNs, which costs a large multiple of normal training time and
    holds extra memory. Leaving it on for a real training run is a mistake, so it
    is off here by default and restored only with ``--detect-anomaly``.

    ``tf32`` and ``matmul_precision`` let Ampere-and-later cards use TensorFloat-32
    for matmuls and convolutions. The denoiser is matmul-bound, so this is the
    single biggest free speedup; it costs a little mantissa precision, which
    diffusion training tolerates. ``cudnn_benchmark`` lets cuDNN pick algorithms
    per shape -- worth it because shapes here are fixed after the first batch.

    Returns what was actually applied, so the run manifest records it.
    """

    applied: Dict[str, Any] = {'device': device, 'detect_anomaly': bool(detect_anomaly)}

    torch.autograd.set_detect_anomaly(bool(detect_anomaly))

    if threads:
        torch.set_num_threads(int(threads))
    applied['torch_threads'] = torch.get_num_threads()

    if device.startswith('cuda') and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision(matmul_precision)
        # TF32 is an Ampere-and-later tensor-core format: compute capability 8.0+
        # (RTX 30 = 8.6, RTX 40 = 8.9, RTX 50 = 12.0).  On Turing and older the
        # flags above are accepted and silently do nothing, so the manifest must
        # not record TF32 as active there -- the whole point of writing it down is
        # to know afterwards which run got the speedup.
        major, minor = torch.cuda.get_device_capability(0)
        tf32_supported = major >= 8
        applied.update(
            tf32=bool(tf32) and tf32_supported,
            tf32_requested=bool(tf32),
            tf32_supported=tf32_supported,
            cudnn_benchmark=bool(cudnn_benchmark),
            matmul_precision=matmul_precision,
            gpu_name=torch.cuda.get_device_name(0),
            gpu_count=torch.cuda.device_count(),
            gpu_capability=f'{major}.{minor}',
            gpu_total_memory_gb=round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
            ),
        )
    else:
        applied['cuda_available'] = torch.cuda.is_available()

    if seed is not None:
        import random as _random

        import numpy as _np

        torch.manual_seed(seed)
        _np.random.seed(seed)
        _random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        applied['seed'] = int(seed)

    return applied


class _MATEConfigMixin:
    """The MATE-specific deviations from DIMA's defaults, in one place."""

    def _apply_mate_defaults(self) -> None:
        self.ENV_TYPE = Env.MATE

        # REQUIRED, not a preference: WorldModelEnv.__init__ reads
        # `rew_end_model.config.tokens_per_block` and later uses `.transformer`
        # with KV caching (DIMA/agent/world_models/world_model_env.py:64,103,255).
        # Only TransRewEndModel provides those; the RewEndModel (rnn) default
        # inherited from DreamerConfig raises AttributeError as soon as the world
        # model is rolled out.
        self.rew_end_model_type = 'transformer'

        # MATE camera actions are a finite Discrete(levels**2) set.
        self.CONTINUOUS_ACTION = False
        self.ACTION_SPACE = None
        self.policy_class = 'discrete'

        # "s + id": the state decoder maps the global state plus an agent id to
        # that agent's observation.  Option 2 ("s + last_obs") would make the
        # decoder input depend on the previous observation, which the closed-loop
        # planner does not have for imagined states.
        self.state_decoder_type = StateDecoderType.OPTION1

        # Set by train.py from CLI flags rather than by the config classes, so
        # they must be provided explicitly here (DIMA/train.py:210-211, 217-218).
        self.use_ce_for_cont = False
        self.compute_end_in_TD = False
        self.load_pretrained = False
        self.load_path = None
        self.sample_temperature = 'inf'

        self.DEVICE = default_device()


class MATEDreamerLearnerConfig(_MATEConfigMixin, DreamerLearnerConfig):
    """``DreamerLearnerConfig`` with the MATE deviations applied."""

    def __init__(self) -> None:
        super().__init__()
        self._apply_mate_defaults()


class MATEDreamerControllerConfig(_MATEConfigMixin, DreamerControllerConfig):
    """``DreamerControllerConfig`` with the MATE deviations applied.

    Only needed by the online (Ray/DreamerWorker) route.  ``epsilon`` must stay
    at DIMA's default of ``0.``: ``DreamerController.step``'s epsilon branch
    samples from ``avail_actions``, which is ``None`` for MATE
    (``DIMA/agent/controllers/DreamerController.py:134-136``).
    """

    def __init__(self) -> None:
        super().__init__()
        self._apply_mate_defaults()
        assert self.epsilon == 0.0, 'MATE provides no action mask; epsilon exploration would crash.'


#: Config fields that determine module *shapes*, and therefore must match between
#: the run that wrote a checkpoint and the run that loads it.
#:
#: ``horizon`` is the sharp one: it feeds ``trans_config.max_blocks``, which sizes
#: ``TransRewEndModel``'s positional embedding, causal masks and head slicers.
#: Training at ``--horizon 5`` and loading with the default 15 fails with a wall of
#: ``size mismatch`` errors that say nothing about horizons.
CHECKPOINT_CONFIG_FIELDS = (
    'horizon',
    'IN_DIM',
    'STATE_DIM',
    'ACTION_SIZE',
    'NUM_AGENTS',
    'CONTINUOUS_ACTION',
    'vq_type',
    'nums_obs_token',
    'EMBED_DIM',
    'OBS_VOCAB_SIZE',
    'rew_end_model_type',
    'TRANS_EMBED_DIM',
    'HEADS',
    'state_decoder_type',
    'use_ce_for_cont',
)

CHECKPOINT_CONFIG_FILENAME = 'config.json'


def export_checkpoint_config(config, path) -> 'Path':
    """Write the shape-determining fields next to a checkpoint.

    Without this, a checkpoint is not self-describing and reloading it depends on
    the caller happening to pass the same ``--horizon``.
    """

    from pathlib import Path as _Path

    payload = {field: _jsonable(getattr(config, field)) for field in CHECKPOINT_CONFIG_FIELDS}
    payload['num_steps_denoising'] = int(config.diffusion_sampler_cfg.num_steps_denoising)
    payload['agent_order'] = str(config.diffusion_sampler_cfg.agent_order)
    payload['num_steps_conditioning'] = int(
        config.denoiser_cfg.inner_model.num_steps_conditioning
    )

    destination = _Path(path)
    destination.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return destination


def apply_checkpoint_config(config, checkpoint_path) -> Optional[Dict[str, Any]]:
    """Apply the sidecar written by :func:`export_checkpoint_config`, if present.

    Returns the loaded mapping, or ``None`` when the checkpoint has no sidecar
    (checkpoints written before this existed, or copied without it).
    """

    from pathlib import Path as _Path

    sidecar = _Path(checkpoint_path).parent / CHECKPOINT_CONFIG_FILENAME
    if not sidecar.is_file():
        return None

    payload = json.loads(sidecar.read_text(encoding='utf-8'))
    for field in CHECKPOINT_CONFIG_FIELDS:
        if field not in payload:
            continue
        value = payload[field]
        if field == 'state_decoder_type':
            value = StateDecoderType(value)
        setattr(config, field, value)

    config.SEQ_LENGTH = config.horizon
    config.worldmodel_env_cfg.horizon = config.horizon
    config.trans_config.max_blocks = config.horizon
    if 'num_steps_denoising' in payload:
        config.diffusion_sampler_cfg.num_steps_denoising = int(payload['num_steps_denoising'])
    if payload.get('agent_order'):
        config.diffusion_sampler_cfg.agent_order = str(payload['agent_order'])
    return payload


def _jsonable(value):
    if isinstance(value, StateDecoderType):
        return value.value
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return str(value)


def apply_env_info(configs, env: MATEEnv) -> None:
    """Copy the environment's dimensions into DIMA configs.

    This is ``DIMA/train.py:82-88`` verbatim.  It is duplicated rather than
    imported because importing ``train`` pulls in ``configs.EnvConfigs``, whose
    module-level imports require SMAC, SMACv2, PettingZoo+SuperSuit, Google
    Research Football and MuJoCo to all be installed.
    """

    for config in configs:
        config.IN_DIM = env.n_obs
        config.STATE_DIM = env.state_dim
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = env.individual_action_space


def build_learner_config(
    env: MATEEnv,
    *,
    seed: int = 0,
    run_dir: str = '.',
    device: Optional[str] = None,
    horizon: Optional[int] = None,
    train_actor_critic: bool = False,
    overrides: Optional[Dict[str, Any]] = None,
) -> MATEDreamerLearnerConfig:
    """Build the DIMA learner config for a given MATE environment.

    Parameters
    ----------
    train_actor_critic:
        The research pipeline replaces DIMA's actor with a planner, so actor-critic
        training is off by default.  It is disabled *through configuration*
        (``EPOCHS = ac_steps_first_epoch = 0`` makes ``DreamerLearner.step``'s
        actor-critic loop iterate zero times), not by editing DIMA.
    overrides:
        Any further attribute assignments, applied last.  Use this for sweeps
        rather than adding new config classes.
    """

    config = MATEDreamerLearnerConfig()
    apply_env_info([config], env)

    if horizon is not None:
        config.horizon = int(horizon)
        config.SEQ_LENGTH = config.horizon
        config.worldmodel_env_cfg.horizon = config.horizon
        config.trans_config.max_blocks = config.horizon

    # DIMA/train.py:212-213 -- the diffusion sampler denoises one agent's action
    # per step, so the number of denoising steps must be a multiple of the number
    # of agents (DiffusionSampler.sample_agent_order asserts this).
    config.diffusion_sampler_cfg.num_steps_denoising = (
        config.NUM_AGENTS if config.NUM_AGENTS > 2 else config.NUM_AGENTS * 2
    )

    # Sampler knobs are reachable through `overrides` under these names, because
    # they live on a nested dataclass that the flat override loop below cannot
    # address.  Both change how much of each agent's action survives to the
    # predicted state: see DiffusionSampler.sample_agent_order.
    sampler_overrides = {
        key: (overrides or {}).pop(key)
        for key in ('num_steps_denoising', 'agent_order')
        if key in (overrides or {})
    }
    for key, value in sampler_overrides.items():
        setattr(
            config.diffusion_sampler_cfg,
            key,
            int(value) if key == 'num_steps_denoising' else str(value),
        )
    steps = config.diffusion_sampler_cfg.num_steps_denoising
    if steps % config.NUM_AGENTS:
        raise ValueError(
            f'num_steps_denoising={steps} must be a multiple of the {config.NUM_AGENTS} '
            f'agents; DiffusionSampler.sample_agent_order asserts this.'
        )

    if device is not None:
        config.DEVICE = device

    if not train_actor_critic:
        config.EPOCHS = 0
        config.ac_steps_first_epoch = 0

    config.seed = int(seed)
    config.RUN_DIR = str(run_dir)
    config.map_name = env.scenario

    # Which observation channels are 0/1 presence flags, so the state decoder can
    # score them with a mean-seeking loss.  DIMA has no idea what MATE's
    # observation contains, so the layout is resolved here and handed down as
    # plain indices.  See MATEDreamerLearnerConfig.obs_binary_indices.
    layout = ObservationLayout.from_env_metadata(env.metadata())
    config.obs_binary_indices = layout.joint_binary_channel_indices(config.NUM_AGENTS).tolist()

    for key, value in (overrides or {}).items():
        if not hasattr(config, key):
            raise AttributeError(f'Unknown DIMA config field {key!r}.')
        setattr(config, key, value)

    return config
