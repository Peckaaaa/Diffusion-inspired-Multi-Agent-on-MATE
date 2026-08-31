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

import torch

import research  # noqa: F401 - installs sys.path + compat shims

from agent.world_models.vq import StateDecoderType
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
from environments import Env

from research.env_adapter import MATEEnv


__all__ = [
    'MATEDreamerLearnerConfig',
    'MATEDreamerControllerConfig',
    'apply_env_info',
    'build_learner_config',
    'default_device',
    'allow_dima_checkpoint_globals',
    'export_checkpoint_config',
    'apply_checkpoint_config',
    'CHECKPOINT_CONFIG_FIELDS',
    'CHECKPOINT_CONFIG_FILENAME',
]


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

    if device is not None:
        config.DEVICE = device

    if not train_actor_critic:
        config.EPOCHS = 0
        config.ac_steps_first_epoch = 0

    config.seed = int(seed)
    config.RUN_DIR = str(run_dir)
    config.map_name = env.scenario

    for key, value in (overrides or {}).items():
        if not hasattr(config, key):
            raise AttributeError(f'Unknown DIMA config field {key!r}.')
        setattr(config, key, value)

    return config
