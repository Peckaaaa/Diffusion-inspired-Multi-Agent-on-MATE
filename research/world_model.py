"""World model interface and implementations (brief sections 11, 12, 13, 28, 29).

The planner never sees a DIMA tensor, a DIMA config, or a checkpoint.  It sees a
:class:`Prediction`.  Everything DIMA-specific is confined to
:class:`DIMAWorldModel`, which *reuses* DIMA's own modules -- it builds a real
``DreamerLearner`` and calls into its ``Denoiser`` / ``DiffusionSampler`` /
state decoder / ``TransRewEndModel`` unchanged.

Why not reuse ``DIMA/agent/world_models/world_model_env.py:WorldModelEnv`` directly
--------------------------------------------------------------------------------
``WorldModelEnv.reset()`` draws its conditioning window from a replay dataset
through ``make_generator_init`` (world_model_env.py:276-352).  It has no way to
be initialised from a *live* MATE observation, which is exactly what closed-loop
planning needs.  :class:`DIMAWorldModel` therefore reimplements only the buffer
bookkeeping of ``WorldModelEnv.step`` (about 20 lines, cited line by line below)
and delegates every model call to the same objects ``WorldModelEnv`` would use.

Interface deviation from the brief
----------------------------------
The brief sketches ``predict(obs_history, action_history)``.  DIMA's denoiser is
conditioned on the **global state** history, not the observation history
(``denoiser.forward`` uses ``batch.shared_obs``), so a signature taking only
observations could not be implemented faithfully.  :meth:`WorldModel.predict`
therefore takes a :class:`History` that carries observations, states and actions
together, per brief section 36 ("adapt the architecture to the real repositories").
"""

from __future__ import annotations

import abc
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import research  # noqa: F401 - installs sys.path + compat shims

from research.env_adapter import MATEEnv, MATEObservation


__all__ = [
    'History',
    'Prediction',
    'WorldModel',
    'DIMAWorldModel',
    'OracleWorldModel',
    'AlphaOracleWorldModel',
]


# --------------------------------------------------------------------------- #
# Standardized data carried across the interface
# --------------------------------------------------------------------------- #


@dataclass
class History:
    """Rolling conditioning window.

    ``length`` is the world model's conditioning horizon (3 for DIMA, from
    ``DreamerConfig.inner_model.num_steps_conditioning``).

    Alignment -- the one invariant everything else depends on
    ---------------------------------------------------------
    ::

        states[-1]   is  s_t        the current state
        actions[-1]  is  a_{t-1}    the action that produced s_t
        the candidate passed to predict() is a_t

    So ``actions`` lags ``states`` by one step, and the action slot paired with
    the current state is deliberately empty -- it is what a planner is choosing.
    :meth:`push` maintains this: it records ``(a_t, s_{t+1})`` together.

    DIMA's denoiser conditions on ``[s_{t-2}, s_{t-1}, s_t]`` with
    ``[a_{t-2}, a_{t-1}, a_t]`` (``denoiser.forward`` slices ``shared_obs`` and
    ``act`` with the same indices), so :meth:`DIMAWorldModel.predict` builds its
    action buffer as ``actions[-(sl-1):] + [candidate]``.  Getting this off by
    one is silent: the model still returns plausible states, just conditioned on
    the wrong action, and every action-sensitivity number collapses.

    ``states`` is PRIVILEGED: it holds ``MultiAgentTracking.state()`` rescaled.
    DIMA's denoiser is a *global state* model, so a closed-loop planner driven by
    it is reading privileged information at conditioning time.  This is recorded
    in every run manifest and warned about at construction; see the README's
    "Known limitations".
    """

    length: int
    observations: deque = field(default_factory=deque)  # each (C, obs_dim), rescaled
    states: deque = field(default_factory=deque)  # each (state_dim,), rescaled  [PRIVILEGED]
    actions: deque = field(default_factory=deque)  # each (C,), discrete indices

    def __post_init__(self) -> None:
        self.observations = deque(self.observations, maxlen=self.length)
        self.states = deque(self.states, maxlen=self.length)
        self.actions = deque(self.actions, maxlen=self.length)

    def reset(self, observation: MATEObservation, num_agents: int) -> None:
        """Seed the window by repeating the initial step.

        DIMA always conditions on exactly ``length`` steps; at ``t = 0`` there is
        no history, so the first observation is repeated.  The zero action used
        for the padding steps is action index 0, which for ``DiscreteCamera`` is
        a corner of the action grid, not a no-op -- the no-op is the grid centre.
        :meth:`DIMAWorldModel.noop_action` supplies the right index and
        :meth:`seed` uses it.
        """

        self.observations.clear()
        self.states.clear()
        self.actions.clear()
        for _ in range(self.length):
            self.observations.append(observation.obs)
            self.states.append(observation.state)
            self.actions.append(np.zeros(num_agents, dtype=np.int64))

    def seed(self, observation: MATEObservation, num_agents: int, noop_action: int) -> None:
        self.reset(observation, num_agents)
        filler = np.full(num_agents, int(noop_action), dtype=np.int64)
        self.actions = deque([filler.copy() for _ in range(self.length)], maxlen=self.length)

    def push(self, observation: MATEObservation, action: np.ndarray) -> None:
        """Record the action taken at time ``t`` and the observation at ``t+1``.

        Both go in together, which is what keeps ``actions[-1]`` the action that
        produced ``states[-1]`` (see the class docstring).
        """

        self.actions.append(np.asarray(action, dtype=np.int64).ravel())
        self.observations.append(observation.obs)
        self.states.append(observation.state)

    def conditioning_actions(self, steps: int) -> np.ndarray:
        """The ``steps - 1`` past actions that precede the action being chosen.

        Returns shape ``(steps - 1, num_agents)``.  A world model completes this
        with the candidate action to get a full ``steps``-long action window
        aligned with ``state_array()[-steps:]``.
        """

        if steps <= 1:
            width = self.action_array().shape[-1] if self.actions else 0
            return np.zeros((0, width), dtype=np.int64)
        past = self.action_array()
        if past.shape[0] < steps - 1:
            raise ValueError(
                f'History holds {past.shape[0]} actions but {steps - 1} are needed '
                f'to condition a {steps}-step window.'
            )
        return past[-(steps - 1) :]

    @property
    def is_full(self) -> bool:
        return len(self.states) == self.length

    def state_array(self) -> np.ndarray:
        return np.stack(list(self.states), axis=0)

    def action_array(self) -> np.ndarray:
        return np.stack(list(self.actions), axis=0)

    def observation_array(self) -> np.ndarray:
        return np.stack(list(self.observations), axis=0)


@dataclass(frozen=True)
class Prediction:
    """A world model's answer, in units a planner can reason about.

    ``observations`` are in **MATE world units** (already un-rescaled), so a
    planner can build a :class:`research.views.SceneView` from any horizon step
    without knowing which model produced it.

    Shapes, with ``B`` candidate action sequences, ``H`` horizon, ``C`` cameras:

    ``observations``  ``(B, H, C, obs_dim)``
    ``states``        ``(B, H, state_dim)``   -- model space (rescaled), PRIVILEGED
    ``rewards``       ``(B, H)`` or ``None``
    ``continues``     ``(B, H)`` or ``None``
    ``uncertainty``   ``(B, H)`` or ``None``  -- ``None`` means "not computed", reported as N/A
    ``actions``       ``(B, H, C)``           -- the action sequence each row answers for
    """

    observations: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    horizon: int
    rewards: Optional[np.ndarray] = None
    continues: Optional[np.ndarray] = None
    uncertainty: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_candidates(self) -> int:
        return int(self.observations.shape[0])

    def candidate(self, index: int) -> 'Prediction':
        """A single-candidate view, keeping the leading batch axis."""

        def take(arr):
            return None if arr is None else arr[index : index + 1]

        return Prediction(
            observations=self.observations[index : index + 1],
            states=self.states[index : index + 1],
            actions=self.actions[index : index + 1],
            horizon=self.horizon,
            rewards=take(self.rewards),
            continues=take(self.continues),
            uncertainty=take(self.uncertainty),
            metadata=dict(self.metadata),
        )


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class WorldModel(abc.ABC):
    """Everything a planner is allowed to know about a world model."""

    #: Conditioning window the model needs, in environment steps.
    conditioning_steps: int = 1
    #: True if the model reads MATE's privileged global state.
    uses_privileged_state: bool = True

    def __init__(self, name: str) -> None:
        self.name = name

    def reset(self, observation: MATEObservation) -> None:  # noqa: B027 - optional hook
        """Called once per episode, before the first prediction."""

    def observe(self, observation: MATEObservation, action: np.ndarray) -> None:  # noqa: B027
        """Called after every real environment step."""

    @abc.abstractmethod
    def predict(
        self,
        history: History,
        actions: np.ndarray,
        *,
        horizon: int = 1,
        **kwargs: Any,
    ) -> Prediction:
        """Predict the consequence of one or more candidate action sequences.

        Parameters
        ----------
        history:
            Conditioning window; ``history.actions[-1]`` is ignored, it is
            replaced by the candidate action for step 0.
        actions:
            ``(C,)``, ``(B, C)`` or ``(B, H, C)`` discrete joint actions.
        horizon:
            Number of steps to imagine.  Must equal ``H`` when a full
            ``(B, H, C)`` action sequence is given.
        """

    def describe(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'class': type(self).__name__,
            'conditioning_steps': self.conditioning_steps,
            'uses_privileged_state': self.uses_privileged_state,
        }

    # -- shared helper ------------------------------------------------------ #

    @staticmethod
    def _normalise_actions(actions, num_agents: int, horizon: int) -> np.ndarray:
        arr = np.asarray(actions, dtype=np.int64)
        if arr.ndim == 1:
            arr = arr[None, None, :]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        elif arr.ndim != 3:
            raise ValueError(f'actions must have 1, 2 or 3 dims, got {arr.shape}.')

        if arr.shape[1] == 1 and horizon > 1:
            arr = np.repeat(arr, horizon, axis=1)
        if arr.shape[1] != horizon:
            raise ValueError(
                f'actions horizon {arr.shape[1]} does not match requested horizon {horizon}.'
            )
        if arr.shape[2] != num_agents:
            raise ValueError(f'actions must have {num_agents} entries per step, got {arr.shape}.')
        return arr


# --------------------------------------------------------------------------- #
# DIMA
# --------------------------------------------------------------------------- #


class DIMAWorldModel(WorldModel):
    """DIMA's diffusion world model, driven from live MATE observations.

    Construction goes through :class:`DreamerLearner` so that the denoiser, the
    VQ state decoder and the reward/termination transformer are built by exactly
    the code that trained them (``DreamerLearner.__init__``:91-149) and loaded by
    exactly the code that saved them (``DreamerLearner.load_pretrained``).
    ``CAPACITY`` is reduced for inference because ``DreamerLearner`` also
    allocates the training replay buffers, which would otherwise reserve well
    over a gigabyte for a model we only want to run forward.
    """

    uses_privileged_state = True

    def __init__(
        self,
        env: MATEEnv,
        checkpoint_path: str,
        *,
        config=None,
        device: Optional[str] = None,
        name: str = 'dima',
        num_samples: int = 1,
    ) -> None:
        super().__init__(name)

        from research.config import (
            CHECKPOINT_CONFIG_FILENAME,
            allow_dima_checkpoint_globals,
            apply_checkpoint_config,
            build_learner_config,
        )

        allow_dima_checkpoint_globals()

        explicit_config = config is not None
        if config is None:
            config = build_learner_config(env, device=device)
        if device is not None:
            config.DEVICE = device

        # A checkpoint is only loadable with the shapes it was trained at; the
        # sidecar makes it self-describing.  See research/config.py.
        sidecar = None if explicit_config else apply_checkpoint_config(config, checkpoint_path)
        self.checkpoint_config = sidecar

        config.CAPACITY = max(2048, config.denoiser_cfg.inner_model.num_steps_conditioning + 8)
        config.load_pretrained = True
        config.load_path = checkpoint_path

        from agent.learners.DreamerLearner import DreamerLearner

        try:
            self._learner = DreamerLearner(config)
        except RuntimeError as exc:
            if 'size mismatch' not in str(exc):
                raise
            hint = (
                f'no {CHECKPOINT_CONFIG_FILENAME} sidecar was found next to the checkpoint, '
                f'so its training-time shapes are unknown'
                if sidecar is None
                else f'the sidecar says {sidecar}'
            )
            raise RuntimeError(
                f'Checkpoint {checkpoint_path} does not match the model built for this run.\n'
                f'{hint}.\n'
                f'The usual cause is `horizon`: it sizes TransRewEndModel\'s positional '
                f'embedding, causal masks and head slicers, so a model trained with '
                f'--horizon H can only be loaded at the same H (this run built '
                f'horizon={config.horizon}).\n'
                f'Original error: {exc}'
            ) from exc
        # DreamerLearner.__init__ turns on global anomaly detection for training;
        # this object only ever runs forward passes.
        torch.autograd.set_detect_anomaly(False)

        self.config = config
        self.env_metadata = env.metadata()
        self.device = torch.device(config.DEVICE)
        self.num_agents = int(config.NUM_AGENTS)
        self.num_actions = int(config.ACTION_SIZE)
        self.state_dim = int(config.STATE_DIM)
        self.obs_dim = int(config.IN_DIM)
        self.conditioning_steps = int(config.denoiser_cfg.inner_model.num_steps_conditioning)
        self.checkpoint_path = checkpoint_path
        #: >1 draws several diffusion samples per query and reports their spread
        #: as ``Prediction.uncertainty``.  The denoiser is stochastic, so this is
        #: sampling spread, not a calibrated uncertainty.
        self.num_samples = max(1, int(num_samples))

        from agent.world_models.diffusion import DiffusionSampler

        self.sampler = DiffusionSampler(self._learner.denoiser, config.diffusion_sampler_cfg)
        self.state_decoder = self._learner.state_decoder
        self.rew_end_model = self._learner.rew_end_model
        self.state_rms = self._learner.state_rms

        self._camera_observation_space = env.camera_observation_space
        self._unrescale_obs = env.unrescale_obs

        # DiscreteCamera's grid is a meshgrid over [-1, 1]^2; the centre index is
        # the zero-delta ("hold") action.  levels**2 // 2 is that centre for any
        # odd number of levels, which DiscreteCamera already enforces.
        self._noop_action = (env.discrete_levels**2) // 2

        self._rms_mean = torch.as_tensor(
            self.state_rms.mean, dtype=torch.float32, device=self.device
        )
        self._rms_std = torch.sqrt(
            torch.as_tensor(self.state_rms.var + 1e-8, dtype=torch.float32, device=self.device)
        )

    # -- properties -------------------------------------------------------- #

    @property
    def noop_action(self) -> int:
        """Discrete index of the zero rotation / zero zoom camera action."""

        return self._noop_action

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.update(
            checkpoint=self.checkpoint_path,
            num_steps_denoising=int(self.config.diffusion_sampler_cfg.num_steps_denoising),
            agent_order=str(self.config.diffusion_sampler_cfg.agent_order),
            vq_type=str(self.config.vq_type),
            num_samples=self.num_samples,
            device=str(self.device),
        )
        return info

    # -- normalisation ------------------------------------------------------ #

    def _normalize_state(self, states: torch.Tensor) -> torch.Tensor:
        """``DreamerLearner.normalize_state`` (line 266) on a ``(B, T, D)`` tensor."""

        return (states - self._rms_mean) / self._rms_std

    def _denormalize_state(self, states: torch.Tensor) -> torch.Tensor:
        return states * self._rms_std + self._rms_mean

    # -- prediction --------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self,
        history: History,
        actions: np.ndarray,
        *,
        horizon: int = 1,
        with_reward: bool = True,
        **kwargs: Any,
    ) -> Prediction:
        action_seq = self._normalise_actions(actions, self.num_agents, horizon)
        batch = action_seq.shape[0]
        sl = self.conditioning_steps

        past_states = history.state_array()[-sl:]  # (sl, state_dim)
        if past_states.shape[0] < sl:
            raise ValueError(
                f'History holds {past_states.shape[0]} steps but the model conditions on {sl}.'
            )
        # The last action slot belongs to the candidate, so only sl-1 past
        # actions are taken; see History's alignment invariant.
        conditioning_actions = history.conditioning_actions(sl)  # (sl-1, C)
        past_actions = np.concatenate(
            [conditioning_actions, np.zeros((1, self.num_agents), dtype=np.int64)], axis=0
        )

        samples = self.num_samples
        total = batch * samples

        states = torch.as_tensor(past_states, dtype=torch.float32, device=self.device)
        states = states.unsqueeze(0).expand(total, -1, -1).contiguous()
        states = self._normalize_state(states)
        # WorldModelEnv keeps the conditioning window in the denoiser's latent
        # scale (make_generator_init line 333).
        state_buffer = self.sampler.encode(states)

        act_buffer = self._one_hot(
            np.broadcast_to(past_actions, (total, sl, self.num_agents))
        )  # (total, sl, C, A)

        obs_out = np.empty((total, horizon, self.num_agents, self.obs_dim), dtype=np.float32)
        state_out = np.empty((total, horizon, self.state_dim), dtype=np.float32)
        rew_out = np.zeros((total, horizon), dtype=np.float32) if with_reward else None
        cont_out = np.zeros((total, horizon), dtype=np.float32) if with_reward else None

        # TransRewEndModel's positional embedding and causal masks are sized for
        # `max_blocks` steps (= the training horizon), and each imagined step
        # appends one block to the KV cache.  Rolling the *state* further is fine
        # -- the denoiser has no such limit -- so reward/termination is reported
        # for the first `max_blocks` steps and NaN afterwards, rather than
        # silently truncating the rollout or crashing inside the transformer.
        reward_steps = min(horizon, int(self.rew_end_model.config.max_blocks))
        if with_reward and reward_steps < horizon:
            rew_out[:, reward_steps:] = np.nan
            cont_out[:, reward_steps:] = np.nan

        kv_cache = attn_mask = None
        if with_reward:
            kv_cache, attn_mask = self._fresh_rew_end_cache(total)

        for h in range(horizon):
            step_action = np.repeat(action_seq[:, h], samples, axis=0)  # (total, C)
            act_buffer[:, -1] = self._one_hot(step_action[:, None, :])[:, 0]

            if with_reward and h < reward_steps:
                rew, cont, kv_cache, attn_mask = self._predict_rew_end(
                    state_buffer, act_buffer, kv_cache, attn_mask
                )
                rew_out[:, h] = rew
                cont_out[:, h] = cont

            # DiffusionSampler.sample returns the next state in latent scale.
            next_latent, _ = self.sampler.sample(state_buffer, act_buffer)
            next_state = self.sampler.decode(next_latent.squeeze(1))  # (total, state_dim)

            flat_obs = self.state_decoder.encode_decode(next_state)
            next_obs = flat_obs.reshape(total, self.num_agents, self.obs_dim)

            obs_out[:, h] = next_obs.detach().cpu().numpy()
            state_out[:, h] = next_state.detach().cpu().numpy()

            # WorldModelEnv.step lines 181-186: roll the window and append.
            state_buffer = state_buffer.roll(-1, dims=1)
            act_buffer = act_buffer.roll(-1, dims=1)
            state_buffer[:, -1] = next_latent.squeeze(1)

        obs_world = self._unrescale_obs(obs_out.astype(np.float64))

        if samples > 1:
            obs_world = obs_world.reshape(batch, samples, horizon, self.num_agents, self.obs_dim)
            state_out = state_out.reshape(batch, samples, horizon, self.state_dim)
            uncertainty = state_out.std(axis=1).mean(axis=-1)  # (B, H)
            obs_world = obs_world.mean(axis=1)
            state_mean = state_out.mean(axis=1)
            if rew_out is not None:
                rew_out = rew_out.reshape(batch, samples, horizon).mean(axis=1)
                cont_out = cont_out.reshape(batch, samples, horizon).mean(axis=1)
        else:
            state_mean = state_out
            uncertainty = None

        return Prediction(
            observations=obs_world,
            states=state_mean,
            actions=action_seq,
            horizon=horizon,
            rewards=rew_out,
            continues=cont_out,
            uncertainty=uncertainty,
            metadata={
                'model': self.name,
                'num_samples': samples,
                'observation_units': 'mate',
                'state_units': 'rms-normalized',
                'reward_context': 'per-call transformer cache (no cross-step burn-in)',
                'reward_steps': reward_steps if with_reward else 0,
                'max_reward_blocks': int(self.rew_end_model.config.max_blocks),
            },
        )

    # -- internals ---------------------------------------------------------- #

    def _one_hot(self, indices: np.ndarray) -> torch.Tensor:
        idx = torch.as_tensor(np.ascontiguousarray(indices), dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(idx, num_classes=self.num_actions).float()

    def _fresh_rew_end_cache(self, batch: int):
        kv = self.rew_end_model.transformer.generate_empty_keys_values(
            n=batch, max_tokens=self.rew_end_model.config.max_tokens
        )
        tokens = self.rew_end_model.config.tokens_per_block
        mask = torch.tril(torch.ones(tokens, tokens, device=self.device))[None].repeat(batch, 1, 1)
        return kv, mask

    def _predict_rew_end(self, state_buffer, act_buffer, kv_cache, attn_mask):
        """``WorldModelEnv.predict_rew_end`` (lines 244-274), one step."""

        current_state = self.sampler.decode(state_buffer[:, -1:].clone())
        act_cond = self.rew_end_model.get_act_emb(act_buffer[:, -1]).unsqueeze(1)
        model_input = torch.cat([current_state, torch.empty_like(current_state)], dim=1)

        out = self.rew_end_model(
            model_input,
            perattn_out=act_cond,
            past_keys_values=kv_cache,
            attention_mask=attn_mask,
        )
        rew = out.pred_rewards.float().squeeze(1).squeeze(-1)

        if self.rew_end_model.use_ce_for_end:
            end = torch.distributions.Categorical(logits=out.logits_ends).sample().squeeze(1)
            cont = (1 - end).float()
        else:
            cont = torch.sigmoid(out.logits_ends).squeeze(1).squeeze(-1)

        tokens = self.rew_end_model.config.tokens_per_block
        attn_mask = torch.cat(
            [attn_mask, torch.ones(attn_mask.shape[0], tokens, tokens, device=self.device)], dim=-1
        )
        return (
            rew.detach().cpu().numpy(),
            cont.detach().cpu().numpy().reshape(rew.shape[0], -1).mean(axis=-1),
            kv_cache,
            attn_mask,
        )


# --------------------------------------------------------------------------- #
# Oracles
# --------------------------------------------------------------------------- #


class OracleWorldModel(WorldModel):
    """Ground truth from MATE itself (brief section 28).

    Forks the live environment and actually steps it, so the "prediction" is the
    real consequence.  Running a planner against this and against
    :class:`DIMAWorldModel` separates *planner* failure from *world model* failure.

    Cost is ``B * H`` real MATE steps per query, so it is a diagnostic tool, not
    a training component.

    Caveats, which make this an oracle and not a perfect one: MATE's target team
    and its obstacle-transmittance checks are stochastic, and a fork advances its
    own RNG, so the ground truth returned here is *one sample* of the real
    dynamics rather than the branch the live environment will take.
    """

    conditioning_steps = 1
    uses_privileged_state = True

    def __init__(self, env: MATEEnv, name: str = 'oracle') -> None:
        super().__init__(name)
        self._env = env
        self.num_agents = env.n_agents
        self.obs_dim = env.n_obs
        self.state_dim = env.state_dim

    def predict(
        self,
        history: History,
        actions: np.ndarray,
        *,
        horizon: int = 1,
        **kwargs: Any,
    ) -> Prediction:
        action_seq = self._normalise_actions(actions, self.num_agents, horizon)
        batch = action_seq.shape[0]

        obs_out = np.empty((batch, horizon, self.num_agents, self.obs_dim), dtype=np.float64)
        state_out = np.empty((batch, horizon, self.state_dim), dtype=np.float64)
        rew_out = np.zeros((batch, horizon), dtype=np.float64)
        cont_out = np.ones((batch, horizon), dtype=np.float64)

        for b in range(batch):
            fork = self._env.fork()
            for h in range(horizon):
                observation, reward, done, _ = fork.step(action_seq[b, h])
                obs_out[b, h] = observation.obs_raw
                state_out[b, h] = observation.state
                rew_out[b, h] = float(np.mean(reward))
                cont_out[b, h] = 0.0 if done else 1.0
                if done:
                    obs_out[b, h + 1 :] = observation.obs_raw
                    state_out[b, h + 1 :] = observation.state
                    cont_out[b, h + 1 :] = 0.0
                    break
            fork.close()

        return Prediction(
            observations=obs_out,
            states=state_out,
            actions=action_seq,
            horizon=horizon,
            rewards=rew_out,
            continues=cont_out,
            uncertainty=None,
            metadata={'model': self.name, 'observation_units': 'mate', 'state_units': 'rescaled'},
        )


class AlphaOracleWorldModel(WorldModel):
    """Oracle with the action-dependent part of the prediction attenuated.

    Brief section 29's controlled experiment.  For a candidate action ``a``::

        prediction(a) = baseline + alpha * (oracle(a) - baseline) + noise

    where ``baseline`` is the *action-independent* component, taken as the mean
    oracle outcome over the candidate actions actually being compared.  At
    ``alpha = 1`` this is the oracle; at ``alpha = 0`` every action looks
    identical and any planner built on it degenerates to its tie-break rule.
    Sweeping ``alpha`` estimates how much action-dependent signal a world model
    must carry before model-based planning beats a reactive baseline.

    This is a **diagnostic only**: it forks the real environment and is never
    used to produce training data.
    """

    conditioning_steps = 1
    uses_privileged_state = True

    def __init__(
        self,
        env: MATEEnv,
        alpha: float = 1.0,
        noise_scale: float = 0.0,
        seed: int = 0,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name or f'alpha_oracle(alpha={alpha})')
        self.alpha = float(alpha)
        self.noise_scale = float(noise_scale)
        self._oracle = OracleWorldModel(env, name='oracle')
        self._rng = np.random.default_rng(seed)
        self.num_agents = env.n_agents

    def predict(self, history: History, actions: np.ndarray, *, horizon: int = 1, **kwargs):
        truth = self._oracle.predict(history, actions, horizon=horizon, **kwargs)

        baseline_obs = truth.observations.mean(axis=0, keepdims=True)
        baseline_state = truth.states.mean(axis=0, keepdims=True)
        baseline_rew = truth.rewards.mean(axis=0, keepdims=True)

        obs = baseline_obs + self.alpha * (truth.observations - baseline_obs)
        state = baseline_state + self.alpha * (truth.states - baseline_state)
        rew = baseline_rew + self.alpha * (truth.rewards - baseline_rew)

        if self.noise_scale > 0.0:
            scale = self.noise_scale * (np.std(truth.observations) + 1e-12)
            obs = obs + self._rng.normal(0.0, scale, size=obs.shape)

        return Prediction(
            observations=obs,
            states=state,
            actions=truth.actions,
            horizon=horizon,
            rewards=rew,
            continues=truth.continues,
            uncertainty=None,
            metadata={
                'model': self.name,
                'alpha': self.alpha,
                'noise_scale': self.noise_scale,
                'observation_units': 'mate',
                'diagnostic_only': True,
            },
        )
