# Modifications to the upstream repositories

Eleven files under `DIMA/` are modified. **`mate/` is byte-for-byte upstream.**
Patches 1-3 are purely additive. Patches 4-6 change behaviour, but each one is
gated on a config field that defaults to DIMA's original behaviour, so an
unmodified DIMA config still trains exactly as it did.

[`upstream-patches.diff`](upstream-patches.diff) currently carries patches 1 and 2
only; regenerate it against the pinned commits to include the checkpointing
methods of patch 3. It can be re-applied after rebasing onto a newer upstream:

```bash
git apply research/upstream-patches.diff
```

`bash research/scripts/verify_upstream.sh` re-clones both repositories at the
pinned commits and confirms these are the *only* differences.

---

## 1. `DIMA/environments.py` — 1 line

```python
class Env(str, Enum):
    ...
    SMACv2 = "SMACv2"
+   MATE = "mate"
```

### Why an adapter cannot solve this

`Env` is a `str`-valued `Enum`. Python enums cannot be extended with new members
from outside the class, and `DreamerLearner` dispatches on `self.env_type` with
`==` and `in [...]` against `Env` members. There is no injection point.

### The alternative, and why it was rejected

Setting `ENV_TYPE = Env.MAMUJOCO` works — every `env_type` branch on the world-model
training path is either `in [Env.STARCRAFT, Env.SMACv2]` (false for MATE, which is
correct: MATE has no action mask) or affects only actor-critic training, which this
project disables. It needs **zero** upstream edits.

It was rejected because the label is then a lie: every wandb run, every TensorBoard
tag, every `config.to_dict()` dump and every log line would say `mamujoco` for a
camera-tracking experiment. For a project whose whole purpose is attributing failure
to a layer, a mislabelled environment is exactly the wrong trade. One line of
upstream is cheaper than a permanently confusing audit trail.

---

## 2. `DIMA/agent/learners/DreamerLearner.py` — 15 lines (`add_experience_to_dataset`)

Adds an `elif self.env_type == Env.MATE:` branch to `add_experience_to_dataset()`
that builds a `MamujocoEpisode` from the rollout dictionary.

### Why an adapter cannot solve this

`add_experience_to_dataset` is called from `DreamerLearner.step()`, which is *the*
entry point into DIMA's training. Its `else` clause is `raise NotImplementedError`,
so an unknown `env_type` cannot reach the replay buffer at all. The method is not a
hook, takes no strategy object, and is not passed anything the caller controls
besides the data dictionary.

Monkey-patching it from the research layer was considered and rejected: it would
put a silent, invisible override on the hot path of the exact component whose
correctness this project is trying to establish.

### Why `MamujocoEpisode` and not a new episode class

DIMA already has an episode dataclass with exactly the seven fields MATE produces
and no `av_action` field:

```python
MamujocoEpisode(observation, shared_obs, next_shared_obs, action, reward, done, filled)
```

MATE has no action mask (every camera action is always legal), so this is the right
shape. Adding a `MATEEpisode` would have duplicated `MamujocoEpisode.segment()` and
`__len__()` verbatim — brief section 10 says to adapt MATE into DIMA's existing
abstraction rather than introducing another one. The new branch is a copy of the
MAMuJoCo branch because the data shape is genuinely identical; only the label differs.

---

## 3. Resumable checkpointing — `DreamerLearner.py`, `dataset.py`, `DreamerMemory.py`

Adds `training_state()` / `save_full()` / `load_full()` to `DreamerLearner`, and a
`state_dict()` / `load_state_dict()` pair to `MultiAgentEpisodesDataset`
(`DIMA/dataset.py`) and `DreamerMemory` so the replay buffers can travel inside
that checkpoint.

### Why an adapter cannot solve this

`DreamerLearner.save()` writes weights only: no optimiser state, no LR-schedule
state, no RNG, no counters, and in particular not the `cur_wandb_epoch` /
`accum_samples` / `train_count` fields that decide what `step()` does next. A run
restarted from `save()` therefore restarts DIMA's schedule from its first-epoch
branch on already-trained weights, which is not a resume. Reconstructing all of
that from outside would mean reaching into private attributes of five modules and
two buffers, and would break silently whenever upstream adds one.

The methods are additive: nothing upstream calls them, and `save()` /
`load_pretrained()` are untouched, so the existing training and evaluation paths
behave exactly as before.

### Scope

`research/train_wm.py --resume` is the only caller. The replay buffer is included
only when `--save-buffer` is passed, because `MultiAgentEpisodesDataset` holds raw
rollouts and the checkpoint would otherwise grow by gigabytes.

---

## 4. Joint action conditioning --- `perceiver.py`, `inner_model.py`, `denoiser.py`, `diffusion_sampler.py`

Adds `JointActionEmb` next to `SequentialActionEmb`, a
`StateInnerModelConfig.action_cond` field selecting between them, and the two
branches in `Denoiser.forward` / `DiffusionSampler.sample` that stop building a
single-agent `act_mask` when the joint encoder is selected.

### The defect

`DiffusionSampler.sample` exposed one agent's action per denoising step through
`act_mask`, and the Perceiver's cross-attention mask hid the other three. Sigma
falls across those steps, so the agent that landed on the last, lowest-sigma step
is the one whose action survives into the output. Measured with common random
numbers on a trained MATE-4v8-9 checkpoint: the last-step camera moved the
predicted state by 0.1029, the first-step one by 0.0106. Reversing `agent_order`
mirrored the split exactly, which is what identifies it as an artefact of the
schedule rather than of the environment.

Any planner built on that model is choosing three of four camera actions almost
blind, and the `tiled` order added earlier only spreads the problem thinner
rather than removing it.

### The change

`action_cond = 'joint'` embeds each agent's action with its own
`nn.Embedding(num_actions, 64)` table, concatenates them into the joint action,
adds a conditioning-step embedding and projects with a two-layer MLP. The result
enters the existing `cond` vector, and therefore every ResBlock's `AdaGroupNorm`
--- which is already FiLM (`x * (1 + scale) + shift`), so no new modulation
mechanism is introduced. Every denoising step sees the same, whole joint action.

Measured on the untrained wiring, averaged over 5 seeds
(`research/scripts/check_action_symmetry.py`), the per-camera effect ratio is
1.12:1 under `joint` and 2.95:1 under `sequential`.

### Knock-on simplification

`num_steps_denoising` no longer has to be a multiple of the agent count: nothing
about the schedule encodes agent identity any more. `--num-denoising-steps` is
free under `joint` and still validated under `sequential`.

### Default

`StateInnerModelConfig.action_cond` defaults to `'sequential'`, so DIMA is
unchanged. `research/config.py` sets `'joint'` for MATE, and records it in the
checkpoint sidecar; a sidecar without the field describes a checkpoint that
predates it and is therefore read as `'sequential'`.

---

## 5. A separate decoder branch for 0/1 flags --- `vq.py`, `DreamerLearner.py`

Adds `_BinaryFlagHead` plus a `binary_indices` argument to both autoencoders, and
switches `DreamerLearner._reconstruction_loss`'s flag block to
`BCEWithLogitsLoss(pos_weight=...)`.

### The defect

On MATE-4v8-9 the state decoder reconstructs 4 x 21 = 84 sighting / obstacle /
teammate flags together with 420 continuous channels, from 12 quantised tokens.
The flags are positive 13.6% of the time. Under the shared L1 head their optimum
is the conditional median --- 0 --- so the decoder predicted "never"; sighting
recall stalled at 0.25-0.31. Weighting a squared error instead moved the optimum
to the conditional mean but left the rare class contributing 13.6% of the
gradient, which was not enough.

### The change

The flags get their own two-layer head off the same quantised latent, emitting
logits, and a positive-weighted BCE. `pos_weight` defaults to 6.0, the
`(1 - p) / p` ratio at the measured positive rate.

`forward` (the training path) scatters the head's **logits** into the flag
channels; `encode_decode` (every inference path: `WorldModelEnv.step`,
`DIMAWorldModel.predict`, `log_compounding_errors`) scatters `sigmoid(logits)`.
Every downstream reader already applies a 0.5 cut to those channels and keeps
working unchanged, and MATE rescales a 0/1 flag to itself, so the cut still means
what it did.

### Default

`obs_binary_head` defaults to `False`; the branch is built only when the
environment names flag channels *and* the run asks for it. It is recorded in the
sidecar, and its absence there means a checkpoint written before it existed.

---

## 6. Imagined reward reads the successor observation --- `env_loop.py`, `loss.py`, `DreamerLearner.py`

`rollout_policy_with_env{,_wo_reset}` record `info['next_obs']`,
`rollout_diffusion_world_models` returns the stacked sequence, and
`DreamerLearner.imagined_coverage` scores that instead of `obs`.

### The defect

The rollout's `obs[:, n]` is the observation the policy *acted on*, so a coverage
reward read from it does not depend on action `n` at all --- it was settled before
the action was taken. The first fix shifted the series by one, which is right for
every step but the last, whose successor is not in `obs` at all and which
therefore kept its own value.

### The change

The successor observation is recorded where the world model produces it, so the
reward for acting at step `n` is the coverage of the state that action reached,
for every step including the last. Nothing is reconstructed by shifting.

### Default

Only read when `imagined_reward == 'coverage'`, which is off by default. The extra
tensor is collected unconditionally; it is one imagined observation per step, next
to the denoising trajectory the same dict already carries.

---

## 7. Sampling buffers pinned to the CPU --- `DreamerMemory.py`

`sample_visits` is allocated with `torch.zeros(capacity, dtype=torch.long)` in
`init_buffer` and `clean`, and `validate_indices` builds its offsets with a bare
`torch.arange`. Neither names a device, so both follow the process default.

### The failure

`_compute_visit_probs` ends with `assert probs.device.type == 'cpu'` and
`sample_indices` is commented "stay on cpu", so the sampling path requires these
on the host. On a machine where the torch default device is CUDA they are
allocated on the GPU instead, and the first training round --- the first
`DreamerLearner.step` that passes both gates, which on a server preset is several
minutes in --- dies on a bare `AssertionError` with no message.

### The change

`device="cpu"` on the three `sample_visits` allocations and on the `arange`, and
`load_state_dict` moves a restored buffer to the CPU rather than inheriting the
device it was saved from. On a host whose default is already the CPU every one of
these is a no-op, so the behaviour is unchanged where it already worked.

---

## Compatibility handled *without* touching upstream

For contrast, these four problems were all solved inside `research/` and required
no upstream change:

| Problem | Where it is solved |
|---|---|
| `np.bool8` removed in NumPy 2.0 (MATE uses it in 6 modules) | `research/_compat.py` |
| gym ≥ 0.26 returns `np.random.Generator`, MATE calls `randint`/`rand`/`randn` | `research/_compat.py` |
| gym ≥ 0.26 `gym.make` wraps in `PassiveEnvChecker`, which rejects MATE's 4-value `step()` and 2-tuple `reset()` | `research/env_adapter.py` uses `mate.make_environment`; `research/mate_evaluate.py` redirects `mate.make` |
| PyTorch ≥ 2.6 `torch.load(weights_only=True)` refuses DIMA's `RunningMeanStd` checkpoint entry | `research.config.allow_dima_checkpoint_globals()` |
| `DreamerLearner.step` trains actor-critic, which this project does not want | configuration only: `EPOCHS = ac_steps_first_epoch = 0` |
| `rew_end_model_type` defaults to `'rnn'`, but `WorldModelEnv` requires `TransRewEndModel` | configuration only: `MATEDreamerLearnerConfig` sets `'transformer'` |
| `DEVICE` hardcoded to `'cuda'` | configuration only: `research.config.default_device()` |

## What upstream still does on its own

Both repositories remain independently usable:

```bash
cd DIMA && python train.py --env mamujoco --env_name HalfCheetah-v2 ...   # unchanged
cd mate && python -m mate.evaluate --camera-agent mate:GreedyCameraAgent  # unchanged
```

(The MATE command needs `research/_compat.py`'s shims on gym ≥ 0.26 / NumPy ≥ 2 —
run `python -m research.mate_evaluate` with the same arguments, or pin
`gym==0.25.2` and `numpy<2`, which makes both shims inert.)
