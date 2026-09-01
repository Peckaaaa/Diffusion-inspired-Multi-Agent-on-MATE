# Modifications to the upstream repositories

Four files under `DIMA/` are modified. **`mate/` is byte-for-byte upstream.**
Every edit is additive -- a new `Env.MATE` branch, or a new method -- and no
existing code path changes behaviour.

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
