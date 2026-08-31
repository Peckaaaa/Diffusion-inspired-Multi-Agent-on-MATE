# Upstream provenance

`DIMA/` and `mate/` in this repository are **vendored copies of the official
upstream repositories, byte-for-byte**, at the commits recorded below. They are
tracked as ordinary directories (the nested `.git/` was removed) so that this
repository has a single, self-contained history.

| Vendored path | Upstream | Commit | Date | Upstream version |
|---|---|---|---|---|
| `DIMA/`  | <https://github.com/breez3young/DIMA>  | `3dcacaa80162cf6822bf5972b4e3ad4cb2e6ceb0` | 2025-11-11 | — (single-commit repo) |
| `mate/`  | <https://github.com/XuehaiPan/mate>    | `3e631c0c3b043990fc53ae5fc3a37b0f65f230c5` | 2023-03-31 | `0.1.0` |

## Verifying the vendored copies

```bash
# from the repository root
bash research/scripts/verify_upstream.sh
```

The script re-clones both repositories into a temporary directory at the pinned
commits and diffs them against the vendored trees. A clean run prints only the
files listed under "Accepted modifications" below.

## Accepted modifications to upstream

Everything the research layer needs lives in `research/`. Only the two edits
below touch upstream source; both are additive and change no existing
behaviour. See `research/UPSTREAM_PATCHES.md` for the rationale, the exact
diffs, and how to re-apply them after a rebase onto a newer upstream.

| File | Change |
|---|---|
| `DIMA/environments.py` | `+1 line` — add `MATE = "mate"` to the `Env` enum |
| `DIMA/agent/learners/DreamerLearner.py` | `+11 lines` — add an `Env.MATE` branch to `add_experience_to_dataset()` |

`mate/` is **unmodified**.

## History note

The `main` branch of this repository holds an earlier integration attempt that
modified six DIMA files and three MATE files, and placed the adapter *inside*
`DIMA/env/mate/`. This branch (`research-layer`) resets both vendored trees to
pristine upstream and moves all integration code into `research/`.
