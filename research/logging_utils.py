"""Logging and provenance (brief sections 20, 21, 31).

Nothing here replaces DIMA's logging.  DIMA already logs to Weights & Biases
(``DIMA/train.py:308``, ``DIMA/utils.py:wandb_log``) and to TensorBoard through the
``LOGGER`` singleton in ``DIMA/tb_logger.py``.  :class:`RunLogger` fans out to
both of those *and* adds the two things the research questions need and DIMA has
no place for:

* a category-tagged console stream (brief section 21), so a failure can be
  attributed to a layer by reading the terminal;
* a per-run directory holding a JSON manifest (brief section 31) and JSONL
  records, so diagnostics survive without dumping tensors to stdout
  (brief section 25).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import research  # noqa: F401 - installs sys.path + compat shims


__all__ = ['CATEGORIES', 'RunLogger', 'git_provenance', 'dependency_versions', 'log']


#: Console categories.  ``FAIL`` is reserved for the smoke test's layer report.
CATEGORIES = (
    'ENV',
    'DATA',
    'WM',
    'WM-DIAG',
    'PLANNER',
    'PLANNER-DIAG',
    'ACTION',
    'EVAL',
    'RUN',
    'WARN',
    'ERROR',
    'FAIL',
    'PASS',
)

_COLOURS = {
    'WARN': '\033[33m',
    'ERROR': '\033[31m',
    'FAIL': '\033[31m',
    'PASS': '\033[32m',
    'WM-DIAG': '\033[36m',
    'PLANNER-DIAG': '\033[36m',
    'EVAL': '\033[35m',
}
_RESET = '\033[0m'


def _use_colour() -> bool:
    return sys.stdout.isatty() and os.environ.get('NO_COLOR') is None


def log(category: str, message: str) -> None:
    """Print one categorised console line."""

    if category not in CATEGORIES:
        raise ValueError(f'Unknown log category {category!r}. Known: {", ".join(CATEGORIES)}')
    tag = f'[{category}]'
    if _use_colour() and category in _COLOURS:
        tag = f'{_COLOURS[category]}{tag}{_RESET}'
    print(f'{tag} {message}', flush=True)


# --------------------------------------------------------------------------- #
# Provenance (brief section 31)
# --------------------------------------------------------------------------- #


def _git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(
            ['git', *args],
            cwd=str(cwd or research.REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _pinned_upstream_commits() -> Dict[str, Optional[str]]:
    """Read the DIMA/MATE commits recorded in ``UPSTREAM.md``.

    ``DIMA/`` and ``mate/`` are vendored, not submodules, so their upstream
    commits live in that file rather than in git metadata.
    """

    commits: Dict[str, Optional[str]] = {'DIMA': None, 'mate': None}
    manifest = research.REPO_ROOT / 'UPSTREAM.md'
    if not manifest.is_file():
        return commits
    for line in manifest.read_text(encoding='utf-8').splitlines():
        for key in commits:
            if line.startswith(f'| `{key}/`') and '`' in line:
                fields = [f.strip().strip('`') for f in line.split('|')]
                for field_value in fields:
                    if len(field_value) == 40 and all(c in '0123456789abcdef' for c in field_value):
                        commits[key] = field_value
    return commits


def git_provenance() -> Dict[str, Any]:
    """Commits and working-tree state, for the run manifest."""

    upstream = _pinned_upstream_commits()
    return {
        'research_commit': _git('rev-parse', 'HEAD'),
        'research_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
        'research_dirty': bool(_git('status', '--porcelain')),
        'dima_upstream_commit': upstream['DIMA'],
        'mate_upstream_commit': upstream['mate'],
    }


def dependency_versions() -> Dict[str, Optional[str]]:
    """Exact versions of everything the pipeline actually imports."""

    import importlib.metadata as md

    names = (
        'torch',
        'torchvision',
        'numpy',
        'gym',
        'einops',
        'termcolor',
        'tqdm',
        'scipy',
        'pyglet',
        'PyYAML',
        'psutil',
        'wandb',
        'tensorboard',
        'ray',
    )
    versions: Dict[str, Optional[str]] = {'python': platform.python_version()}
    for name in names:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    return versions


# --------------------------------------------------------------------------- #
# Run logger
# --------------------------------------------------------------------------- #


WANDB_REF_PREFIX = 'wandb://'
WANDB_RUN_ID_FILENAME = 'wandb_run_id.txt'


def read_wandb_run_id(run_dir: Path | str) -> Optional[str]:
    """The wandb run this directory belongs to, if one was recorded."""

    path = Path(run_dir) / WANDB_RUN_ID_FILENAME
    return path.read_text(encoding='utf-8').strip() or None if path.is_file() else None


def write_wandb_run_id(run_dir: Path | str, run_id: str) -> None:
    (Path(run_dir) / WANDB_RUN_ID_FILENAME).write_text(str(run_id), encoding='utf-8')


def download_checkpoint_artifact(ref: str, dest_dir: Path | str) -> Path:
    """Fetch a checkpoint artifact version from wandb into ``dest_dir``.

    This is the path that matters when the instance that produced the checkpoint
    is gone: its run directory went with it, and wandb holds the only copy.
    Returns the directory the version was downloaded into.
    """

    import wandb

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api().artifact(ref, type='world-model')
    return Path(artifact.download(root=str(dest_dir)))


class RunLogger:
    """One run directory + console + wandb + DIMA's TensorBoard logger.

    Parameters
    ----------
    wandb_mode:
        ``'disabled'`` (default), ``'offline'`` or ``'online'`` -- the same values
        DIMA's ``train.py --mode`` takes.  ``wandb.init`` is always called because
        ``DreamerLearner.step`` calls ``wandb_log`` unconditionally.
    tensorboard:
        Initialise DIMA's ``tb_logger.LOGGER`` under the run directory.
    """

    def __init__(
        self,
        run_dir: Path | str,
        *,
        name: str,
        config: Optional[Mapping[str, Any]] = None,
        wandb_mode: str = 'disabled',
        tensorboard: bool = False,
        project: str = 'dima-mate',
        group: Optional[str] = None,
        resume_run_id: Optional[str] = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'ckpt').mkdir(exist_ok=True)
        self.name = name
        self._started = time.time()
        self._files: Dict[str, Any] = {}
        self._wandb_mode = wandb_mode

        import wandb

        self._wandb = wandb
        # Resuming into the original run keeps one continuous set of curves rather
        # than splitting a preempted job's history across two wandb runs.
        wandb.init(
            project=project,
            group=group,
            name=name,
            mode=wandb_mode,
            config=dict(config or {}),
            dir=str(self.run_dir),
            reinit=True,
            id=resume_run_id,
            resume='must' if resume_run_id is not None else None,
        )

        # Recorded so a restart in this directory can find the run again.
        if wandb.run is not None and getattr(wandb.run, 'id', None):
            write_wandb_run_id(self.run_dir, wandb.run.id)

        from tb_logger import LOGGER

        self._tb = LOGGER
        if tensorboard:
            LOGGER.initialize(log_dir=str(self.run_dir / 'tb'))

        self.manifest: Dict[str, Any] = {
            'name': name,
            'run_dir': str(self.run_dir),
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
            'argv': list(sys.argv),
            'provenance': git_provenance(),
            'dependencies': dependency_versions(),
            'compat_shims': research.COMPAT_SHIMS,
            'config': dict(config or {}),
        }

    @property
    def wandb_run(self):
        """The live wandb run, or None when wandb produced no run."""

        return self._wandb.run

    # -- manifest ----------------------------------------------------------- #

    def update_manifest(self, **entries: Any) -> None:
        self.manifest.update(entries)

    def write_manifest(self) -> Path:
        path = self.run_dir / 'manifest.json'
        payload = dict(self.manifest, duration_seconds=round(time.time() - self._started, 3))
        path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
        return path

    # -- streams ------------------------------------------------------------ #

    def scalars(self, metrics: Mapping[str, float], step: int, prefix: str = '') -> None:
        payload = {f'{prefix}{k}': v for k, v in metrics.items()}
        self._wandb.log({**payload, 'step': step})
        for key, value in payload.items():
            try:
                self._tb.log_scalar(key, float(value), step)
            except (TypeError, ValueError):
                pass

    def tee_dima_scalars(self, stream: str = 'dima_scalars') -> None:
        """Also capture every scalar DIMA logs, without touching DIMA.

        ``DreamerLearner.step`` reports its losses through ``tb_logger.LOGGER``
        (and wandb), and those are the curves that say whether training is going
        anywhere: ``denoiser/train/loss_denoising``,
        ``state_decoder/train/vq/rec_loss``, ``rew_end_model/train/loss_total``
        and friends. ``LOGGER`` is a module-level singleton, so wrapping its
        ``log_scalar`` here tees the whole stream into ``<run_dir>/<stream>.jsonl``
        while leaving TensorBoard behaviour unchanged. Nothing inside DIMA is
        modified or monkey-patched -- the wrapper lives on the object this
        process happens to hold.
        """

        if getattr(self._tb, '_research_tee', None) is not None:
            return

        original = self._tb.log_scalar
        logger = self

        def log_scalar(tag, value, step):
            try:
                logger.records(stream, [{'tag': tag, 'value': float(value), 'step': int(step)}])
            except (TypeError, ValueError):
                pass
            return original(tag, value, step)

        self._tb.log_scalar = log_scalar
        self._tb._research_tee = original

    def stop_tee_dima_scalars(self) -> None:
        original = getattr(self._tb, '_research_tee', None)
        if original is not None:
            self._tb.log_scalar = original
            self._tb._research_tee = None

    def log_checkpoint(self, *paths: Path | str, aliases: Iterable[str] = ()) -> None:
        """Push a checkpoint to wandb as one version of this run's model artifact.

        A rented GPU box is not durable storage: the run directory disappears with
        the instance, so the checkpoints have to leave the machine while the run is
        still going. Every call adds a new version of the single artifact
        ``<run name>-ckpt``, which is why the config sidecar is passed alongside the
        weights -- a checkpoint is only loadable at the horizon and dims recorded
        there, so the two travel together or neither is useful.

        Uploading is skipped entirely when wandb is disabled. ``'offline'`` records
        the versions locally for a later ``wandb sync``.
        """

        if self._wandb_mode == 'disabled':
            return
        artifact = self._wandb.Artifact(f'{self.name}-ckpt', type='world-model')
        for path in paths:
            path = Path(path)
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        self._wandb.log_artifact(artifact, aliases=list(aliases))

    def records(self, stream: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append structured rows to ``<run_dir>/<stream>.jsonl``.

        Brief section 25: diagnostics are saved, never dumped to stdout.
        """

        handle = self._files.get(stream)
        if handle is None:
            handle = (self.run_dir / f'{stream}.jsonl').open('a', encoding='utf-8')
            self._files[stream] = handle
        for row in rows:
            handle.write(json.dumps(_finite(row), default=_jsonable) + '\n')
        handle.flush()

    def close(self) -> None:
        self.stop_tee_dima_scalars()
        for handle in self._files.values():
            handle.close()
        self._files.clear()
        self.write_manifest()
        self._tb.close()
        self._wandb.finish()

    def __enter__(self) -> 'RunLogger':
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def _finite(value: Any) -> Any:
    """Replace NaN / +-Inf with ``null`` so the JSONL stays valid JSON.

    ``json.dumps`` happily writes bare ``NaN`` and ``Infinity``, which Python can
    read back but strict parsers -- pandas, jq, most JS -- cannot. These values
    are meaningful here (``NaN`` = not applicable this step, ``Inf`` = a
    deterministic model's sensitivity ratio), so they are recorded as ``null``
    rather than dropped; the console rendering keeps them as ``N/A``.
    """

    import math

    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    import math

    import numpy as np

    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    return str(value)
