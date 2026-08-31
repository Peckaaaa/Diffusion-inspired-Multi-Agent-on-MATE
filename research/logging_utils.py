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
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'ckpt').mkdir(exist_ok=True)
        self.name = name
        self._started = time.time()
        self._files: Dict[str, Any] = {}

        import wandb

        self._wandb = wandb
        wandb.init(
            project=project,
            group=group,
            name=name,
            mode=wandb_mode,
            config=dict(config or {}),
            dir=str(self.run_dir),
            reinit=True,
        )

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

    def records(self, stream: str, rows: Iterable[Mapping[str, Any]]) -> None:
        """Append structured rows to ``<run_dir>/<stream>.jsonl``.

        Brief section 25: diagnostics are saved, never dumped to stdout.
        """

        handle = self._files.get(stream)
        if handle is None:
            handle = (self.run_dir / f'{stream}.jsonl').open('a', encoding='utf-8')
            self._files[stream] = handle
        for row in rows:
            handle.write(json.dumps(row, default=_jsonable) + '\n')
        handle.flush()

    def close(self) -> None:
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


def _jsonable(value: Any) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return str(value)
