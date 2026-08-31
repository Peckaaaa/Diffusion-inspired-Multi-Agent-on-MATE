"""DIMA x MATE research integration layer.

This package is the *third* layer described in the project brief:

    official DIMA  +  official MATE  +  this adapter / research layer

Neither ``DIMA/`` nor ``mate/`` is imported through a modified path, and neither
is copied into this package.  Both stay usable on their own -- see
``UPSTREAM.md`` for the pinned commits and the two accepted upstream edits.

Importing this package
----------------------
``DIMA/`` is not an installable package: its modules import each other by
top-level name (``from agent.learners... import``, ``from episode import``,
``import utils``).  So ``DIMA/`` has to be on ``sys.path`` as a *root*, which is
what ``_bootstrap_paths()`` does below.  ``mate/`` is a normal package root.

Both are inserted at the *end* of ``sys.path`` so they can never shadow a real
installed distribution, except that ``DIMA/`` ships a module named ``utils``
which would collide with any top-level ``utils`` module -- there is none in the
pinned dependency set, and the smoke test asserts that ``utils`` resolves to
``DIMA/utils.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DIMA_ROOT = REPO_ROOT / 'DIMA'
MATE_ROOT = REPO_ROOT / 'mate'


def _drop_namespace_mate() -> None:
    """Evict a half-imported ``mate`` namespace package.

    Run from the repository root, ``import mate`` *before* this package has
    extended ``sys.path`` resolves to the repository's ``mate/`` directory, which
    has no ``__init__.py`` and therefore becomes an empty namespace package --
    ``mate.ASSETS_DIR`` and everything else is missing, with a confusing
    ``AttributeError`` rather than an import failure.  Once ``mate/`` is on
    ``sys.path`` the real package at ``mate/mate/`` wins, so the fix is to forget
    the namespace stub and let it be imported again.
    """

    stub = sys.modules.get('mate')
    if stub is not None and not hasattr(stub, 'ASSETS_DIR'):
        for name in [n for n in sys.modules if n == 'mate' or n.startswith('mate.')]:
            del sys.modules[name]


def _bootstrap_paths() -> None:
    for root in (DIMA_ROOT, MATE_ROOT):
        entry = str(root)
        if entry not in sys.path:
            sys.path.append(entry)
    _drop_namespace_mate()


_bootstrap_paths()

# Must run before `import mate` anywhere in the process.
from research import _compat  # noqa: E402


COMPAT_SHIMS = _compat.install()

# DIMA's train.py disables warnings for the same reason: the pinned gym emits a
# deprecation warning on every `env.seed()` call, once per episode.
if os.environ.get('RESEARCH_KEEP_WARNINGS', '') != '1':
    import warnings

    warnings.filterwarnings('ignore', category=DeprecationWarning, module='gym')
    warnings.filterwarnings('ignore', category=UserWarning, module='gym')

__all__ = ['REPO_ROOT', 'DIMA_ROOT', 'MATE_ROOT', 'COMPAT_SHIMS']
