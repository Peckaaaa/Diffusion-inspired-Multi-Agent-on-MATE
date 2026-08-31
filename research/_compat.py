"""Compatibility shims that let MATE 0.1.0 run on modern gym / numpy.

MATE 0.1.0 (2023-03-31) predates two API removals:

1. ``np.bool8`` was removed in NumPy 2.0.  MATE uses it in ``constants.py``,
   ``environment.py``, ``entities.py``, ``agents/greedy.py``, ``agents/utils.py``
   and ``agents/heuristic.py``.
2. ``gym.utils.seeding.np_random`` returned a ``numpy.random.RandomState`` up to
   gym 0.25; from gym 0.26 it returns a ``numpy.random.Generator``, which has no
   ``randint`` / ``rand`` / ``randn``.  MATE calls ``randint`` in
   ``environment.py``, ``entities.py``, ``wrappers/single_team.py``,
   ``agents/base.py`` and ``agents/mixture.py``.

Both shims are installed **here, in the research layer** rather than by editing
``mate/``, which stays byte-for-byte upstream.  Both are idempotent and both are
no-ops on a dependency set that does not need them, so pinning
``gym==0.25.2`` + ``numpy<2`` makes this module inert without any code change.

``install()`` must run before ``import mate``; ``research/__init__.py`` does that.
"""

from __future__ import annotations

import copy

import numpy as np


_INSTALLED = False


class LegacyRandomProxy:
    """A ``numpy.random.Generator`` carrying the ``RandomState`` aliases MATE calls.

    ``Generator`` is an immutable C type, so the aliases cannot be attached to
    it directly.  This proxy delegates everything except the three legacy names
    to the wrapped generator, so seeding semantics are unchanged.

    It is defined at module scope, and implements ``__deepcopy__`` / ``__reduce__``
    explicitly, because the Oracle world model forks a live MATE environment with
    ``copy.deepcopy`` and MATE environments hold their RNG in this proxy.
    """

    __slots__ = ('_generator',)

    def __init__(self, generator):
        object.__setattr__(self, '_generator', generator)

    @property
    def generator(self):
        """The wrapped ``numpy.random.Generator``."""

        return object.__getattribute__(self, '_generator')

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, '_generator'), name)

    def __deepcopy__(self, memo):
        return LegacyRandomProxy(
            copy.deepcopy(object.__getattribute__(self, '_generator'), memo)
        )

    def __reduce__(self):
        return (LegacyRandomProxy, (object.__getattribute__(self, '_generator'),))

    def __repr__(self):
        return f'LegacyRandomProxy({object.__getattribute__(self, "_generator")!r})'

    def randint(self, low, high=None, size=None, dtype=int):
        return object.__getattribute__(self, '_generator').integers(
            low, high, size=size, dtype=dtype
        )

    def rand(self, *shape):
        return object.__getattribute__(self, '_generator').random(shape or None)

    def randn(self, *shape):
        return object.__getattribute__(self, '_generator').standard_normal(shape or None)


def _install_numpy_bool8() -> bool:
    """Restore the ``np.bool8`` alias removed in NumPy 2.0.

    ``np.bool8`` was always just an alias of ``np.bool_``; restoring it changes
    no behaviour for code that does not use it.
    """

    if hasattr(np, 'bool8'):
        return False

    np.bool8 = np.bool_  # type: ignore[attr-defined]
    return True


def _install_gym_legacy_np_random() -> bool:
    """Give gym >= 0.26's ``Generator`` the ``RandomState`` methods MATE calls.

    ``numpy.random.Generator`` is an immutable C type, so the aliases cannot be
    attached to it.  Instead ``gym.utils.seeding.np_random`` is wrapped so that
    it hands out a proxy that delegates everything except the three legacy
    names.  The proxy draws from the same underlying generator, so seeding
    semantics are unchanged.
    """

    from gym.utils import seeding

    probe, _ = seeding.np_random(0)
    if hasattr(probe, 'randint'):
        return False

    original = seeding.np_random

    def np_random(seed=None):
        generator, used_seed = original(seed)
        return LegacyRandomProxy(generator), used_seed

    np_random.__wrapped__ = original  # type: ignore[attr-defined]
    seeding.np_random = np_random
    return True


def install() -> dict:
    """Install every shim MATE needs on the current dependency set.

    Returns a mapping of shim name -> whether it was actually needed, so the
    smoke test and the run manifest can record which compatibility path a given
    experiment ran on.
    """

    global _INSTALLED  # noqa: PLW0603 - module-level one-shot guard

    if _INSTALLED:
        return _INSTALLED  # type: ignore[return-value]

    applied = {
        'numpy.bool8': _install_numpy_bool8(),
        'gym.seeding.np_random': _install_gym_legacy_np_random(),
    }
    _INSTALLED = applied
    return applied
