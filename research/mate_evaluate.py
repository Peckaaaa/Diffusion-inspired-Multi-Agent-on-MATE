"""``python -m mate.evaluate`` made runnable on the pinned dependency set.

MATE's official evaluation script is reused as-is -- for the reactive baselines,
and as the cross-check that this project's adapter reproduces MATE's own numbers.
Two things stop it launching directly, both of them gym-version problems rather
than anything to do with this project:

1. MATE 0.1.0 calls ``np_random.randint``; gym >= 0.26 hands out a
   ``numpy.random.Generator``.  Fixed by ``research/_compat.py``.
2. ``mate.evaluate`` builds the environment with ``mate.make`` (which is
   ``gym.make``).  From gym 0.26 that wraps the environment in
   ``PassiveEnvChecker``, which rejects MATE's four-value ``step()`` -- MATE
   returns ``info`` as a *list* of per-agent dicts, and its ``reset()`` returns
   ``(camera_obs, target_obs)`` rather than ``(obs, info)``.

``mate.make`` is therefore redirected to ``mate.make_environment``, MATE's own
constructor, which takes the same ``config=`` / ``wrappers=`` keywords and skips
gym's wrapper stack.  This is exactly what ``research/env_adapter.py`` does, so
the two paths build the same environment and their results are directly
comparable.  No argument is intercepted and no default is altered.

Usage is identical to the upstream script::

    python -m research.mate_evaluate \\
        --config mate/mate/assets/MATE-4v2-9.yaml \\
        --camera-agent mate:GreedyCameraAgent \\
        --seed 0 --episodes 10 --no-render

Known upstream limitation: ``--camera-discrete-levels`` cannot be combined with
MATE's rule-based camera agents.  Those agents build their action space from
``CameraStatePrivate.action_space`` and always emit continuous ``(rotation, zoom)``
pairs, which ``DiscreteCamera.action`` then rejects.  This project's adapter
handles the same situation by projecting with ``DiscreteCamera.reverse_action``
(see ``MATEEnv.action_from_continuous``); ``mate.evaluate`` has no such step, so
run it in continuous mode.
"""

from __future__ import annotations

import sys

import research  # noqa: F401 - installs sys.path + compat shims


def main() -> None:
    import mate

    def make(env_id, **kwargs):  # noqa: ARG001 - the id is always MultiAgentTracking-v0
        return mate.make_environment(**kwargs)

    mate.make = make

    from mate.evaluate import main as mate_main

    sys.argv[0] = 'python -m mate.evaluate'
    mate_main()


if __name__ == '__main__':
    main()
