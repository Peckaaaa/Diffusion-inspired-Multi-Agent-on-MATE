"""Standardized scene representation handed to planners (brief sections 13, 14).

A planner must not receive DIMA tensors or MATE objects.  It receives a
:class:`SceneView`: plain NumPy arrays with named, physical meaning.  The same
class describes a *real* observation and a *predicted* one, so a planner cannot
tell -- and does not need to know -- whether it is looking at the environment or
at the world model's imagination.

Where the numbers come from
---------------------------
Nothing here re-derives MATE's observation layout by hand.  The slices come from
:func:`mate.constants.camera_observation_slices_of`, and the per-entity field
meanings come from MATE's own ``CameraStatePrivate`` / ``TargetStatePublic``
accessors (``mate/agents/utils.py:189-294``).

CTDE status
-----------
:class:`SceneView` is a **team** view assembled from the camera team's joint
observation.  It contains no privileged state: every entry is something some
camera actually observed this step, and MATE's own camera agents already share
sighted target states with teammates through ``send_responses`` /
``receive_responses`` (``mate/agents/greedy.py:158-227``).  Targets no camera can
see are reported with ``target_sighted == False`` and a zeroed position, exactly
as they appear in the raw observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

import research  # noqa: F401 - installs sys.path + compat shims

from mate import constants as consts


__all__ = ['SceneView', 'ObservationLayout']


class ObservationLayout:
    """Cached slice/index bookkeeping for one environment size."""

    def __init__(self, num_cameras: int, num_targets: int, num_obstacles: int) -> None:
        self.num_cameras = int(num_cameras)
        self.num_targets = int(num_targets)
        self.num_obstacles = int(num_obstacles)
        self.slices: Dict[str, slice] = consts.camera_observation_slices_of(
            self.num_cameras, self.num_targets, self.num_obstacles
        )
        self.obs_dim = int(
            consts.camera_observation_indices_of(
                self.num_cameras, self.num_targets, self.num_obstacles
            )[-1]
        )

    @classmethod
    def from_env_metadata(cls, metadata: Dict) -> 'ObservationLayout':
        return cls(
            metadata['num_cameras'], metadata['num_targets'], metadata['num_obstacles']
        )


@dataclass(frozen=True)
class SceneView:
    """What the camera team can see, in MATE world units.

    Shapes, with ``C`` cameras and ``T`` targets:

    ``camera_positions``      ``(C, 2)``   -- x, y
    ``camera_orientations``   ``(C,)``     -- degrees
    ``camera_viewing_angles`` ``(C,)``     -- degrees
    ``camera_sight_ranges``   ``(C,)``
    ``camera_max_sight_ranges`` ``(C,)``
    ``camera_rotation_steps`` ``(C,)``     -- per-step action bound
    ``camera_zooming_steps``  ``(C,)``     -- per-step action bound
    ``target_positions``      ``(T, 2)``   -- zeros where not sighted
    ``target_sighted``        ``(T,)``     -- bool
    ``target_loaded``         ``(T,)``     -- bool, meaningless where not sighted
    """

    camera_positions: np.ndarray
    camera_orientations: np.ndarray
    camera_viewing_angles: np.ndarray
    camera_sight_ranges: np.ndarray
    camera_max_sight_ranges: np.ndarray
    camera_rotation_steps: np.ndarray
    camera_zooming_steps: np.ndarray
    target_positions: np.ndarray
    target_sighted: np.ndarray
    target_loaded: np.ndarray

    @property
    def num_cameras(self) -> int:
        return int(self.camera_positions.shape[0])

    @property
    def num_targets(self) -> int:
        return int(self.target_positions.shape[0])

    @classmethod
    def from_joint_observation(
        cls, joint_observation: np.ndarray, layout: ObservationLayout
    ) -> 'SceneView':
        """Build a scene view from a ``(C, obs_dim)`` camera joint observation.

        The observation must be in **MATE world units**, not rescaled.
        """

        obs = np.asarray(joint_observation, dtype=np.float64)
        if obs.ndim != 2 or obs.shape[1] != layout.obs_dim:
            raise ValueError(
                f'Expected a joint observation of shape (C, {layout.obs_dim}), got {obs.shape}.'
            )

        self_state = obs[:, layout.slices['self_state']]  # (C, 9), CameraStatePrivate

        camera_positions = self_state[:, 0:2]
        # MATE stores the sight vector, not the angle: see CameraStatePublic.orientation
        # and .sight_range in mate/agents/utils.py:206-222.
        sight_vector = self_state[:, 3:5]
        camera_sight_ranges = np.linalg.norm(sight_vector, axis=-1)
        camera_orientations = np.rad2deg(np.arctan2(sight_vector[:, 1], sight_vector[:, 0]))
        camera_viewing_angles = self_state[:, 5]
        camera_max_sight_ranges = self_state[:, 6]
        camera_rotation_steps = self_state[:, 7]
        camera_zooming_steps = self_state[:, 8]

        opponents = obs[:, layout.slices['opponent_states_with_mask']].reshape(
            layout.num_cameras, layout.num_targets, consts.TARGET_STATE_DIM_PUBLIC + 1
        )
        masks = opponents[..., consts.TARGET_STATE_DIM_PUBLIC] > 0.5  # (C, T)
        sighted = masks.any(axis=0)

        # For each target take the first camera that sees it; MATE zeroes the
        # entries of unsighted entities, so unsighted targets stay at the origin.
        target_positions = np.zeros((layout.num_targets, 2), dtype=np.float64)
        target_loaded = np.zeros(layout.num_targets, dtype=bool)
        for t in range(layout.num_targets):
            seers = np.flatnonzero(masks[:, t])
            if seers.size:
                entry = opponents[seers[0], t]
                target_positions[t] = entry[0:2]
                target_loaded[t] = bool(entry[3] > 0.5)

        return cls(
            camera_positions=camera_positions,
            camera_orientations=camera_orientations,
            camera_viewing_angles=camera_viewing_angles,
            camera_sight_ranges=camera_sight_ranges,
            camera_max_sight_ranges=camera_max_sight_ranges,
            camera_rotation_steps=camera_rotation_steps,
            camera_zooming_steps=camera_zooming_steps,
            target_positions=target_positions,
            target_sighted=sighted,
            target_loaded=target_loaded,
        )

    # ---------------------------------------------------------------- scoring --

    def tracking_matrix(self, tolerance_deg: float = 0.0) -> np.ndarray:
        """``(C, T)`` bool: is target ``t`` inside camera ``c``'s field of view?

        Reproduces MATE's own visibility test geometrically -- a target is
        tracked when it is within the camera's sight range and within half the
        viewing angle of the camera's orientation.  Obstacle occlusion is *not*
        modelled here (MATE resolves it stochastically with
        ``obstacle_transmittance``), so this is an optimistic estimate and is
        only ever used as a planning utility, never reported as coverage.
        """

        delta = self.target_positions[np.newaxis, :, :] - self.camera_positions[:, np.newaxis, :]
        distance = np.linalg.norm(delta, axis=-1)
        bearing = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))
        offset = np.abs(((bearing - self.camera_orientations[:, np.newaxis]) + 180.0) % 360.0 - 180.0)

        within_range = distance <= self.camera_sight_ranges[:, np.newaxis]
        within_angle = offset <= (self.camera_viewing_angles[:, np.newaxis] / 2.0 + tolerance_deg)
        visible = self.target_sighted[np.newaxis, :]
        return within_range & within_angle & visible

    def coverage_estimate(self) -> float:
        """Fraction of *sighted* targets covered by at least one camera."""

        if self.num_targets == 0:
            return 0.0
        return float(self.tracking_matrix().any(axis=0).sum()) / float(self.num_targets)

    def margin_to(self, target_positions: np.ndarray) -> np.ndarray:
        """``(C, K)`` field-of-view margin to *arbitrary* positions, unclipped below.

        Same geometry as :meth:`margin_matrix`, but scoring positions the caller
        supplies instead of the ones this view happens to have observed.  A
        planner needs this: a camera cannot be rewarded for turning *towards* a
        target it cannot currently see, so the utility has to be evaluated
        against remembered positions, exactly as MATE's own ``GreedyCameraAgent``
        aims at ``self.memory`` rather than at what it sees this instant
        (``mate/agents/greedy.py:81-99``).

        Unlike :meth:`margin_matrix` this is **not** clipped at ``-1``.  That clip
        is right for a bounded *score* and wrong for a planning *gradient*: a
        camera whose target is more than one sight-range away saturates at ``-1``
        for every action it could take, its candidate utilities all tie, and it
        stops moving.  Measured on MATE-4v2-9, clipping left two of four cameras
        with an exactly-zero utility spread on 100% of planning steps.  Only the
        upper bound is kept, so "already well inside the view cone" still saturates.
        """

        positions = np.atleast_2d(np.asarray(target_positions, dtype=np.float64))
        if positions.size == 0:
            return np.zeros((self.num_cameras, 0), dtype=np.float64)

        delta = positions[np.newaxis, :, :] - self.camera_positions[:, np.newaxis, :]
        distance = np.linalg.norm(delta, axis=-1)
        bearing = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))
        offset = np.abs(((bearing - self.camera_orientations[:, np.newaxis]) + 180.0) % 360.0 - 180.0)

        half_angle = np.maximum(self.camera_viewing_angles[:, np.newaxis] / 2.0, 1e-6)
        sight = np.maximum(self.camera_sight_ranges[:, np.newaxis], 1e-6)

        angular = 1.0 - offset / half_angle
        radial = 1.0 - distance / sight
        return np.minimum(np.minimum(angular, radial), 1.0)

    def margin_matrix(self) -> np.ndarray:
        """``(C, T)`` float: normalised margin to camera ``c``'s field-of-view boundary.

        ``margin >= 0`` exactly when :meth:`tracking_matrix` is true, and it grows
        as the target moves towards the centre of the view cone.  It is the
        smaller of the angular slack and the range slack, each normalised by its
        own limit, clipped to ``[-1, 1]``::

            angular = 1 - |bearing - orientation| / (viewing_angle / 2)
            radial  = 1 - distance / sight_range
            margin  = clip(min(angular, radial), -1, 1)

        This is *not* MATE's ``soft_coverage_score``
        (``mate/wrappers/auxiliary_camera_rewards.py:196``): that one needs a live
        ``Camera`` entity's ``boundary_between``, which cannot be recovered from an
        observation vector -- and a *predicted* observation is all a world-model
        planner has.  This margin is defined here, and used only as a planning
        utility; coverage that gets reported always comes from MATE.
        """

        delta = self.target_positions[np.newaxis, :, :] - self.camera_positions[:, np.newaxis, :]
        distance = np.linalg.norm(delta, axis=-1)
        bearing = np.rad2deg(np.arctan2(delta[..., 1], delta[..., 0]))
        offset = np.abs(((bearing - self.camera_orientations[:, np.newaxis]) + 180.0) % 360.0 - 180.0)

        half_angle = np.maximum(self.camera_viewing_angles[:, np.newaxis] / 2.0, 1e-6)
        sight = np.maximum(self.camera_sight_ranges[:, np.newaxis], 1e-6)

        angular = 1.0 - offset / half_angle
        radial = 1.0 - distance / sight
        margin = np.clip(np.minimum(angular, radial), -1.0, 1.0)

        # Targets nobody has ever seen carry a zeroed position; scoring them would
        # reward pointing at the origin.
        return np.where(self.target_sighted[np.newaxis, :], margin, -1.0)

    def soft_coverage_estimate(self) -> float:
        """Mean over targets of the best camera margin -- a continuous coverage proxy.

        Ranges in ``[-1, 1]`` and is ``>= 0`` for every covered target, so it
        orders actions even when none of them covers anything yet.  That
        tie-breaking is the whole reason a hard coverage count is not enough as a
        planning utility.
        """

        if self.num_targets == 0:
            return 0.0
        return float(self.margin_matrix().max(axis=0).mean())
