"""Online planning over sampled target trajectories -- no actor, no policy gradient.

At every decision the whole team samples ``K`` candidate command sequences,
rolls its optics forward under them, scores how much of the *generated* target
motion each sequence would cover, and executes the softmax-weighted average.
Nothing here is trained; the trajectory head is the only learned component in
the loop, and the moment its samples are good the cameras point where the
targets are going instead of where they were.

Why a geometric rollout rather than a learned dynamics model for the *camera*: a
camera's dynamics are exactly known and closed-form -- MATE rotates by the
commanded angle, clips the viewing angle, and rescales the sight range to keep
the sector area constant -- and detection is a distance test and an angle test.
Rolling those forward is arithmetic.  The generative model is spent where the
uncertainty actually is, which is the targets.

Everything is torch and batched over cameras and candidates together, so a
decision is a handful of kernel launches whatever the team size:

    candidates   (n_agents, K, H, 2)
    optics       (n_agents, K, H)      orientation, viewing angle, sight range
    coverage     (n_agents, K, H, n_targets)
    rewards      (n_agents, K)

Two approximations, both deliberate:

  Obstacles are ignored while scoring.  MATE lets a camera see a target behind
  an obstacle with the obstacle's transmittance, so occlusion changes the score
  by a probability rather than a hard zero, and the planner would need obstacle
  geometry it does not carry.

  Detection is scored softly.  A hard count is flat almost everywhere -- most
  sampled sequences cover the same integer number of targets -- and a softmax
  over a flat landscape is a coin flip.  Soft margins keep "nearly in view"
  distinguishable from "hopeless".

Coordination is decentralized and one step stale: each camera publishes what its
plan intends to cover on the same peer-to-peer channel that carries the belief,
and discounts targets its neighbours claimed on the previous decision.  Without
it, cameras that share a belief plan the same sweep, since coverage counts
distinct targets and nothing in an independent score says so.
"""

import numpy as np
import torch


def wrap_degrees(angles):
    """Signed angular difference in degrees, wrapped into ``[-180, 180)``."""

    return (angles + 180.0) % 360.0 - 180.0


class CameraKinematics:
    """MATE's camera dynamics for the whole team, read out of their private states.

    A private state is ``[x, y, radius, Rs cos(phi), Rs sin(phi), theta, Rs_max,
    rotation_step, zooming_step]``: the sight vector is stored in cartesian form,
    so range and orientation come back out of it.  The conserved sector area
    ``theta * Rs^2`` gives the minimum viewing angle, which the clip needs and
    which is not otherwise in the state.
    """

    MAX_VIEWING_ANGLE = 180.0

    def __init__(self, camera_states):
        state = torch.as_tensor(camera_states, dtype=torch.float32)
        # Every derived quantity below inherits this tensor's device.

        self.location = state[:, 0:2]                                    # (n, 2)
        sight_vector = state[:, 3:5]
        self.sight_range = sight_vector.norm(dim=-1)                     # (n,)
        self.orientation = torch.rad2deg(
            torch.atan2(sight_vector[:, 1], sight_vector[:, 0])
        )
        self.viewing_angle = state[:, 5]
        self.max_sight_range = state[:, 6]

        self.area_product = self.viewing_angle * self.sight_range ** 2
        self.min_viewing_angle = self.area_product / self.max_sight_range.clamp(min=1e-8) ** 2

    def roll(self, actions):
        """``(n, K, H, 2)`` commands in MATE units -> optics at every step.

        Each returned tensor is ``(n, K, H)``.  The clips are the environment's
        own, not a second policy.
        """

        orientation = self.orientation.view(-1, 1, 1) + actions[..., 0].cumsum(dim=-1)
        # A tensor lower bound and a scalar upper bound cannot be mixed in one
        # clamp call, and the lower bound is per camera.
        viewing_angle = torch.minimum(
            torch.maximum(
                self.viewing_angle.view(-1, 1, 1) + actions[..., 1].cumsum(dim=-1),
                self.min_viewing_angle.view(-1, 1, 1),
            ),
            torch.full_like(self.viewing_angle.view(-1, 1, 1), self.MAX_VIEWING_ANGLE),
        )
        sight_range = torch.sqrt(self.area_product.view(-1, 1, 1) / viewing_angle)
        return wrap_degrees(orientation), viewing_angle, sight_range


class MPPIPlanner:
    """Model-predictive path integral control over the whole camera team."""

    def __init__(
        self,
        horizon=4,
        samples=64,
        temperature=0.05,
        noise_scale=0.6,
        discount=0.95,
        range_softness=60.0,
        angle_softness=6.0,
        intent_discount=0.8,
        seed=0,
        device='cpu',
    ):
        self.horizon = horizon
        self.samples = samples
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.discount = discount
        self.range_softness = range_softness
        self.angle_softness = angle_softness
        self.intent_discount = intent_discount
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        self.mean = None

    def reset(self, n_agents, action_dim):
        """Warm starts, one plan per camera, cleared at an episode boundary."""

        self.mean = torch.zeros(n_agents, self.horizon, action_dim, device=self.device)

    # ------------------------------------------------------------------ scoring

    def _coverage(self, kinematics, orientation, viewing_angle, sight_range, positions):
        """``(n, K, H, n_targets)`` soft detection scores.

        ``positions`` is ``(n, n_targets, H, 2)`` in map units: where the
        trajectory head sampled each target to be at each step of the plan.
        """

        relative = positions - kinematics.location.view(-1, 1, 1, 2)     # (n, T, H, 2)
        distance = relative.norm(dim=-1).transpose(1, 2).unsqueeze(1)    # (n, 1, H, T)
        bearing = torch.rad2deg(
            torch.atan2(relative[..., 1], relative[..., 0])
        ).transpose(1, 2).unsqueeze(1)                                   # (n, 1, H, T)

        in_range = torch.sigmoid(
            (sight_range.unsqueeze(-1) - distance) / self.range_softness
        )
        offset = wrap_degrees(bearing - orientation.unsqueeze(-1)).abs()
        in_view = torch.sigmoid(
            (0.5 * viewing_angle.unsqueeze(-1) - offset) / self.angle_softness
        )
        return in_range * in_view

    def _rewards(self, coverage, believed, peer_intent):
        """Discounted soft coverage of believed targets, minus what peers claimed."""

        weight = believed * (1.0 - self.intent_discount * peer_intent)   # (n, T)
        discount = self.discount ** torch.arange(
            coverage.shape[2], device=coverage.device, dtype=coverage.dtype
        )
        per_step = (coverage * weight.view(weight.shape[0], 1, 1, -1)).sum(dim=-1)
        return (per_step * discount.view(1, 1, -1)).sum(dim=-1)          # (n, K)

    # --------------------------------------------------------------------- plan

    @torch.no_grad()
    def plan(self, camera_states, predicted_positions, believed, peer_intent, action_low, action_high):
        """One decision for the whole team.

        Args:
            camera_states: ``(n, 9)`` raw private camera states.
            predicted_positions: ``(n, n_targets, horizon, 2)`` in map units.
            believed: ``(n, n_targets)`` which slots each camera knows.
            peer_intent: ``(n, n_targets)`` what neighbours claimed last decision.
            action_low, action_high: MATE's camera action box.

        Returns:
            ``(actions in [-1, 1], intent)`` as tensors -- the command to execute
            and what it means to cover, for the next round of the channel.
        """

        tensor = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        camera_states = tensor(camera_states)
        predicted_positions = tensor(predicted_positions)
        believed = tensor(believed)
        peer_intent = tensor(peer_intent)
        low, high = tensor(action_low), tensor(action_high)

        n_agents, action_dim = camera_states.shape[0], low.shape[0]
        if self.mean is None or self.mean.shape[0] != n_agents:
            self.reset(n_agents, action_dim)

        noise = torch.randn(
            (n_agents, self.samples, self.horizon, action_dim),
            device=self.device,
            generator=self.generator,
        ) * self.noise_scale * (high - low) / 2.0
        candidates = torch.clamp(self.mean.unsqueeze(1) + noise, low, high)
        # One candidate is the warm start itself, so a good plan is never lost to
        # sampling.
        candidates[:, 0] = torch.clamp(self.mean, low, high)

        # `camera_states` is already on the device, so the kinematics are built
        # there: no host round trip inside a decision.
        kinematics = CameraKinematics(camera_states)

        orientation, viewing_angle, sight_range = kinematics.roll(candidates)
        coverage = self._coverage(
            kinematics, orientation, viewing_angle, sight_range, predicted_positions
        )
        rewards = self._rewards(coverage, believed, peer_intent)         # (n, K)

        weights = torch.softmax(rewards / max(self.temperature, 1e-8), dim=-1)
        plan = (weights.view(n_agents, -1, 1, 1) * candidates).sum(dim=1)  # (n, H, 2)

        actions = plan[:, 0]
        # Shift the warm start: what was planned for the next step becomes this
        # camera's starting guess at the next decision.
        self.mean = torch.cat(
            [plan[:, 1:], torch.zeros(n_agents, 1, action_dim, device=self.device)], dim=1
        )

        # What the executed plan expects to cover, for the neighbours.
        expected = (weights.view(n_agents, -1, 1, 1) * coverage).sum(dim=1)  # (n, H, T)
        intent = (expected.max(dim=1).values * believed).clamp(0.0, 1.0)

        normalized = 2.0 * (actions - low) / (high - low) - 1.0
        return normalized, intent

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32)
