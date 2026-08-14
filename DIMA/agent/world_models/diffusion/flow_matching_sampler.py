from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from .flow_matching_denoiser import FlowMatchingDenoiser
from .agent_order import get_agent_for_step, sample_agent_order


@dataclass
class FlowMatchingSamplerConfig:
    num_steps_sampling: int = 4
    agent_order: str = ""


class FlowMatchingSampler:
    def __init__(self, denoiser: FlowMatchingDenoiser, cfg: FlowMatchingSamplerConfig) -> None:
        self.denoiser = denoiser.eval()
        self.cfg = cfg

    @torch.no_grad()
    def encode(self, state: Tensor) -> Tensor:
        return self.denoiser.encode(state)

    @torch.no_grad()
    def decode(self, state: Tensor) -> Tensor:
        return self.denoiser.decode(state)

    @torch.no_grad()
    def sample(self, prev_state: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        """
        Integrates the velocity field using forward Euler ODE solver from tau=0 to tau=1.
        """
        device = prev_state.device
        if prev_state.ndim == 4:   # (b, seq_length, num_agents, state_dim)
            prev_state = prev_state.mean(dim=2)

        if not self.denoiser.is_continuous_act and prev_act.ndim == 4:
            prev_act = prev_act.argmax(-1)

        b, t, d = prev_state.size()
        
        # Start at tau = 0 with pure Gaussian noise z0 ~ N(0, I)
        x = torch.randn(b, 1, d, device=device)
        trajectory = [x]

        num_agents = self.denoiser.num_agents
        order = sample_agent_order(num_agents, self.cfg.agent_order)
        num_steps = max(self.cfg.num_steps_sampling, 1)
        d_tau = 1.0 / num_steps

        for step_idx in range(num_steps):
            tau_val = step_idx * d_tau
            tau_tensor = torch.full((b,), tau_val, device=device, dtype=torch.float32)

            # Get active agent for current progress in [0, 1] using shared get_agent_for_step
            active_agent = get_agent_for_step(step_idx, num_steps, order)

            act_mask = torch.ones(*prev_act.shape[:3], device=device, dtype=torch.long)
            act_mask[:, -1] = F.one_hot(torch.as_tensor(active_agent, device=device, dtype=torch.long), num_classes=num_agents).expand(act_mask.size(0), -1)

            # Evaluate target velocity field v_theta(z_tau, tau, s_t, a_t^k)
            v = self.denoiser.compute_velocity(x, tau_tensor, prev_state, prev_act, act_mask)

            # Euler step: z_{tau + d_tau} = z_tau + d_tau * v
            x = x + d_tau * v
            trajectory.append(x)

        return x, trajectory
