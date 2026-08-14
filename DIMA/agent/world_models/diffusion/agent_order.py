import torch
from typing import Union, List

def get_agent_for_step(step_idx: int, total_steps: int, agent_order: Union[torch.Tensor, List[int]]) -> int:
    """
    Computes the agent index for a given sampling step based strictly on step progress in [0, 1].
    This is independent of time direction (whether tau increases 0 -> 1 or sigma decreases sigma_max -> sigma_min).
    """
    progress = step_idx / max(total_steps, 1)
    if isinstance(agent_order, list):
        num_agents = len(agent_order)
    else:
        num_agents = agent_order.numel()
        
    agent_pos = int(progress * num_agents)
    agent_pos = min(agent_pos, num_agents - 1)
    
    return agent_order[agent_pos]

def sample_agent_order(num_agents: int, order: str = "default") -> torch.Tensor:
    """
    Generates the ordered tensor of agent indices for sampling.
    """
    if order == 'default' or order == "" or order is None:
        agent_order = torch.flip(torch.arange(num_agents), [0])
    elif order == 'reverse':
        agent_order = torch.arange(num_agents)
    elif order == 'random':
        agent_order = torch.randperm(num_agents)
    else:
        raise NotImplementedError(f"Unsupported agent order: {order}")
        
    return agent_order
