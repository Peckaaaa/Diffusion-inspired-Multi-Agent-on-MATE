"""FSQ tokenizer for the MATE world state.

``[s_t, b_t^{1:n}] -> z_t -> I_t`` and back to ``(hat_s_t, hat_o_t^{1:n},
hat_b_t^{1:n})``.  The global decoder feeds the centralized critic and the local
decoder feeds the decentralized actors, both from the *same* discrete code --
that is what keeps imagination consistent between the two.

The encoder is fed the beliefs alongside the state, not the state alone: MATE's
global state describes cameras, targets and obstacles as they are, and carries
no trace of who knows what.  A code built from ``s_t`` alone could not
reconstruct ``b_t^i``, and the actor -- which acts on its belief, not on the
world -- would be handed noise in imagination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fsq_quantizer import FiniteScalarQuantization


def mlp(sizes, activation=nn.SiLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


class StateAutoEncoder(nn.Module):
    """Discrete state tokenizer: ``num_tokens`` FSQ tokens of ``V`` classes each.

    ``num_tokens = 16`` with ``levels = [8, 8, 8]`` carries ``16 * log2(512)``
    bits.  A single token is 9 bits, which cannot describe a 220-dimensional
    MATE state; the budget is a measured choice -- quadrupling it improves
    held-out reconstruction of the target slots by under a fifth.
    """

    def __init__(
        self,
        state_dim,
        obs_dim,
        belief_dim,
        n_agents,
        levels=(8, 8, 8),
        num_tokens=16,
        hidden_dim=512,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.belief_dim = belief_dim
        self.n_agents = n_agents
        self.num_tokens = num_tokens

        self.quantizer = FiniteScalarQuantization(levels)
        self.codebook_size = self.quantizer.codebook_size
        self.code_dim = self.quantizer.codebook_dim
        self.latent_dim = num_tokens * self.code_dim
        self.encoder_input_dim = state_dim + n_agents * belief_dim

        self.encoder = mlp([self.encoder_input_dim, hidden_dim, hidden_dim, self.latent_dim])
        self.global_decoder = mlp([self.latent_dim, hidden_dim, hidden_dim, state_dim])
        self.local_decoder = mlp(
            [self.latent_dim, hidden_dim, hidden_dim, n_agents * (obs_dim + belief_dim)]
        )

    # ------------------------------------------------------------------ encoding

    def encode(self, state, beliefs):
        """``(B, state_dim)`` + ``(B, n, belief_dim)`` -> ``(codes (B, T, d), indices (B, T))``."""

        joint = torch.cat([state, beliefs.reshape(state.shape[0], -1)], dim=-1)
        z = self.encoder(joint).view(-1, self.num_tokens, self.code_dim)
        return self.quantizer(z)

    def encode_indices(self, state, beliefs):
        with torch.no_grad():
            _, indices = self.encode(state, beliefs)
        return indices

    # ------------------------------------------------------------------ decoding

    def decode(self, codes):
        """``(B, T, d)`` codes -> ``(hat_state, hat_obs, hat_belief)``."""

        flat = codes.reshape(codes.shape[0], self.latent_dim)
        hat_state = self.global_decoder(flat)
        local = self.local_decoder(flat).view(-1, self.n_agents, self.obs_dim + self.belief_dim)
        hat_obs, hat_belief = torch.split(local, [self.obs_dim, self.belief_dim], dim=-1)
        return hat_state, hat_obs, hat_belief

    def decode_indices(self, indices):
        """``(B, T)`` integer codes -> ``(hat_state, hat_obs, hat_belief)``.

        This is the imagination path: the diffusion model emits indices, never
        continuous latents.
        """

        codes = self.quantizer.indices_to_codes(indices).to(self.encoder[0].weight.dtype)
        return self.decode(codes)

    # --------------------------------------------------------------------- loss

    def forward(self, state, beliefs):
        codes, indices = self.encode(state, beliefs)
        hat_state, hat_obs, hat_belief = self.decode(codes)
        return hat_state, hat_obs, hat_belief, indices

    def loss(self, state, obs, beliefs):
        """``L_AE = ||hat_s - s||^2 + ||hat_o - o||^2 + ||hat_b - b||^2``."""

        hat_state, hat_obs, hat_belief, indices = self(state, beliefs)
        state_loss = F.mse_loss(hat_state, state)
        obs_loss = F.mse_loss(hat_obs, obs)
        belief_loss = F.mse_loss(hat_belief, beliefs)

        with torch.no_grad():
            usage = indices.unique().numel() / float(self.codebook_size)

        metrics = {
            'ae/state_loss': state_loss.item(),
            'ae/obs_loss': obs_loss.item(),
            'ae/belief_loss': belief_loss.item(),
            'ae/codebook_usage': usage,
        }
        return state_loss + obs_loss + belief_loss, metrics
