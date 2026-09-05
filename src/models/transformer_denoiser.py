"""All-camera joint-attention denoiser for the categorical diffusion process.

Predicts the clean next-state tokens from their noised version:

    p_theta(hat_I_{t+1}^0 | I_{t+1}^tau, tau, I_t, a_t^{1:n}, m_t^{emit})

Every camera is one token in a single unmasked sequence, so all cameras are
attended to simultaneously and no camera ordering can bias the prediction -- the
sequential per-agent denoising of the original DIMA is gone.
"""

import math

import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim):
    """Sinusoidal embedding of the diffusion step ``tau``.  ``(B,)`` -> ``(B, dim)``."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(10_000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class JointCameraDenoiser(nn.Module):
    """Transformer over ``[I_{t+1}^tau, I_t, Cam_1 ... Cam_n]``.

    The noisy tokens carry one extra embedding row for the absorbing ``[MASK]``
    state, so the same module serves both the uniform and the absorbing forward
    process.  Logits are read out at the noisy positions only.
    """

    def __init__(
        self,
        num_classes,
        num_tokens,
        n_agents,
        action_dim,
        msg_dim,
        hidden_dim=256,
        n_layers=4,
        n_heads=8,
        dropout=0.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.n_agents = n_agents

        # +1 row: the absorbing [MASK] token, index ``num_classes``.
        self.noisy_embedding = nn.Embedding(num_classes + 1, hidden_dim)
        self.context_embedding = nn.Embedding(num_classes, hidden_dim)
        self.noisy_pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.context_pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))

        self.camera_projection = nn.Linear(action_dim + msg_dim, hidden_dim)
        self.camera_id_embedding = nn.Embedding(n_agents, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self.hidden_dim = hidden_dim
        nn.init.normal_(self.noisy_pos_embedding, std=0.02)
        nn.init.normal_(self.context_pos_embedding, std=0.02)

    def forward(self, noisy_indices, timesteps, context_indices, actions, emissions):
        """-> logits ``(B, num_tokens, num_classes)`` over the clean next-state tokens.

        Args:
            noisy_indices: ``(B, T)`` in ``[0, num_classes]`` (``num_classes`` = ``[MASK]``).
            timesteps: ``(B,)`` diffusion step.
            context_indices: ``(B, T)`` clean tokens of the current state ``I_t``.
            actions: ``(B, n, action_dim)``.
            emissions: ``(B, n, msg_dim)`` messages broadcast at this step.
        """

        noisy = self.noisy_embedding(noisy_indices) + self.noisy_pos_embedding
        context = self.context_embedding(context_indices) + self.context_pos_embedding

        ids = torch.arange(self.n_agents, device=noisy_indices.device)
        cameras = self.camera_projection(
            torch.cat([actions, emissions], dim=-1)
        ) + self.camera_id_embedding(ids)

        time = self.time_mlp(timestep_embedding(timesteps, self.hidden_dim)).unsqueeze(1)

        tokens = torch.cat([noisy, context, cameras], dim=1) + time
        embeddings = self.norm(self.transformer(tokens))
        return self.head(embeddings[:, : self.num_tokens])
