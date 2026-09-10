"""Reward and termination heads on a minGPT-style temporal backbone.

Reads a sequence of ``(I_t, a_t^{1:n})`` pairs and predicts, per timestep, the
team mean coverage rate MATE returns for that transition and the done flag.

Each timestep is compressed into one embedding before the causal stack: the
state tokens are mean-pooled, and the camera projections are mean-pooled over
cameras (with a camera-ID embedding kept inside the sum) so that camera order
cannot change the prediction, exactly as in the denoiser.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardTerminationModel(nn.Module):
    """``(I_{t:t+L}, a_{t:t+L}) -> (hat_r, hat_done)`` for every step in the window."""

    def __init__(
        self,
        num_classes,
        num_tokens,
        n_agents,
        action_dim,
        hidden_dim=256,
        n_layers=2,
        n_heads=8,
        max_length=64,
        dropout=0.0,
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.n_agents = n_agents
        self.max_length = max_length

        self.state_embedding = nn.Embedding(num_classes, hidden_dim)
        self.state_pos_embedding = nn.Parameter(torch.zeros(1, num_tokens, hidden_dim))
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.camera_id_embedding = nn.Embedding(n_agents, hidden_dim)
        self.time_pos_embedding = nn.Parameter(torch.zeros(1, max_length, hidden_dim))
        self.input_norm = nn.LayerNorm(hidden_dim)

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

        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )
        self.termination_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )

        nn.init.normal_(self.state_pos_embedding, std=0.02)
        nn.init.normal_(self.time_pos_embedding, std=0.02)

    def forward(self, indices, actions):
        """``indices (B, L, T)``, ``actions (B, L, n, A)`` -> ``(reward (B, L), logits (B, L, 2))``."""

        batch, length = indices.shape[0], indices.shape[1]
        assert length <= self.max_length, f'sequence of {length} exceeds max_length'

        state = (self.state_embedding(indices) + self.state_pos_embedding).mean(dim=2)

        ids = torch.arange(self.n_agents, device=indices.device)
        cameras = (self.action_projection(actions) + self.camera_id_embedding(ids)).mean(dim=2)

        tokens = self.input_norm(state + cameras) + self.time_pos_embedding[:, :length]
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            length, device=indices.device
        )
        embeddings = self.norm(self.transformer(tokens, mask=causal_mask, is_causal=True))

        return self.reward_head(embeddings).squeeze(-1), self.termination_head(embeddings)

    @torch.no_grad()
    def predict(self, indices, actions):
        """Single-step convenience for imagination.

        ``indices (B, T)``, ``actions (B, n, A)`` -> ``(reward (B,), done probability (B,))``.
        """

        reward, term_logits = self(indices.unsqueeze(1), actions.unsqueeze(1))
        return reward[:, 0], F.softmax(term_logits[:, 0], dim=-1)[:, 1]

    def loss(self, indices, actions, rewards, dones):
        """Smooth L1 on the coverage rate, cross-entropy on the done flag.

        Accepts either single transitions (``(B, T)`` / ``(B, n, A)`` / ``(B,)``) or
        windows (``(B, L, ...)``).
        """

        if indices.dim() == 2:
            indices, actions = indices.unsqueeze(1), actions.unsqueeze(1)
            rewards, dones = rewards.unsqueeze(1), dones.unsqueeze(1)

        pred_reward, term_logits = self(indices, actions)
        reward_loss = F.smooth_l1_loss(pred_reward, rewards)
        term_loss = F.cross_entropy(
            term_logits.reshape(-1, 2), dones.reshape(-1).long()
        )

        with torch.no_grad():
            term_accuracy = (
                (term_logits.argmax(dim=-1) == dones.long()).float().mean().item()
            )

        metrics = {
            'rew/smooth_l1': reward_loss.item(),
            'rew/term_ce': term_loss.item(),
            'rew/term_accuracy': term_accuracy,
        }
        return reward_loss + term_loss, metrics
