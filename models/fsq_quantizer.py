"""Finite Scalar Quantization (Mentzer et al., https://arxiv.org/abs/2309.15505).

Pure quantization math: bounded rounding with a straight-through estimator, and
the bijection between a rounded code and its integer index.  ``levels = [8, 6, 5]``
gives ``V = 240`` classes per token.
"""

import numpy as np
import torch
import torch.nn as nn


def round_ste(z):
    """Round with a straight-through gradient."""

    return z + (z.round() - z).detach()


class FiniteScalarQuantization(nn.Module):
    """Quantize each channel of a ``len(levels)``-dimensional vector onto its own grid.

    Shapes: ``(..., len(levels))`` in, ``(..., len(levels))`` out, plus an integer
    index in ``[0, prod(levels))`` per position.
    """

    def __init__(self, levels):
        super().__init__()

        self.levels = list(levels)
        self.codebook_dim = len(self.levels)
        self.codebook_size = int(np.prod(self.levels))

        self.register_buffer(
            '_levels', torch.tensor(self.levels, dtype=torch.int64), persistent=False
        )
        basis = torch.cumprod(torch.tensor([1] + self.levels[:-1], dtype=torch.int64), dim=0)
        self.register_buffer('_basis', basis, persistent=False)

    def bound(self, z):
        """Squash ``z`` so that rounding lands inside the grid of each channel."""

        half_l = (self._levels - 1) * (1.0 - 1e-3) / 2.0
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """Continuous latent -> quantized code, normalized to roughly ``[-1, 1]``."""

        return round_ste(self.bound(z)) / (self._levels // 2)

    def codes_to_indices(self, codes):
        """Normalized codes ``(..., d)`` -> integer indices ``(...)``."""

        half_width = self._levels // 2
        digits = (codes * half_width) + half_width
        return (digits.round().to(torch.int64) * self._basis).sum(dim=-1)

    def indices_to_codes(self, indices):
        """Integer indices ``(...)`` -> normalized codes ``(..., d)``."""

        digits = (indices.unsqueeze(-1) // self._basis) % self._levels
        half_width = self._levels // 2
        return (digits - half_width) / half_width

    def forward(self, z):
        codes = self.quantize(z)
        return codes, self.codes_to_indices(codes)
