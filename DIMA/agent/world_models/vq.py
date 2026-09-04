import torch.nn as nn
import torch

from .vector_quantize_pytorch import FSQ, VectorQuantize

import torch
import torch.nn as nn

from einops import rearrange
from enum import Enum

import ipdb



class StateDecoderType(str, Enum):
    OPTION1 = "s + id"
    OPTION2 = "s + last_obs"


class _BinaryFlagHead(nn.Module):
    """A separate decoder branch for output channels that are 0/1 flags.

    Why a second branch rather than a second loss on the shared one
    ---------------------------------------------------------------
    The state decoder reconstructs the whole joint observation from a handful of
    quantised tokens.  On MATE-4v8-9 that is 4 x 21 = 84 sighting/obstacle/teammate
    flags sharing a 12-token bottleneck with 420 continuous channels, and the flags
    are positive 13.6% of the time.  Under the shared regression head they were
    driven to the majority class and sighting recall stalled at 0.25-0.31.

    Giving the flags their own two-layer head off the same quantised latent means
    the continuous branch no longer has to spend its output layer on them, and the
    head can be trained with a classification loss on its own scale.  It emits
    *logits*: :meth:`forward` on the autoencoder scatters them into the flag
    channels for ``BCEWithLogitsLoss``, while ``encode_decode`` -- the inference
    path every reader goes through -- scatters ``sigmoid(logits)``, so the 0.5 cut
    downstream code already applies keeps meaning exactly what it did.
    """

    def __init__(self, in_dim: int, num_flags: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, num_flags),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class _BinaryHeadMixin:
    """Shared plumbing for the two autoencoders below."""

    def _init_binary_head(self, latent_dim: int, binary_indices, hidden_size: int = 256) -> None:
        if not binary_indices:
            self.binary_head = None
            self.register_buffer('binary_indices', None, persistent=False)
            return
        indices = torch.as_tensor(list(binary_indices), dtype=torch.long)
        self.register_buffer('binary_indices', indices, persistent=False)
        self.binary_head = _BinaryFlagHead(latent_dim, indices.numel(), hidden_size)

    def _merge_binary(self, z: torch.Tensor, rec: torch.Tensor, as_prob: bool) -> torch.Tensor:
        """Put the flag branch's output into the flag channels of ``rec``."""

        if self.binary_head is None:
            return rec
        logits = self.binary_head(z)
        values = torch.sigmoid(logits) if as_prob else logits
        shape = rec.shape
        flat = rec.reshape(-1, shape[-1]).clone()
        flat[:, self.binary_indices] = values.reshape(-1, self.binary_indices.numel()).to(flat.dtype)
        return flat.reshape(shape)


class SimpleVQAutoEncoder(_BinaryHeadMixin, nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, num_tokens: int, output_dim: int = None, hidden_size: int = 512, binary_indices=None, **vq_kwargs):
        super().__init__()

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * num_tokens, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, output_dim if output_dim is not None else in_dim)
        )

        self.codebook = VectorQuantize(dim=embed_dim, **vq_kwargs)
        self._init_binary_head(embed_dim * num_tokens, binary_indices)
        return
    
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, _ = self.codebook(x)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = self.codebook.get_output_from_indices(indices)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False, binary_as_prob: bool = True):
        z_quantized = self.codebook.get_output_from_indices(indices)
        rec = self.decoder(z_quantized)
        rec = self._merge_binary(z_quantized, rec, binary_as_prob)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices, commit_loss = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)
        # Logits, not probabilities: the flag branch is trained with
        # BCEWithLogitsLoss, which needs them unsquashed.
        rec = self._merge_binary(x, rec, as_prob=False)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices, commit_loss
     
    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        '''
        clamp into [-1, 1]
        '''
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x, alpha = 10.):
        out, indices, cmt_loss = self(x, True, True)
        rec_loss = (out - x).abs().mean()
        loss = rec_loss + alpha * cmt_loss
        
        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "cmt_loss": cmt_loss.item(),
            "rec_loss": rec_loss.item(),
            "active": active_rate,
        }
        
        return loss, loss_dict


class SimpleFSQAutoEncoder(_BinaryHeadMixin, nn.Module):
    def __init__(self, in_dim: int, num_tokens: int, levels, output_dim: int = None, binary_indices=None, **fsq_kwargs) -> None:
        super().__init__()

        self.num_tokens = num_tokens
        self.levels = levels
        self.embed_dim = len(levels)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, len(levels) * num_tokens)
        )

        self.decoder = nn.Sequential(
            nn.Linear(len(levels) * num_tokens, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, output_dim if output_dim is not None else in_dim),
        )

        self.codebook = FSQ(levels, **fsq_kwargs)
        self._init_binary_head(len(levels) * num_tokens, binary_indices)
        
    def encode(self, x, should_preprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        z_quantized = self.codebook.indices_to_codes(indices)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        z_quantized = z_quantized.reshape(*shape[:-1], self.num_tokens, self.embed_dim)
        return z_quantized, indices
        

    def decode(self, indices, should_postprocess: bool = False, binary_as_prob: bool = True):
        shape = indices.shape
        indices = rearrange(indices, "... h -> (...) h")

        z_quantized = self.codebook.indices_to_codes(indices)
        z_quantized = rearrange(z_quantized, "... h d -> (...) (h d)")

        rec = self.decoder(z_quantized)
        rec = self._merge_binary(z_quantized, rec, binary_as_prob)

        rec = rec.reshape(*shape[:-1], -1)

        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
    
    @torch.no_grad()
    def encode_decode(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        z_q, indices = self.encode(x, should_preprocess)
        rec = self.decode(indices, should_postprocess)
        return rec

    def forward(self, x, should_preprocess: bool = False, should_postprocess: bool = False):
        if should_preprocess:
            x = self.preprocess_input(x)

        shape = x.shape
        x = self.encoder(x)

        x = rearrange(x, '... (h d) -> (...) h d', h=self.num_tokens, d=self.embed_dim)
        x, indices = self.codebook(x)
        
        x = x.reshape(*shape[:-1], -1)
        rec = self.decoder(x)
        rec = self._merge_binary(x, rec, as_prob=False)

        indices = indices.reshape(*shape[:-1], self.num_tokens)
        
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec, indices


    def preprocess_input(self, x):
        return x
    
    def postprocess_output(self, y):
        # return y.clamp(-1., 1.)
        return y
    
    def compute_loss(self, x):
        out, indices = self(x, True, True)
        loss = (out - x).abs().mean()

        active_rate = indices.detach().unique().numel() / self.codebook.codebook_size * 100
        
        loss_dict = {
            "rec_loss": loss.item(),
            "active": active_rate,
        }
        
        return loss, loss_dict