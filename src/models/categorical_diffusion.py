"""Discrete categorical diffusion (D3PM) over FSQ state tokens.

Forward process: a Markov chain that corrupts the clean next-state tokens
``I_{t+1}^0`` into ``I_{t+1}^tau``, in cumulative form so any ``tau`` is reachable
in one shot.

  ``absorbing``  ``q(x^tau | x^0) = alpha_bar(tau) delta_{x^0} + (1 - alpha_bar(tau)) delta_[MASK]``
  ``uniform``    ``q(x^tau | x^0) = alpha_bar(tau) delta_{x^0} + (1 - alpha_bar(tau)) Uniform(V)``

Reverse process: the denoiser predicts the clean tokens from the noisy ones, and
sampling walks the schedule down in a handful of steps -- ``sample_steps`` is
independent of ``num_steps``, so a model trained with 20 diffusion steps can be
rolled out in 4.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalDiffusion(nn.Module):
    """Noise schedule, corruption, training objective and sampler for the denoiser."""

    def __init__(
        self,
        denoiser,
        num_classes,
        num_tokens,
        num_steps=20,
        sample_steps=4,
        noise_type='absorbing',
        schedule='cosine',
    ):
        super().__init__()

        assert noise_type in ('absorbing', 'uniform'), noise_type
        assert schedule in ('cosine', 'linear'), schedule

        self.denoiser = denoiser
        self.num_classes = num_classes
        self.num_tokens = num_tokens
        self.num_steps = num_steps
        self.sample_steps = sample_steps
        self.noise_type = noise_type
        self.schedule = schedule
        self.mask_token = num_classes

    # ----------------------------------------------------------------- schedule

    def alpha_bar(self, timesteps):
        """Fraction of tokens still clean at ``tau``.  ``tau = 0`` is clean, ``num_steps`` is pure noise."""

        fraction = timesteps.float() / self.num_steps
        if self.schedule == 'linear':
            return (1.0 - fraction).clamp(0.0, 1.0)
        return torch.cos(fraction * math.pi / 2.0).clamp(0.0, 1.0)

    # ------------------------------------------------------------------ forward

    def q_sample(self, clean_indices, timesteps):
        """Corrupt ``(B, T)`` clean tokens to step ``tau``.  Returns ``(noisy, corrupted_mask)``."""

        keep_prob = self.alpha_bar(timesteps).unsqueeze(-1).expand_as(clean_indices.float())
        corrupt = torch.rand_like(keep_prob) > keep_prob

        if self.noise_type == 'absorbing':
            replacement = torch.full_like(clean_indices, self.mask_token)
        else:
            replacement = torch.randint_like(clean_indices, high=self.num_classes)

        return torch.where(corrupt, replacement, clean_indices), corrupt

    # --------------------------------------------------------------------- loss

    def loss(self, clean_indices, context_indices, actions, emissions):
        """Variational cross-entropy on the reconstructed clean tokens.

        The absorbing chain only ever needs the masked positions predicted -- the
        others are given -- so the objective is restricted to them; the uniform
        chain has no such certainty and is scored everywhere.
        """

        batch = clean_indices.shape[0]
        timesteps = torch.randint(
            1, self.num_steps + 1, (batch,), device=clean_indices.device
        )
        noisy, corrupted = self.q_sample(clean_indices, timesteps)

        logits = self.denoiser(noisy, timesteps, context_indices, actions, emissions)
        per_token = F.cross_entropy(
            logits.reshape(-1, self.num_classes), clean_indices.reshape(-1), reduction='none'
        ).view_as(clean_indices)

        if self.noise_type == 'absorbing':
            weight = corrupted.float()
            loss = (per_token * weight).sum() / weight.sum().clamp(min=1.0)
        else:
            loss = per_token.mean()

        with torch.no_grad():
            predicted = logits.argmax(dim=-1)
            accuracy = (predicted == clean_indices).float()
            corrupted_accuracy = (accuracy * corrupted.float()).sum() / corrupted.float().sum().clamp(min=1.0)

        metrics = {
            'diff/ce_loss': loss.item(),
            'diff/token_accuracy': accuracy.mean().item(),
            'diff/corrupted_accuracy': corrupted_accuracy.item(),
        }
        return loss, metrics

    # ------------------------------------------------------------------ reverse

    @torch.no_grad()
    def sample(self, context_indices, actions, emissions, sample_steps=None, temperature=1.0):
        """Draw ``I_{t+1}^0`` starting from pure noise.  ``(B, T)`` out.

        ``absorbing``: MaskGIT-style confidence unmasking -- each round keeps the
        most confident predictions and re-masks the rest, so ``sample_steps``
        rounds are enough.
        ``uniform``: predict-then-renoise down the schedule.
        """

        steps = sample_steps or self.sample_steps
        batch = context_indices.shape[0]
        shape = (batch, self.num_tokens)
        device = context_indices.device

        def predict(noisy, tau_value):
            timesteps = torch.full((batch,), float(tau_value), device=device)
            logits = self.denoiser(noisy, timesteps, context_indices, actions, emissions)
            if temperature <= 0:
                probabilities = F.one_hot(logits.argmax(-1), self.num_classes).float()
            else:
                probabilities = F.softmax(logits / temperature, dim=-1)
            drawn = torch.multinomial(
                probabilities.reshape(-1, self.num_classes), num_samples=1
            ).view(shape)
            confidence = probabilities.gather(-1, drawn.unsqueeze(-1)).squeeze(-1)
            return drawn, confidence

        if self.noise_type == 'uniform':
            noisy = torch.randint(0, self.num_classes, shape, device=device)
            for i in range(steps):
                tau = self.num_steps * (1.0 - i / steps)
                clean, _ = predict(noisy, tau)
                if i == steps - 1:
                    return clean
                next_tau = torch.full(
                    (batch,), self.num_steps * (1.0 - (i + 1) / steps), device=device
                )
                noisy, _ = self.q_sample(clean, next_tau)
            return clean

        noisy = torch.full(shape, self.mask_token, device=device, dtype=torch.long)
        clean = noisy.clone()
        for i in range(steps):
            tau = self.num_steps * (1.0 - i / steps)
            drawn, confidence = predict(noisy, tau)

            still_masked = noisy == self.mask_token
            clean = torch.where(still_masked, drawn, clean)
            if i == steps - 1:
                return clean

            # Cosine unmasking schedule: keep the most confident tokens, re-mask
            # the rest for the next round.
            keep_ratio = math.cos((i + 1) / steps * math.pi / 2.0)
            num_masked = max(int(round(self.num_tokens * keep_ratio)), 0)
            if num_masked == 0:
                return clean

            scores = torch.where(still_masked, confidence, torch.full_like(confidence, float('inf')))
            remask_index = scores.argsort(dim=-1)[:, :num_masked]
            noisy = clean.clone()
            noisy.scatter_(1, remask_index, self.mask_token)
        return clean
