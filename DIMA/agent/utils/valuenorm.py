"""ValueNorm."""
import numpy as np
import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """Normalize a vector of observations - across the first norm_axes dimensions"""

    def __init__(
        self,
        input_shape,
        norm_axes=1,
        beta=0.99999,
        per_element_update=False,
        epsilon=1e-5,
        device=torch.device("cpu"),
    ):
        super(ValueNorm, self).__init__()

        self.input_shape = input_shape
        self.norm_axes = norm_axes
        self.epsilon = epsilon
        self.beta = beta
        self.per_element_update = per_element_update
        self.tpdv = dict(dtype=torch.float32, device=device)

        # register_buffer, KHONG phai `nn.Parameter(...).to(device)`. Calling .to() on a
        # Parameter with a *different* device returns a plain Tensor, so the old code only
        # registered these on CPU: on GPU they became untracked attributes and state_dict()
        # came back empty. That is why GPU-trained checkpoints carry `value_normalizer: {}`
        # and why the value scale reset at every session boundary. Buffers are always
        # registered, always in state_dict(), and still work with the in-place mul_/add_
        # updates below (requires_grad=False is implicit for buffers).
        self.register_buffer("running_mean", torch.zeros(input_shape, **self.tpdv))
        self.register_buffer("running_mean_sq", torch.zeros(input_shape, **self.tpdv))
        self.register_buffer("debiasing_term", torch.tensor(0.0, **self.tpdv))

    def running_mean_var(self):
        """Get running mean and variance."""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var
    
    def running_mean_var_cpu(self):
        """Get running mean and variance."""
        debiased_mean = self.running_mean / self.debiasing_term.clamp(min=self.epsilon)
        debiased_mean_sq = self.running_mean_sq / self.debiasing_term.clamp(
            min=self.epsilon
        )
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean.cpu(), debiased_var.cpu()

    @torch.no_grad()
    def update(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        batch_mean = input_vector.mean(dim=tuple(range(self.norm_axes)))
        batch_sq_mean = (input_vector**2).mean(dim=tuple(range(self.norm_axes)))

        if self.per_element_update:
            batch_size = np.prod(input_vector.size()[: self.norm_axes])
            weight = self.beta**batch_size
        else:
            weight = self.beta

        self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
        self.running_mean_sq.mul_(weight).add_(batch_sq_mean * (1.0 - weight))
        self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))

    def normalize(self, input_vector):
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (input_vector - mean[(None,) * self.norm_axes]) / torch.sqrt(var)[
            (None,) * self.norm_axes
        ]

        return out

    def denormalize(self, input_vector):
        """Transform normalized data back into original distribution"""
        if isinstance(input_vector, np.ndarray):
            input_vector = torch.from_numpy(input_vector)
        input_vector = input_vector.to(**self.tpdv)

        mean, var = self.running_mean_var()
        out = (
            input_vector * torch.sqrt(var)[(None,) * self.norm_axes]
            + mean[(None,) * self.norm_axes]
        )

        # out = out.cpu().numpy()

        return out
