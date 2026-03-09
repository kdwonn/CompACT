import torch
import torch.nn as nn
from typing import Callable, Optional


class RandomConditioning(nn.Module):
    """Random conditioning with potentially learnt mean and stddev."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        learn_mean: bool = True,
        learn_std: bool = True,
        mean_init: Callable[[torch.Tensor], None] = torch.nn.init.xavier_uniform_,
        logsigma_init: Callable[[torch.Tensor], None] = nn.init.xavier_uniform_,
    ):
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        if learn_mean:
            self.slots_mu = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_mu", torch.zeros(1, 1, object_dim))

        if learn_std:
            self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, object_dim))
        else:
            self.register_buffer("slots_logsigma", torch.zeros(1, 1, object_dim))

        with torch.no_grad():
            mean_init(self.slots_mu)
            logsigma_init(self.slots_logsigma)

    def forward(self, batch_size: int, num_slots=None):
        if num_slots is None:
            num_slots = self.n_slots

        mu = self.slots_mu.expand(batch_size, num_slots, -1)
        sigma = self.slots_logsigma.exp().expand(batch_size, num_slots, -1)
        return mu + sigma * torch.randn_like(mu)


class LearntConditioning(nn.Module):
    """Conditioning with a learnt set of slot initializations, similar to DETR."""

    def __init__(
        self,
        object_dim: int,
        n_slots: int,
        slot_init: Optional[Callable[[torch.Tensor], None]] = None,
    ):
        """Initialize LearntConditioning.

        Args:
            object_dim: Dimensionality of the conditioning vector to generate.
            n_slots: Number of conditioning vectors to generate.
            slot_init: Callable used to initialize individual slots.
        """
        super().__init__()
        self.n_slots = n_slots
        self.object_dim = object_dim

        self.slots = nn.Parameter(torch.zeros(1, n_slots, object_dim))

        if slot_init is None:
            slot_init = nn.init.normal_

        with torch.no_grad():
            slot_init(self.slots)

    def forward(self, batch_size: int):
        """Generate conditioining vectors for `batch_size` instances.

        Args:
            batch_size: Number of instances to create conditioning vectors for.

        Returns:
            The conditioning vectors.
        """
        return self.slots.expand(batch_size, -1, -1)
