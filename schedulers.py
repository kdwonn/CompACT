import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithPlateau(_LRScheduler):
    """Cosine annealing scheduler that plateaus at lr_min after T_max iterations.

    The learning rate follows linear warmup for warmup_steps iterations,
    then cosine annealing from initial lr to lr_min over T_max iterations,
    then stays constant at lr_min for all subsequent iterations.

    Args:
        optimizer: Wrapped optimizer.
        T_max: Maximum number of iterations for cosine annealing.
        eta_min: Minimum learning rate (plateau value).
        warmup_steps: Number of warmup steps. Default: 0.
        last_epoch: The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, eta_min=0, warmup_steps=0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        elif self.last_epoch < self.T_max:
            # Cosine annealing phase
            cosine_steps = self.T_max - self.warmup_steps
            cosine_epoch = self.last_epoch - self.warmup_steps
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * cosine_epoch / cosine_steps))
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            # Plateau phase - maintain lr_min
            return [self.eta_min for _ in self.base_lrs]
