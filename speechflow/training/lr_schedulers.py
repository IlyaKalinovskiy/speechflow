import math

from torch import optim
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["ConstLR", "WarmupInvRsqrtLR", "WarmupCosine"]


class ConstLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_max):
        self._lr_max = lr_max
        super().__init__(optimizer)

    def get_lr(self):
        return [self._lr_max for _ in self.optimizer.param_groups]


class WarmupInvRsqrtLR(optim.lr_scheduler._LRScheduler):
    """Increases learning rate linearly for `warmup` steps, then decays it at inverse sqrt
    rate."""

    def __init__(self, optimizer, lr_max, step_factor=1):
        self._lr_max = lr_max
        self._step_factor = step_factor
        super().__init__(optimizer)

    def current_rate(self):
        step = self._step_count * self._step_factor
        return min(1e-6 * step, 1 / math.sqrt(step)) * self._lr_max * 100.0

    def get_lr(self):
        rate = self.current_rate()
        return [rate for _ in self.optimizer.param_groups]


class WarmupCosine(LambdaLR):
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        super().__init__(optimizer, lr_lambda, last_epoch)
