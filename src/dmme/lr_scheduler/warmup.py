from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, warmup=0.0, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        steps = self.optimizer._step_count + 1

        if steps < self.warmup_steps:
            return [
                group["initial_lr"] * (steps / self.warmup_steps)
                for group in self.optimizer.param_groups
            ]
        else:
            return [group["initial_lr"] for group in self.optimizer.param_groups]
