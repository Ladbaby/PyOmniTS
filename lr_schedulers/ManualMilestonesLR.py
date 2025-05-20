from torch.optim.lr_scheduler import _LRScheduler

class ManualMilestonesLR(_LRScheduler):
    '''
    Custom lr scheduler class.

    Originally named as 'type2' lradj
    '''
    def __init__(self, optimizer, milestones, last_epoch=-1):
        self.milestones = milestones  # {epoch: lr_value}
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            return [self.milestones[self.last_epoch] for _ in self.base_lrs]
        return [group['lr'] for group in self.optimizer.param_groups]