# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 02:28:29 2019

@author: Austin Hsu
"""

from torch.optim.lr_scheduler import _LRScheduler

class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, delay_epochs, after_scheduler):
        self.delay_epochs = delay_epochs
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.delay_epochs:
            if not self.finished:
                self.after_scheduler.base_lrs = self.base_lrs
                self.finished = True
            return self.after_scheduler.get_lr()
        return [base_lrs * (self.last_epoch / self.delay_epochs) for base_lrs in self.base_lrs]

    def step(self, epoch=None):
        if self.finished:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.delay_epochs)
        else:
            return super(WarmupScheduler, self).step(epoch)

def WarmupLR(optimizer, delay_epochs, base_scheduler):
    return WarmupScheduler(optimizer, delay_epochs, base_scheduler)