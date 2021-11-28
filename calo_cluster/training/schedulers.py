import functools

import numpy as np
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from ..utils.comm import get_world_size


def cosine_schedule_with_warmup(k, devices, num_training_steps):

    if devices == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // devices

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) / num_training_steps))



def cosine_schedule_with_warmup_factory(devices):
    return lambda optimizer, num_training_steps: LambdaLR(optimizer, functools.partial(cosine_schedule_with_warmup, num_training_steps=num_training_steps, devices=devices))

def one_cycle_lr_factory(max_lr, last_epoch):
    return lambda optimizer, num_training_steps: OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_training_steps, last_epoch=last_epoch)