from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
import functools
from ..utils.comm import get_world_size
import numpy as np

def lr_lambda(k, num_training_steps):

    if get_world_size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // get_world_size()

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) /
                                 num_training_steps))


def lambda_lr_factory():
    last_epoch = -1
    return lambda optimizer, num_training_steps: LambdaLR(optimizer, functools.partial(lr_lambda, num_training_steps=num_training_steps), last_epoch)

def one_cycle_lr_factory(max_lr, last_epoch):
    return lambda optimizer, num_training_steps: OneCycleLR(optimizer, max_lr=max_lr, total_steps=num_training_steps, last_epoch=last_epoch)