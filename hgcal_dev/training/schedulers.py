from torch.optim.lr_scheduler import LambdaLR
import functools
from ..utils.comm import get_world_size
import numpy as np

def lr_lambda(k, num_epochs, batch_size):
    batch_size *= get_world_size()

    if get_world_size() == 1:
        warmup_iters = 0
    else:
        warmup_iters = 1000 // get_world_size()

    if k < warmup_iters:
        return (k + 1) / warmup_iters
    else:
        iter_per_epoch = (9000 + batch_size - 1) // batch_size
        return 0.5 * (1 + np.cos(np.pi * (k - warmup_iters) /
                                 (num_epochs * iter_per_epoch)))


def lambda_lr_factory(num_train_samples, num_epochs, batch_size):
    world_size = get_world_size()
    iter_per_epoch = (num_train_samples + batch_size * world_size
                      - 1) // (batch_size * world_size)
    #TODO: Add last_epoch handling back in when model loading is added.
    #if last_epoch > -1:
    #    last_epoch = (last_epoch + 1) * iter_per_epoch
    #else:
    #    last_epoch = -1
    last_epoch = -1
    _lr_lambda = functools.partial(lr_lambda, num_epochs=num_epochs, batch_size=batch_size)
    return lambda optimizer: LambdaLR(optimizer, _lr_lambda, last_epoch)
