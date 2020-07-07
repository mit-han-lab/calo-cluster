import functools

import numpy as np
import torch.optim as optim

from models.hgcal import SPVCNN
from utils.comm import *
from utils.config import Config, configs


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


configs.model = Config(SPVCNN)
configs.model.num_classes = configs.data.num_classes

configs.train.batch_size = 2
configs.train.num_epochs = 15

configs.train.optimizer = Config(optim.SGD)
configs.train.optimizer.lr = 0.24
configs.train.optimizer.weight_decay = 1e-4
configs.train.optimizer.momentum = 0.9
configs.train.optimizer.nesterov = True

configs.train.scheduler = Config(optim.lr_scheduler.LambdaLR)
configs.train.scheduler.lr_lambda = functools.partial(
    lr_lambda,
    num_epochs=configs.train.num_epochs,
    batch_size=configs.train.batch_size)
