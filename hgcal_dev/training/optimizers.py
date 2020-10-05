from torch.optim import SGD

def sgd_factory(lr: float, weight_decay: float, momentum: float, nesterov: bool):
    return lambda params: SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)