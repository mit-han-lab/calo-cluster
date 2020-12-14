from torch.optim import SGD, AdamW

def sgd_factory(lr: float, weight_decay: float, momentum: float, nesterov: bool):
    return lambda params: SGD(params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

def adam_factory(lr: float, weight_decay: float):
    return lambda params: AdamW(params, lr=lr, weight_decay=weight_decay)