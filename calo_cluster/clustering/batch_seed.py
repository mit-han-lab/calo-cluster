# Author Mengyang Zhao <Mengyang.Zhao@tufts.edu>

import math

import numpy as np
import torch
from torch import exp


def cos_batch(a, b):
    num = a@b.T
    denom = torch.norm(a, dim=1).reshape(-1, 1) * torch.norm(b, dim=1)
    return num / denom


def get_weight(sim, bandwidth):

    thr = 1-bandwidth
    max = torch.tensor(1.0).double().cuda()
    min = torch.tensor(0.0).double().cuda()
    dis = torch.where(sim > thr, max, min)

    return dis


def gaussian(dist, bandwidth):
    return exp(-0.5 * ((dist / bandwidth))**2) / (bandwidth * math.sqrt(2 * math.pi))


def meanshift_torch(data, seed, bandwidth, max_iter=300):

    stop_thresh = 1e-3 * bandwidth
    iter = 0

    X = torch.from_numpy(np.copy(data)).double().cuda()
    S = torch.from_numpy(np.copy(seed)).double().cuda()
    B = torch.tensor(bandwidth).double().cuda()

    while True:
        weight = get_weight(cos_batch(S, X), B)

        num = (weight[:, :, None] * X).sum(dim=1)
        S_old = S
        S = num / weight.sum(1)[:, None]
        iter += 1

        if (torch.norm(S - S_old, dim=1).mean() < stop_thresh or iter == max_iter):
            break

    p_num = []
    for line in weight:
        p_num.append(line[line == 1].size()[0])

    my_mean = S.cpu().numpy()

    return my_mean, p_num
