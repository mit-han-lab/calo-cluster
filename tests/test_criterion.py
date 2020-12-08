import numpy as np
import pandas as pd
import plotly.express as px
import torch
from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.losses import TripletMarginLoss
from scipy import stats


def generate_1d_data(center, n):
    return np.random.rand(n) - (center + 1/2)


def normal_loss(c, scale):
    criterion = TripletMarginLoss(
        triplets_per_anchor=1, distance=LpDistance(normalize_embeddings=False, p=1))
    l_rv = stats.norm(loc=c, scale=scale)
    r_rv = stats.norm(scale=scale)
    r_rv = l_rv
    l = l_rv.rvs(10).reshape((5, 2))
    r = r_rv.rvs(10).reshape((5, 2))
    df = pd.DataFrame()
    df['x'] = np.concatenate((l[:, 0], r[:, 0]))
    df['y'] = np.concatenate((l[:, 1], r[:, 1]))
    df['label'] = np.concatenate((np.zeros(5), np.ones(5)))
    embeddings = torch.as_tensor(np.concatenate((l, r)))
    labels = torch.as_tensor(df['label'])
    loss = criterion(embeddings, labels)
    print(f'center = {c}, loss = {loss}')


def main():
    for c in np.arange(-1., 1.1, 0.1):
        normal_loss(c, 0.1)


if __name__ == '__main__':
    main()
