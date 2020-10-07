from sklearn import cluster
from torch import nn


class MeanShift(nn.Module):
    def __init__(self, bandwidth: float = None):
        self.clusterer = cluster.MeanShift(bandwidth=bandwidth)

    def forward(self, x):
        labels = self.clusterer.fit_predict(x)
        return labels
