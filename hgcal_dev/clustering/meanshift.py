from sklearn import cluster


class MeanShift():
    def __init__(self, bandwidth: float = None):
        self.clusterer = cluster.MeanShift(bandwidth=bandwidth)

    def cluster(self, x):
        labels = self.clusterer.fit_predict(x)
        return labels
