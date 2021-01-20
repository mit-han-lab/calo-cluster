  
from sklearn import cluster
from hgcal_dev.clustering.mean_shift_cosine_gpu import MeanShiftCosine

class MeanShift():
    def __init__(self, bandwidth: float = None, use_gpu=True):
        if use_gpu:
            self.clusterer = MeanShiftCosine(bandwidth=bandwidth)
        else:
            self.clusterer = cluster.MeanShift(bandwidth=bandwidth)

    def cluster(self, x):
        self.clusterer.fit(x)
        labels = self.clusterer.labels_
        return labels