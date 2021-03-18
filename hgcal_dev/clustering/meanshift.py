  
from sklearn import cluster
from hgcal_dev.clustering.mean_shift_cosine_gpu import MeanShiftCosine

class MeanShift():
    def __init__(self, *, use_gpu=True, **kwargs):
        if use_gpu:
            self.clusterer = MeanShiftCosine(**kwargs)
        else:
            self.clusterer = cluster.MeanShift(**kwargs)

    def cluster(self, x):
        self.clusterer.fit(x)
        labels = self.clusterer.labels_
        return labels