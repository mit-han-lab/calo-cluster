  
from sklearn import cluster
from hgcal_dev.clustering.mean_shift_cosine_gpu import MeanShiftCosine
import numpy as np

class MeanShift():
    def __init__(self, *, use_gpu=True, use_semantic=False, ignore_label=None, **kwargs):
        self.use_semantic = use_semantic
        self.ignore_label = ignore_label
        if use_gpu:
            self.clusterer = MeanShiftCosine(**kwargs)
        else:
            self.clusterer = cluster.MeanShift(**kwargs)

    def cluster(self, event):
        """Clusters hits in event. If self.use_semantic, clusters only within each predicted semantic subset. 
           If self.ignore_label, ignores hits with the given semantic label."""
        if self.use_semantic:
            pred_cluster_labels = np.full_like(event.pred_class_labels, fill_value=-1)
            unique_labels = np.unique(event.pred_class_labels)
            for l in unique_labels:
                if l == self.ignore_label:
                    continue
                mask = (event.pred_class_labels == l)
                self.clusterer.fit(event.embedding[mask])
                pred_cluster_labels[mask] = self.clusterer.labels_
        else:
            self.clusterer.fit(event.embedding)
            pred_cluster_labels = self.clusterer.labels_
        return pred_cluster_labels