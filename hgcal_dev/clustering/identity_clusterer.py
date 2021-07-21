from hgcal_dev.clustering.base_clusterer import BaseClusterer


class IdentityClusterer(BaseClusterer):
    def __init__(self, use_semantic):
        super().__init__(use_semantic, ignore_semantic_labels=None)

    def cluster(self, event):
        return event.embedding
