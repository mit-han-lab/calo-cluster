class BaseClusterer:
    def __init__(self, use_semantic, ignore_semantic_labels):
        self.use_semantic = use_semantic
        self.ignore_semantic_labels = ignore_semantic_labels

    def cluster(self, event):
        raise NotImplementedError()
