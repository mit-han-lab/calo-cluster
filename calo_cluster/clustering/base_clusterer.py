class BaseClusterer:
    def __init__(self, use_semantic, valid_semantic_labels):
        self.use_semantic = use_semantic
        self.valid_semantic_labels = valid_semantic_labels

    def cluster(self, embedding, semantic_labels=None):
        raise NotImplementedError()
