class BaseClusterer:
    def __init__(self, use_semantic, ignore_class_label):
        self.use_semantic = use_semantic
        self.ignore_class_label = ignore_class_label

    def cluster(self, event):
        raise NotImplementedError()
