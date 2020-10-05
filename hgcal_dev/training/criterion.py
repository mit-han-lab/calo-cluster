from torch import nn
from torch.nn import CrossEntropyLoss
from pytorch_metric_learning.losses import TripletMarginLoss

class JointLoss(nn.Module):
    def __init__(self, ignore_index: int = 255, alpha: float = 1.0):
        self.alpha = alpha
        self.class_loss = CrossEntropyLoss(ignore_index=ignore_index)
        self.instance_loss = TripletMarginLoss()

    def forward(self, outputs, targets):
        embeddings, class_out = outputs
        instance_labels, class_labels = targets
        return self.alpha * self.instance_loss(embeddings, instance_labels) + self.class_loss(class_out, class_labels)


























































































    
