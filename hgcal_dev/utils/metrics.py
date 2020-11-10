from pytorch_lightning.metrics.metric import Metric
from pytorch_lightning.metrics.functional import iou, average_precision
import torch

class IoU(Metric):
    def __init__(self, num_classes, ignore_index=None, absent_score=0.0, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.absent_score = absent_score


    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape
        for c in range(self.num_classes):
            c_preds = preds[preds == c]
            c_target = target[target == c]
            self.overlap[c] += (c_preds == c_target).sum()
            self.union[c] += c_preds.shape[0] + c_target.shape[0]

    def compute(self):
        return (self.overlap / self.union).mean()