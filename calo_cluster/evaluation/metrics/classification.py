from dataclasses import dataclass
import torch
from torchmetrics import IoU
from tqdm.auto import tqdm

from calo_cluster.evaluation.metrics.metric import Metric

@dataclass
class IoUMetric(Metric):
    num_classes: int
    ignore_index: int
    absent_score: float
    reduction: str


    def __post_init__(self):
        self.iou = IoU(self.num_classes, self.ignore_index, self.absent_score, self.reduction).to(torch.device("cuda", 0))

    def add(self, pred_labels, labels):
        """Add data."""
        self.iou(pred_labels, labels)

    def add_from_dict(self, subbatch):
        """Add data from subbatch dict."""
        self.add(pred_labels=subbatch['pred_semantic_labels'], labels=subbatch['semantic_labels_mapped'])

    def _save(self, path):
        print(f'mIoU = {self.iou.compute()}')
