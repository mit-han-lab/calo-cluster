from pytorch_lightning.metrics import IoU, ConfusionMatrix, MulticlassPrecisionRecallCurve
from pytorch_lightning.metrics.metric import Metric
from typing import List

def create_metrics(names: str, num_classes: int) -> List[Metric]:
    metrics_list = []
    if 'iou' in names:
        metrics_list.append(IoU())
    if 'confusion_matrix' in names:
        metrics_list.append(ConfusionMatrix(normalize=True))
    if 'precision_recall' in names:
        metrics_list.append(MulticlassPrecisionRecallCurve(num_classes=num_classes))
    return metrics_list

