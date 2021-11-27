from typing import Any, List, Optional
import torchmetrics

class IoU(torchmetrics.IoU):
    def __init__(self, num_classes: int, ignore_index: Optional[int] = None, absent_score: float = 0, threshold: float = 0.5, reduction: str = "elementwise_mean", compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None) -> None:
        super().__init__(num_classes, ignore_index=None, absent_score=absent_score, threshold=threshold, reduction=reduction, compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group)
    
    def update(self, inputs_list: List[dict], outputs_list: List[dict]):
        """Add data from subbatch dict."""
        for inputs, outputs in zip(inputs_list, outputs_list):
            target = inputs['semantic_labels_raw']
            pred = outputs['pred_semantic_labels']
            if self.ignore_index is not None:
                mask = (pred != self.ignore_index)
                pred = pred[mask]
                target = target[mask]
            super().update(pred, target)

