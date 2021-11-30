from typing import Any, List, Optional
import torchmetrics

class IoU(torchmetrics.IoU):
    def update(self, inputs: dict, outputs: dict):
        """Add data from subbatch dict."""
        target = inputs['semantic_labels_raw']
        pred = outputs['pred_semantic_labels']
        if self.ignore_index is not None:
            mask = (target != self.ignore_index)
            pred = pred[mask]
            target = target[mask]
        super().update(pred, target)
            

