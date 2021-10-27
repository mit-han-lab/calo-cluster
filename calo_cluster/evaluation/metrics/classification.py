from torchmetrics import IoU
from tqdm.auto import tqdm
import torch

def mIoU(evts, num_classes, semantic_label, ignore_index=None, absent_score=1.0, reduction='elementwise_mean'):
    assert ignore_index >= 0 and ignore_index < num_classes
    iou = IoU(num_classes=num_classes, ignore_index=ignore_index, absent_score=absent_score, reduction=reduction)
    for evt in tqdm(evts):
        target = torch.tensor(evt.input_event[semantic_label].to_numpy(), dtype=torch.long)
        preds = torch.tensor(evt.pred_semantic_labels, dtype=torch.long)
        iou(preds, target)
    result = iou.compute()
    print(f'\nmIoU = {result}')
    return result
