import time

import numpy as np
import torch

__all__ = ['Meter']


class Meter:
    def __init__(self,
                 reduction='overall',
                 num_classes=20,
                 topk=1,
                 ignore_label=255,
                 **kwargs):
        super(Meter, self).__init__()
        self.reduction = reduction
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.file_dic = {}
        self.reset()

    def reset(self):
        self.total_seen = np.zeros(self.num_classes)
        self.total_correct = np.zeros(self.num_classes)
        self.total_positive = np.zeros(self.num_classes)
        self.total_seen_num = 0
        self.total_correct_num = 0

    def update(self, outputs, targets, pred=False):
        # outputs: 4 x 13 x 2048, targets: 4 x 2048, indices: 4 x 2048
        if not pred:
            predictions = outputs.argmax(1)
        else:
            predictions = outputs

        predictions = predictions[targets != self.ignore_label]
        targets = targets[targets != self.ignore_label]

        if type(outputs) != np.ndarray:
            if self.reduction == 'class' or self.reduction == 'iou':
                for i in range(self.num_classes):
                    self.total_seen[i] += torch.sum(targets == i).item()
                    self.total_correct[i] += torch.sum(
                        (targets == i) & (predictions == targets)).item()
                    self.total_positive[i] += torch.sum(
                        predictions == i).item()

            else:
                self.total_seen_num += targets.numel()
                self.total_correct_num += torch.sum(
                    targets == predictions).item()
        else:
            if self.reduction == 'class' or self.reduction == 'iou':
                for i in range(self.num_classes):
                    self.total_seen[i] += np.sum(targets == i)
                    self.total_correct[i] += np.sum((targets == i)
                                                    & (predictions == targets))
                    self.total_positive[i] += np.sum(predictions == i)

            else:
                self.total_seen_num += targets.size
                self.total_correct_num += np.sum(targets == predictions)

    def compute(self):
        if self.reduction == 'class':
            self.total_correct[self.total_seen == 0] = 1
            self.total_seen[self.total_seen == 0] = 1
            return np.mean(self.total_correct / self.total_seen)
        elif self.reduction == 'iou':
            ious = []
            for i in range(self.num_classes):
                if self.total_seen[i] == 0:
                    ious.append(1)
                else:
                    cur_iou = self.total_correct[i] / (self.total_seen[i] +
                                                       self.total_positive[i] -
                                                       self.total_correct[i])
                    ious.append(cur_iou)
            ious = np.array(ious)
            return np.mean(ious)
        else:
            return self.total_correct_num / self.total_seen_num
