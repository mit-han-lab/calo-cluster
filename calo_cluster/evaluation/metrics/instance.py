from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union
import numpy as np

from torchmetrics import Metric
import torch

class PanopticQuality(Metric):

    def __init__(self, /, num_classes: Union[int, None], semantic: bool, invalid_semantic_label_for_classification: Union[int, None], valid_semantic_labels_for_clustering: List[int], compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None) -> None:
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step, process_group=process_group, dist_sync_fn=dist_sync_fn)
        self.num_classes = num_classes
        self.semantic = semantic
        if not self.semantic:
            self.num_classes = 1
        self.invalid_semantic_label_for_classification = invalid_semantic_label_for_classification
        self.valid_semantic_labels_for_clustering = valid_semantic_labels_for_clustering

        self.add_state('tps', torch.zeros(self.num_classes, dtype=float))
        self.add_state('fps', torch.zeros(self.num_classes, dtype=float))
        self.add_state('fns', torch.zeros(self.num_classes, dtype=float))
        self.add_state('ious', torch.zeros(self.num_classes, dtype=float))

    def update(self, inputs_list, outputs_list):
        for inputs, outputs in zip(inputs_list, outputs_list):
            if self.semantic:
                targets = (inputs['semantic_labels_raw'], inputs['instance_labels_raw'])
                outputs = (outputs['pred_semantic_labels'], outputs['pred_instance_labels'])
            else:
                targets = inputs['instance_labels_raw']
                outputs = outputs['pred_instance_labels']
            _xyxinstances, _xyyinstances, _ious, _xmatched, _ymatched, _xareas, _yareas, _xmapping, _ymapping, _intersections = iou_match(
                outputs, targets, threshold=0.5, semantic=self.semantic, invalid_semantic_label_for_classification=self.invalid_semantic_label_for_classification, valid_semantic_labels_for_clustering=self.valid_semantic_labels_for_clustering, num_classes=self.num_classes)
            for k in range(self.num_classes):
                if k in _xyxinstances:
                    self.tps[k] += _xyxinstances[k].shape[0]
                    self.ious[k] += torch.sum(_ious[k])

                    self.fps[k] += torch.sum(_xmatched[k] == False)
                    self.fns[k] += torch.sum(_ymatched[k] == False)

    def compute(self):
        sq = self.ious / torch.maximum(self.tps, torch.tensor(1e-15))
        rq = self.tps / torch.maximum(self.tps +
                                   self.fps * 0.5 + self.fns * 0.5, torch.tensor(1e-15))
        pq = (sq * rq).mean()
        return pq


def iou_match(outputs, targets, num_classes, threshold=0.5, semantic=False, invalid_semantic_label_for_classification=None, valid_semantic_labels_for_clustering=None, match_highest=False):
    ''' xi: predicted cluster labels
        yi: true cluster labels
        threshold: iou threshold (must be >= 0.5)
        match_highest: if true, match each true cluster to the pred cluster with the highest iou, rather than using a threshold.'''
    assert threshold >= 0.5

    if not semantic:
        xs = torch.zeros(outputs.shape[0])
        ys = xs
        xi = outputs
        yi = targets
    else:
        xs, xi = outputs
        ys, yi = targets


    mask = (ys != invalid_semantic_label_for_classification)
    xs, xi = xs[mask], xi[mask] + 1
    ys, yi = ys[mask], yi[mask] + 1

    _xyxinstances = {}
    _xyyinstances = {}
    _ious = {}
    _xmatched = {}
    _ymatched = {}
    _xareas = {}
    _yareas = {}
    _xmapping = {}
    _ymapping = {}
    _intersections = {}
    for k in range(num_classes):
        if k not in valid_semantic_labels_for_clustering:
            continue

        xik = xi * (xs == k)
        yik = yi * (ys == k)

        xmask = xik != 0
        ymask = yik != 0

        xik = xik - 1
        yik = yik - 1

        xlabels = xik[xmask]
        xinstances = torch.unique(xlabels)
        xmapping = {k: idx for idx, k in enumerate(xinstances.cpu().numpy())}
        xmatched = torch.tensor([False] * xinstances.shape[0], device=xinstances.device)
        xareas = (xlabels == xinstances[..., None]).int().sum(axis=1)

        ylabels = yik[ymask]
        yinstances = torch.unique(ylabels)
        ymapping = {k: idx for idx, k in enumerate(yinstances.cpu().numpy())}
        ymatched = torch.tensor([False] * yinstances.shape[0], device=yinstances.device)
        yareas = (ylabels == yinstances[..., None]).int().sum(axis=1)

        if len(yinstances) == 0 and len(xinstances) == 0:
            continue

        xymask = torch.logical_and(xmask, ymask)
        xylabels = xik[xymask] + (2 ** 32) * yik[xymask]
        xyinstances = torch.unique(xylabels)
        intersections = (xylabels == xyinstances[..., None]).int().sum(axis=1)
        xyxinstances, xyyinstances = xyinstances % (
            2 ** 32), xyinstances // (2 ** 32)

        xyxmask = xlabels == xyxinstances[..., None]
        xyxareas = xyxmask.int().sum(axis=1)

        xyymask = ylabels == xyyinstances[..., None]
        xyyareas = xyymask.int().sum(axis=1)

        unions = xyxareas + xyyareas - intersections
        ious = intersections / torch.maximum(unions, torch.tensor(1e-15))
        ious[intersections == unions] = 1.0
        if match_highest:
            indices = torch.zeros(ious.shape[0])
            ious_per_y = ious * \
                (xyyinstances == yinstances[..., None]).int()
            ious_per_y = ious_per_y[ious_per_y.any(axis=1)]
            indices[torch.argmax(ious_per_y, axis=1)] = 1
            indices = indices.astype(bool)
        else:
            indices = ious > threshold

        xmatched[[xmapping[k] for k in xyxinstances[indices].cpu().numpy()]] = True
        ymatched[[ymapping[k] for k in xyyinstances[indices].cpu().numpy()]] = True

        _xyxinstances[k] = xyxinstances[indices]
        _xyyinstances[k] = xyyinstances[indices]
        _ious[k] = ious[indices]
        _xmatched[k] = xmatched
        _ymatched[k] = ymatched
        _xareas[k] = xareas
        _yareas[k] = yareas
        _xmapping[k] = xmapping
        _ymapping[k] = ymapping
        _intersections[k] = intersections

    return _xyxinstances, _xyyinstances, _ious, _xmatched, _ymatched, _xareas, _yareas, _xmapping, _ymapping, _intersections



if __name__ == '__main__':
    xs, xi = [], []
    ys, yi = [], []
    # some ignore_index stuff
    N_ignore_index = 50
    xs.extend([-1 for i in range(N_ignore_index)])
    xi.extend([0 for i in range(N_ignore_index)])
    ys.extend([-1 for i in range(N_ignore_index)])
    yi.extend([0 for i in range(N_ignore_index)])
    # grass segment
    N_grass = 50
    N_grass_x = 40
    xs.extend([0 for i in range(N_grass_x)])
    xs.extend([1 for i in range(N_grass - N_grass_x)])
    xi.extend([0 for i in range(N_grass)])
    ys.extend([0 for i in range(N_grass)])
    yi.extend([0 for i in range(N_grass)])
    # sky segment
    N_sky = 50
    N_sky_x = 40
    xs.extend([1 for i in range(N_sky_x)])
    xs.extend([0 for i in range(N_sky - N_sky_x)])
    xi.extend([0 for i in range(N_sky)])
    ys.extend([1 for i in range(N_sky)])
    yi.extend([0 for i in range(N_sky)])
    # wrong dog as person xiction
    N_dog = 50
    N_person = N_dog
    xs.extend([2 for i in range(N_person)])
    xi.extend([35 for i in range(N_person)])
    ys.extend([3 for i in range(N_dog)])
    yi.extend([22 for i in range(N_dog)])
    N_person = 50
    xs.extend([2 for i in range(6 * N_person)])
    xi.extend([8 for i in range(4 * N_person)])
    xi.extend([95 for i in range(2 * N_person)])
    ys.extend([2 for i in range(6 * N_person)])
    yi.extend([33 for i in range(3 * N_person)])
    yi.extend([42 for i in range(N_person)])
    yi.extend([11 for i in range(2 * N_person)])
    xs = torch.tensor(xs, dtype=torch.int64).reshape(1, -1)
    xi = torch.tensor(xi, dtype=torch.int64).reshape(1, -1)
    ys = torch.tensor(ys, dtype=torch.int64).reshape(1, -1)
    yi = torch.tensor(yi, dtype=torch.int64).reshape(1, -1)
    evaluator = PanopticQuality(num_classes=4, semantic=True, invalid_semantic_label_for_classification=-1, valid_semantic_labels_for_clustering=[0,1,2,3])
    inputs = {}
    inputs['semantic_labels_raw'] = ys
    inputs['instance_labels_raw'] = yi
    outputs = {}
    outputs['pred_semantic_labels'] = xs
    outputs['pred_instance_labels'] = xi
    evaluator([inputs], [outputs])
    pq = evaluator.compute()
    print('PQ:', pq, torch.isclose(pq, torch.tensor(0.47916666666666663, dtype=float)))
