import numpy as np


class PanopticQuality:

    def __init__(self, *, num_classes=1, ignore_index=None, semantic=True, ignore_class_labels=None):
        if not semantic:
            num_classes = 1
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.semantic = semantic
        self.ignore_class_labels = ignore_class_labels
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.num_classes, dtype=np.float64)
        self.fps = np.zeros(self.num_classes, dtype=np.float64)
        self.fns = np.zeros(self.num_classes, dtype=np.float64)
        self.ious = np.zeros(self.num_classes, dtype=np.float64)

        self.wtps = np.zeros(self.num_classes, dtype=np.float64)
        self.wfps = np.zeros(self.num_classes, dtype=np.float64)
        self.wfns = np.zeros(self.num_classes, dtype=np.float64)

    def add(self, outputs, targets, weights=None):
        _xyxinstances, _xyyinstances, _ious, _xmatched, _ymatched, _xareas, _yareas, _xmapping, _ymapping, _intersections = iou_match(
            outputs, targets, weights=weights, threshold=0.5, semantic=self.semantic, ignore_index=self.ignore_index, ignore_class_labels=self.ignore_class_labels, num_classes=self.num_classes)
        for k in range(self.num_classes):
            if k not in _xyxinstances:
                self.tps[k], self.wtps[k], self.ious[k], self.fps[k], self.fns[k], self.wfps[k], self.wfns[k] = - \
                    1, -1, -1, -1, -1, -1, -1
            else:
                self.tps[k] += np.sum(len(_xyxinstances[k]))
                self.wtps[k] += np.sum(_intersections[k])
                self.ious[k] += np.sum(_ious[k])
                # The iou for each (true, pred) cluster pair is weighted according to each hit, s.t. iou ~= 1 implies that high weight hits are present in the intersection.
                # However, each cluster is still valued equally.

                self.fps[k] += np.sum(_xmatched[k] == False)
                self.fns[k] += np.sum(_ymatched[k] == False)

                self.wfps[k] += np.sum(_xareas[k][_xmatched[k] == False])
                self.wfns[k] += np.sum(_yareas[k][_ymatched[k] == False])

    def compute(self):
        m = self.tps != -1

        sq = self.ious / np.maximum(self.tps, 1e-15)
        rq = self.tps / np.maximum(self.tps +
                                   self.fps * 0.5 + self.fns * 0.5, 1e-15)
        pq = (sq[m] * rq[m]).mean()
        tq = self.tps / np.maximum(self.tps + self.fns, 1e-15)
        wrq = self.wtps / np.maximum(self.wtps +
                                     self.wfps * 0.5 + self.wfns * 0.5, 1e-15)
        wpq = (sq[m] * wrq[m]).mean()
        wtq = self.wtps / np.maximum(self.wtps + self.wfns, 1e-15)

        sq[~m], rq[~m], tq[~m], wrq[~m], wtq[~m] = -1, -1, -1, -1, -1

        return {'sq': sq, 'rq': rq, 'pq': pq, 'tq': tq, 'wrq': wrq, 'wpq': wpq, 'wtq': wtq}


def iou_match(outputs, targets, num_classes, weights=None, threshold=0.5, semantic=False, ignore_index=None, ignore_class_labels=None, match_highest=False):
    ''' xi: predicted cluster labels
        yi: true cluster labels
        weights: weight for each sample
        threshold: iou threshold (must be >= 0.5)
        match_highest: if true, match each true cluster to the pred cluster with the highest iou, rather than using a threshold.'''
    assert threshold >= 0.5

    if not semantic:
        xs = np.zeros(outputs.shape[0])
        ys = xs
        xi = outputs
        yi = targets
    else:
        xs, xi = outputs
        ys, yi = targets

    if weights is None:
        weights = np.ones_like(xi, dtype=np.float64)

    mask = (ys != ignore_index)
    xs, xi = xs[mask], xi[mask] + 1
    ys, yi = ys[mask], yi[mask] + 1
    weights = weights[mask]

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

        if ignore_class_labels is not None and k in ignore_class_labels:
            continue

        xik = xi * (xs == k)
        yik = yi * (ys == k)

        xmask = xik != 0
        ymask = yik != 0

        xik = xik - 1
        yik = yik - 1

        xlabels = xik[xmask]
        xweights = weights[xmask]
        xinstances = np.unique(xlabels)
        xmapping = {k: idx for idx, k in enumerate(xinstances)}
        xmatched = np.array([False] * xinstances.shape[0])
        xareas = (np.broadcast_to(xweights, (len(xinstances), len(xweights)))
                  * (xlabels == xinstances[..., None]).astype(int)).sum(axis=1)

        ylabels = yik[ymask]
        yweights = weights[ymask]
        yinstances = np.unique(ylabels)
        ymapping = {k: idx for idx, k in enumerate(yinstances)}
        ymatched = np.array([False] * yinstances.shape[0])
        yareas = (np.broadcast_to(yweights, (len(yinstances), len(yweights)))
                  * (ylabels == yinstances[..., None]).astype(int)).sum(axis=1)

        if len(yinstances) == 0 and len(xinstances) == 0:
            continue

        xymask = np.logical_and(xmask, ymask)
        xyweights = weights[xymask]
        xylabels = xik[xymask] + (2 ** 32) * yik[xymask]
        xyinstances = np.unique(xylabels)
        intersections = (
            xyweights * (xylabels == xyinstances[..., None]).astype(int)).sum(axis=1)
        xyxinstances, xyyinstances = xyinstances % (
            2 ** 32), xyinstances // (2 ** 32)

        xyxmask = xlabels == xyxinstances[..., None]
        xyxareas = (xweights * xyxmask.astype(int)).sum(axis=1)

        xyymask = ylabels == xyyinstances[..., None]
        xyyareas = (yweights * xyymask.astype(int)).sum(axis=1)

        unions = xyxareas + xyyareas - intersections
        ious = intersections / np.maximum(unions, 1e-15)
        ious[intersections == unions] = 1.0
        if match_highest:
            indices = np.zeros(ious.shape[0])
            ious_per_y = ious * \
                (xyyinstances == yinstances[..., None]).astype(int)
            ious_per_y = ious_per_y[ious_per_y.any(axis=1)]
            indices[np.argmax(ious_per_y, axis=1)] = 1
            indices = indices.astype(bool)
        else:
            indices = ious > 0.5

        xmatched[[xmapping[k] for k in xyxinstances[indices]]] = True
        ymatched[[ymapping[k] for k in xyyinstances[indices]]] = True

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
    xs = np.array(xs, dtype=np.int64).reshape(1, -1)
    xi = np.array(xi, dtype=np.int64).reshape(1, -1)
    ys = np.array(ys, dtype=np.int64).reshape(1, -1)
    yi = np.array(yi, dtype=np.int64).reshape(1, -1)
    evaluator = PanopticQuality(num_classes=4, ignore_index=-1)
    evaluator.add((xs, xi), (ys, yi))
    _, _, pq, _, _, _, _ = evaluator.compute()
    print('PQ:', pq.item(), pq.item() == 0.47916666666666663)
