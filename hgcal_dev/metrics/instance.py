import numpy as np
from sklearn.metrics import jaccard_score


def match_instances(labels, pred_labels, return_maps=True):
    unique_labels = np.unique(labels)
    unique_pred_labels = np.unique(pred_labels)
    ious = np.zeros((unique_labels.shape[0], unique_pred_labels.shape[0]))
    for i, truth_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_pred_labels):
            _labels = labels.copy()
            _labels[labels == truth_label] = 1
            _labels[labels != truth_label] = 0
            _pred_labels = labels.copy()
            _pred_labels[pred_labels == pred_label] = 1
            _pred_labels[pred_labels != pred_label] = 0
            ious[i, j] = jaccard_score(_labels, _pred_labels, average='binary')
    if return_maps:
        truth_to_pred = {k: v for k, v in zip(
            unique_labels, ious.argmax(axis=1))}
        pred_to_truth = {v: k for k, v in truth_to_pred.items()}
        return truth_to_pred, pred_to_truth, ious.max(axis=1)
    else:
        return ious.max(axis=1)


def mIoU(labels, pred_labels):
    ious = match_instances(labels, pred_labels, return_maps=False)
    return ious.mean()


class PanopticQuality:

    def __init__(self, *, num_classes, ignore_index=None, semantic=True):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.semantic = semantic
        self.reset()

    def reset(self):
        self.tps = np.zeros(self.num_classes, dtype=np.float64)
        self.fps = np.zeros(self.num_classes, dtype=np.float64)
        self.fns = np.zeros(self.num_classes, dtype=np.float64)
        self.ious = np.zeros(self.num_classes, dtype=np.float64)
        self.wious = np.zeros(self.num_classes, dtype=np.float64)

        self.wtps = np.zeros(self.num_classes, dtype=np.float64)
        self.wfps = np.zeros(self.num_classes, dtype=np.float64)
        self.wfns = np.zeros(self.num_classes, dtype=np.float64)

    def add(self, outputs, targets, weights=None):
        if not self.semantic:
            xs = np.zeros(outputs.shape[0])
            ys = xs
            xi = outputs
            yi = targets
        else:
            xs, xi = outputs
            ys, yi = targets

        if weights is None:
            weights = np.ones_like(xi, dtype=np.float64)

        mask = (ys != self.ignore_index)
        xs, xi = xs[mask], xi[mask] + 1
        ys, yi = ys[mask], yi[mask] + 1
        weights = weights[mask]

        for k in range(self.num_classes):
            xik = xi * (xs == k)
            yik = yi * (ys == k)

            xmask = xik != 0
            ymask = yik != 0

            xlabels = xik[xmask]
            xweights = weights[xmask]
            xinstances = np.unique(xlabels)
            xmapping = {k: idx for idx, k in enumerate(xinstances)}
            xmatched = np.array([False] * xinstances.shape[0])
            xareas = (np.broadcast_to(xweights, (len(xinstances), len(xweights)))
                      * xlabels == xinstances[..., None]).sum(axis=1)

            ylabels = yik[ymask]
            yweights = weights[ymask]
            yinstances = np.unique(ylabels)
            ymapping = {k: idx for idx, k in enumerate(yinstances)}
            ymatched = np.array([False] * yinstances.shape[0])
            yareas = (np.broadcast_to(yweights, (len(yinstances), len(yweights)))
                      * ylabels == yinstances[..., None]).sum(axis=1)

            if len(yinstances) == 0 and len(xinstances) == 0:
                self.tps[k], self.fps[k], self.fns[k], self.ious[k], self.wtps[k], self.wfps[k], self.wfns[k] = -1, -1, -1, -1, -1, -1, -1

            xymask = np.logical_and(xmask, ymask)
            xyweights = weights[xymask]
            xylabels = xik[xymask] + (2 ** 32) * yik[xymask]
            xyinstances = np.unique(xylabels)
            intersections = (np.broadcast_to(
                xyweights, (len(xyinstances), len(xyweights))) * xylabels == xyinstances[..., None]).sum(axis=1)
            xyxinstances, xyyinstances = xyinstances % (
                2 ** 32), xyinstances // (2 ** 32)

            xyxmask = xlabels == xyxinstances[..., None]
            xyxareas = (np.broadcast_to(
                xweights, (len(xyxinstances), len(xweights))) * xyxmask).sum(axis=1)

            xyymask = ylabels == xyyinstances[..., None]
            xyyareas = (np.broadcast_to(
                yweights, (len(xyyinstances), len(yweights))) * xyymask).sum(axis=1)

            unions = xyxareas + xyyareas - intersections
            ious = intersections / unions
            indices = ious > 0.5

            self.tps[k] += np.sum(indices)
            self.wtps[k] += np.sum(intersections[indices])
            self.ious[k] += np.sum(ious[indices])
            self.wious[k] += np.sum(intersections[indices] * ious[indices]) * self.wtps[k] / (self.wtps[k] + np.sum(intersections[indices]))

            xmatched[[xmapping[k] for k in xyxinstances[indices]]] = True
            ymatched[[ymapping[k] for k in xyyinstances[indices]]] = True

            self.fps[k] += np.sum(xmatched == False)
            self.fns[k] += np.sum(ymatched == False)

            self.wfps[k] += np.sum(xareas[xmatched == False])
            self.wfns[k] += np.sum(yareas[ymatched == False])

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
