from hgcal_dev.evaluation.studies.base_study import BaseStudy
from hgcal_dev.clustering.meanshift import MeanShift
from tqdm import tqdm
import numpy as np

class IoUMatcher():
    '''Match true and predicted clusters using IoU. Can optionally include weighting and use of semantic labels.'''
    def __init__(self) -> None:
        pass

    def match(self, xi, yi, weights=None):
        ''' xi: predicted cluster labels
            yi: true cluster labels
            weights: weight for each sample'''
        if weights is None:
            weights = np.ones_like(xi, dtype=np.float64)

        xinstances = np.unique(xi)
        xmapping = {k: idx for idx, k in enumerate(xinstances)}
        xmatched = np.array([False] * xinstances.shape[0])
        xareas = (np.broadcast_to(weights, (len(xinstances), len(weights)))
                    * xi == xinstances[..., None]).sum(axis=1)

        yinstances = np.unique(yi)
        ymapping = {k: idx for idx, k in enumerate(yinstances)}
        ymatched = np.array([False] * yinstances.shape[0])
        yareas = (np.broadcast_to(weights, (len(yinstances), len(weights)))
                    * yi == yinstances[..., None]).sum(axis=1)

        xylabels = xi + (2 ** 32) * yi
        xyinstances = np.unique(xylabels)
        intersections = (np.broadcast_to(
            weights, (len(xyinstances), len(weights))) * (xylabels == xyinstances[..., None]).astype(int)).sum(axis=1)
        xyxinstances, xyyinstances = xyinstances % (
            2 ** 32), xyinstances // (2 ** 32)

        xyxmask = xi == xyxinstances[..., None]
        xyxareas = (np.broadcast_to(
            weights, (len(xyxinstances), len(weights))) * xyxmask).sum(axis=1)

        xyymask = yi == xyyinstances[..., None]
        xyyareas = (np.broadcast_to(
            weights, (len(xyyinstances), len(weights))) * xyymask).sum(axis=1)

        unions = xyxareas + xyyareas - intersections
        ious = intersections / unions
        indices = ious > 0.5
        return xyxinstances[indices], xyyinstances[indices], ious[indices]


class ClusteringStudy(BaseStudy):

    def __init__(self, experiment, energy_name='energy', clusterer=MeanShift(bandwidth=0.022)) -> None:
        self.energy_name = 'energy'
        self.clusterer = clusterer
        super().__init__(experiment)
    
    def energy_resolution(self, nevents=100):
        events = self.experiment.get_events(split='val', n=nevents)

    def pq(self, nevents=100):
        events = self.experiment.get_events(split='val', n=nevents)
        results = {}
        for event in tqdm(events):
            pq = event.pq()
            for k in pq:
                if k not in results:
                    results[k] = 0.0
                results[k] += pq[k] / nevents
        
        return results

def main():
    xi = np.array([0, 0, 1, 1, 1])
    yi = np.array([2, 2, 2, 1, 1])
    weights = np.array([10, 1, 1, 1, 1])
    matcher = IoUMatcher()
    xclusters, yclusters, ious = matcher.match(xi, yi, weights)
    print(f'xclusters={xclusters}')
    print(f'yclusters={yclusters}')
    print(f'ious={ious}')

if __name__ == '__main__':
    main()