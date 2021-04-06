from hgcal_dev.evaluation.studies.base_study import BaseStudy
from hgcal_dev.clustering.meanshift import MeanShift
from tqdm import tqdm

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