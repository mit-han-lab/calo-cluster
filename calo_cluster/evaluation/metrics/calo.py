from dataclasses import dataclass
from typing import List, Union
import warnings

import numpy as np
import plotly.express as px

from calo_cluster.evaluation.metrics.instance import iou_match
from calo_cluster.evaluation.metrics.metric import Metric
from calo_cluster.clustering.base_clusterer import BaseClusterer

@dataclass
class ResponseMetric(Metric):
    bins: List[float]
    num_classes: int
    semantic: bool
    ignore_index: int
    ignore_semantic_labels: bool
    threshold: float
    min_hits: int
    clusterer: BaseClusterer = None
    use_target: bool = False
    target_instance_label_name: Union[str, None] = None

    
    def __post_init__(self):
        self.bins = np.array(self.bins)
        np.seterr(all="ignore")
        warnings.filterwarnings('ignore', 'Degrees of freedom <= 0 for slice.')
        self.reset()

    def reset(self):
        self.matched_truth = np.zeros((len(self.bins), self.num_classes), dtype=np.int32)
        self.unmatched_truth = np.zeros((len(self.bins), self.num_classes), dtype=np.int32)
        self.matched_pred = np.zeros((len(self.bins), self.num_classes), dtype=np.int32)
        self.unmatched_pred = np.zeros((len(self.bins), self.num_classes), dtype=np.int32)
        self.response = np.zeros((len(self.bins), self.num_classes), dtype=np.float32)
        self.resolution = np.zeros((len(self.bins), self.num_classes), dtype=np.float32)

    def add(self, outputs, targets, energy):
        # x = predicted, y = truth; s = semantic, i = instance
        xs, xi = outputs
        ys, yi = targets
        l, counts = np.unique(yi, return_counts=True)
        valid = (yi == l[counts >= self.min_hits][..., None]).any(axis=0)
        xs = xs[valid]
        xi = xi[valid]
        ys = ys[valid]
        yi = yi[valid]
        energy = energy[valid]
        outputs = (xs, xi)
        targets = (ys, yi)
        matched_pred, matched_truth, *_ = iou_match(
            outputs, targets, weights=energy, threshold=self.threshold, semantic=self.semantic, ignore_index=self.ignore_index, ignore_semantic_labels=self.ignore_semantic_labels, num_classes=self.num_classes)
        
        for k in matched_pred:
            yik = yi[ys == k]
            xik = xi[xs == k]
            ey = energy[ys == k]
            ex = energy[xs == k]
            all_truth = np.unique(yik)
            all_pred = np.unique(xik)


            matched_truth_energy = (ey * (yik == matched_truth[k][..., None]).astype(int)).sum(axis=1)
            matched_pred_energy = (ex * (xik == matched_pred[k][..., None]).astype(int)).sum(axis=1)

            mask = matched_truth_energy != 0.0
            matched_truth_energy = matched_truth_energy[mask]
            matched_pred_energy = matched_pred_energy[mask]

            inds = np.digitize(matched_truth_energy, self.bins) - 1
            bins, n = np.unique(inds, return_counts=True)
            response = matched_pred_energy / matched_truth_energy
            g_r = response * (inds == np.arange(len(self.bins))[..., None]).astype(int)
            binned_response = g_r.sum(axis=1)
            
            m = self.matched_truth[bins, k]
            mu_m = self.response[bins, k]
            mu_n = binned_response[bins] / n
            s_n = np.nanstd(np.where(g_r==0, np.nan, g_r), axis=1)[bins]
            s_n[np.isnan(s_n)] = 0
            s_m = self.resolution[bins, k]
            self.response[bins, k] = m / (m + n) * mu_m + n / (m + n) * mu_n
            self.resolution[bins, k] = (m / (m + n) * s_m**2 + n / (m + n) * s_n**2 + m*n/(m+n)**2 * (mu_m - mu_n)**2)**0.5

            self.matched_truth[bins, k] += n
            
            # count unmatched truth
            unmatched_truth = np.setdiff1d(
                all_truth, matched_truth[k], assume_unique=True)
            unmatched_truth = unmatched_truth[unmatched_truth != -1]
            yi_mask = yik == unmatched_truth[..., None]
            unmatched_truth_energy = (ey * yi_mask.astype(int)).sum(axis=1)
            inds = np.digitize(unmatched_truth_energy, self.bins) - 1
            bins, n = np.unique(inds, return_counts=True)
            self.unmatched_truth[bins, k] += n

            # count unmatched pred
            unmatched_pred = np.setdiff1d(
                all_pred, matched_pred[k], assume_unique=True)
            unmatched_pred = unmatched_pred[unmatched_pred != -1]
            xi_mask = xik == unmatched_pred[..., None]
            unmatched_pred_energy = (ex * xi_mask.astype(int)).sum(axis=1)
            inds = np.digitize(unmatched_pred_energy, self.bins) - 1
            bins, n = np.unique(inds, return_counts=True)
            self.unmatched_pred[bins, k] += n

            # count matched pred
            matched_pred = np.intersect1d(
                all_pred, matched_pred[k], assume_unique=True)
            matched_pred = matched_pred[matched_pred != -1]
            xi_mask = xik == matched_pred[..., None]
            matched_pred_energy = (ex * xi_mask.astype(int)).sum(axis=1)
            inds = np.digitize(matched_pred_energy, self.bins) - 1
            bins, n = np.unique(inds, return_counts=True)
            self.matched_pred[bins, k] += n

            

    def compute(self):
        # efficiency vs energy: % of truth clusters that were matched in each energy bin.
        # -- input: energy bins
        # -- find energy of each cluster, match truth/reco clusters, count matched/unmatched truth in each bin, save

        # response vs energy: mean response (reco/truth energy) in each energy bin for matched truth clusters.
        # -- input: energy bins
        # -- find mean response, n matched in each bin

        # fake rate vs energy: % of reco clusters that were unmatched in each energy bin.
        # -- input: energy bins
        # -- find n unmatched reco in each bin

        # resolution vs energy: std
        efficiency = self.matched_truth / (self.matched_truth + self.unmatched_truth)
        fake_rate = self.unmatched_pred / (self.unmatched_pred + self.matched_pred)
        response = self.response
        resolution = self.resolution

        for arr in [efficiency, fake_rate, response, resolution]:
            arr[np.isnan(arr)] = 0

        return efficiency, fake_rate, response, resolution

    def _save(self, path):
        efficiency, fake_rate, response, resolution = self.compute()
        for k in range(self.num_classes):
            if k in self.ignore_semantic_labels:
                continue
            p = path / f'{k}'
            p.mkdir(exist_ok=True, parents=True)
            efficiency_ = efficiency[:, k]
            fake_rate_ = fake_rate[:, k]
            response_ = response[:, k]
            resolution_ = resolution[:, k]
            eff_path = p / 'efficiency'
            fr_path = p / 'fake_rate'
            response_path = p / 'response'
            resolution_path = p / 'resolution'
            for arr, data_path in zip([efficiency_, fake_rate_, response_, resolution_], [eff_path, fr_path, response_path, resolution_path]):
                np.save(str(data_path), arr)

            x = [(self.bins[i] + self.bins[i+1]) / 2 for i in range(len(self.bins)-1)]
            x.append(self.bins[-1])
            fig = px.scatter(x=x, y=efficiency_, title='efficiency')
            fig.write_image(str(p / 'efficiency.png'), scale=4)

            fig = px.scatter(x=x, y=fake_rate_, title='fake rate')
            fig.write_image(str(p / 'fake_rate.png'), scale=4)

            fig = px.scatter(x=x, y=response_, error_y=resolution_, title='response')
            fig.write_image(str(p / 'response.png'), scale=4)

    def add_from_dict(self, subbatch):
        coordinates = subbatch['coordinates']
        pred_coordinates = subbatch['coordinates'] + subbatch['pred_offsets']
        semantic_labels = subbatch['semantic_labels']
        if self.use_target:
            pred_labels = subbatch[f'{self.target_instance_label_name}_mapped'].F.cpu().numpy()
        elif self.use_nn:
            pred_labels = self.clusterer.cluster(embedding=pred_coordinates, semantic_labels=semantic_labels)
        else:
            pred_labels = self.clusterer.cluster(embedding=coordinates, semantic_labels=semantic_labels)

        outputs = (subbatch['pred_semantic_labels'], pred_labels)
        targets = (subbatch['semantic_labels_mapped'], subbatch['instance_labels_mapped'])
        energy = subbatch['weights_mapped']

        self.add(outputs, targets, energy)

