import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.evaluation.studies.base_study import BaseStudy
from hgcal_dev.evaluation.utils import get_palette
from tqdm import tqdm
import logging
import time
import hgcal_dev.evaluation.studies.functional as F
from hgcal_dev.evaluation.metrics.instance import PanopticQuality

class ClusteringStudy(BaseStudy):

    def __init__(self, experiment, energy_name='energy', clusterer=None) -> None:
        assert experiment.task == 'panoptic' or experiment.task == 'instance'
        self.energy_name = energy_name
        super().__init__(experiment, clusterer)

    @property
    def clusterer(self):
        return self._clusterer

    @clusterer.setter
    def clusterer(self, c):
        self._clusterer = c

    def resolution(self, nevents=100, nbins=11, lo=0, hi=10, splits=('val',), out_dir='.'):
        out_dir = self.out_dir / out_dir
        out_dir.mkdir(exist_ok=True, parents=True)
        for split in splits:
            events = self.experiment.get_events(split=split, n=nevents)
            cluster_df, event_df = F.resolution(events, self.clusterer)
            fig = px.histogram(cluster_df, x='energy resolution')
            out_path = out_dir / f'{split}_energy_resolution_histogram.png'
            fig.write_image(str(out_path), scale=10)

            means, errors, bin_edges = F.make_bins(cluster_df['energy'], cluster_df['energy_resolution'], nbins=nbins, lo=lo, hi=hi)
            bin_df = pd.DataFrame({'resolution': means, 'error': errors, 'energy': bin_edges})
            px.scatter(bin_df, x='energy', y='resolution', error_y='error')

    def pq(self, nevents=100, use_weights=True, ignore_semantic_labels=None):
        events = self.experiment.get_events(split='val', n=nevents)
        results = {}
        for event in tqdm(events):
            pred_instance_labels = self.clusterer.cluster(event)
            if self.experiment.task == 'panoptic':
                pq_metric = PanopticQuality(num_classes=event.num_classes, ignore_index=-1, ignore_semantic_labels=ignore_semantic_labels)

                outputs = (event.pred_class_labels, pred_instance_labels)
                targets = (event.input_event[event.class_label].values,
                        event.input_event[event.instance_label].values)
            elif self.experiment.task == 'instance':
                pq_metric = PanopticQuality(semantic=False, ignore_index=-1)

                outputs = pred_instance_labels
                targets = event.input_event[event.instance_label].values

            if use_weights:
                if event.weight_name is None:
                    raise RuntimeError('No weight name given!')
                weights = event.input_event[event.weight_name].values
            else:
                weights = None
            pq_metric.add(outputs, targets, weights=weights)

            pq = pq_metric.compute()
            for k in pq:
                if k not in results:
                    results[k] = 0.0
                results[k] += pq[k] / nevents
        return results

    def _qualitative_plot(self, out_dir, split, i, event):
        xi = self.clusterer.cluster(event)
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            size = event.weight_name
            weights = event.input_event[event.weight_name].values
        else:
            size = None
            weights = None
        pred_clusters, truth_clusters, _ = F.iou_match(
            xi, yi, weights=weights)
        pred_clusters = pred_clusters.astype(str)
        truth_clusters = truth_clusters.astype(str)
        plot_df = event.input_event.copy()
        plot_df['pred_instance_labels'] = xi
        plot_df['pred_instance_labels'] = plot_df['pred_instance_labels'].astype(
            str)
        plot_df.loc[~plot_df['pred_instance_labels'].isin(
            pred_clusters), 'pred_instance_labels'] = 'unmatched'

        plot_df['truth_instance_labels'] = plot_df[event.instance_label].astype(
            str)
        plot_df.loc[~plot_df['truth_instance_labels'].isin(
            truth_clusters), 'truth_instance_labels'] = 'unmatched'

        new_pred_instance_labels = plot_df['pred_instance_labels'].copy()
        for i, c in enumerate(truth_clusters):
            new_pred_instance_labels[plot_df['pred_instance_labels'] ==
                                     pred_clusters[i]] = c
        plot_df['pred_instance_labels'] = new_pred_instance_labels
        if len(plot_df['pred_instance_labels'].unique()) > len(plot_df['truth_instance_labels'].unique()):
            larger_set = plot_df['pred_instance_labels'].unique()
        else:
            larger_set = plot_df['truth_instance_labels'].unique()
        color_discrete_sequence = get_palette(larger_set)
        color_discrete_map = {}
        for color, cluster in zip(color_discrete_sequence, larger_set):
            color_discrete_map[cluster] = color

        fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='pred_instance_labels',
                            size=size, color_discrete_map=color_discrete_map)
        out_path = out_dir / f'{split}_{i}_matched_instance_pred.png'
        fig.write_image(str(out_path), scale=10)
            
        fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='truth_instance_labels',
                            size=size, color_discrete_map=color_discrete_map)
        out_path = out_dir / f'{split}_{i}_matched_instance_truth.png'
        fig.write_image(str(out_path), scale=10)
        return super()._qualitative_plot(out_dir, split, i, event)


def main():
    #xi = np.array([0, 0, 1, 1, 1])
    #yi = np.array([2, 2, 2, 1, 1])
    #weights = np.array([10, 1, 1, 1, 1])
    #matcher = IoUMatcher()
    #xclusters, yclusters, ious = matcher.match(xi, yi, weights)
    # print(f'xclusters={xclusters}')
    # print(f'yclusters={yclusters}')
    # print(f'ious={ious}')
    from hgcal_dev.evaluation.experiments.simple_experiment import \
        SimpleExperiment
    exp = SimpleExperiment('dhc9f7wl')
    study = ClusteringStudy(exp)
    study.resolution(nevents=100)


if __name__ == '__main__':
    main()
