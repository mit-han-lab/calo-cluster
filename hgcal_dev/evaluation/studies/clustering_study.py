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

        xylabels = xi + (2 ** 32) * yi
        xyinstances = np.unique(xylabels)

        xymask = xylabels == xyinstances[..., None]
        intersections = (weights * xymask.astype(int)).sum(axis=1)

        xyxinstances, xyyinstances = xyinstances % (
            2 ** 32), xyinstances // (2 ** 32)

        xyxmask = xi == xyxinstances[..., None]
        xyxareas = (weights * xyxmask.astype(int)).sum(axis=1)

        xyymask = yi == xyyinstances[..., None]
        xyyareas = (weights * xyymask.astype(int)).sum(axis=1)

        unions = xyxareas + xyyareas - intersections
        ious = intersections / unions
        indices = ious > 0.5
        return xyxinstances[indices], xyyinstances[indices], ious[indices]


class ClusteringStudy(BaseStudy):

    def __init__(self, experiment, energy_name='energy', clusterer=MeanShift(bandwidth=0.022)) -> None:
        self.energy_name = 'energy'
        self.clusterer = clusterer
        super().__init__(experiment)

    def resolution(self, nevents=100, nbins=10, range_x=(0, 2)):
        for split in ('train', 'val'):
            events = self.experiment.get_events(split=split, n=nevents)
            resolutions = []
            energies = []
            n_unmatched = 0
            for event in tqdm(events):
                matcher = IoUMatcher()
                xi = event.pred_instance_labels
                yi = event.input_event[event.instance_label].values
                if event.weight_name:
                    weights = event.input_event[event.weight_name].values
                else:
                    weights = None
                matched_pred, matched_truth, _ = matcher.match(
                    xi, yi, weights=weights)
                all_truth = np.unique(yi)
                yim_mask = yi == matched_truth[..., None]
                matched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

                xim_mask = xi == matched_pred[..., None]
                matched_pred_energies = (weights * xim_mask.astype(int)).sum(axis=1)
                resolution = matched_pred_energies / matched_truth_energies
                resolutions.append(resolution)
                energies.append(matched_truth_energies)

                unmatched_truth = np.setdiff1d(all_truth, matched_truth, assume_unique=True)
                yim_mask = yi == unmatched_truth[..., None]
                unmatched_truth_energies = (weights * yim_mask.astype(int)).sum(axis=1)

                n_unmatched = len(all_truth) - len(matched_truth)
                resolutions.append(np.zeros(n_unmatched))
                energies.append(unmatched_truth_energies)
            resolution = np.concatenate(resolutions)
            energy = np.concatenate(energies)

            plot_df = pd.DataFrame({'energy resolution': resolution, 'energy': energy})
            fig = px.histogram(plot_df, x='energy resolution')
            out_path = self.out_dir / f'{split}_energy_resolution_histogram.png'
            fig.write_image(str(out_path), scale=10)

            _, bin_edges = np.histogram(plot_df['energy'], bins=nbins)
            for i in range(nbins):
                start = bin_edges[i]
                end = bin_edges[i+1]
                fig = px.histogram(plot_df[(plot_df['energy'] > start) & (plot_df['energy'] < end)], x='energy resolution', title=f'{start:.3} < energy < {end:.3}', range_x=range_x, nbins=40)
                out_path = self.out_dir / f'{split}_{i}_energy_resolution.png'
                fig.write_image(str(out_path), scale=10)

    def pq(self, nevents=100, use_weights=False):
        events = self.experiment.get_events(split='val', n=nevents)
        results = {}
        for event in tqdm(events):
            pq = event.pq(use_weights=use_weights)
            for k in pq:
                if k not in results:
                    results[k] = 0.0
                results[k] += pq[k] / nevents
        return results

    def _qualitative_plot(self, out_dir, split, i, event):
        matcher = IoUMatcher()
        xi = event.pred_instance_labels
        yi = event.input_event[event.instance_label].values
        if event.weight_name:
            size = event.weight_name
            weights = event.input_event[event.weight_name].values
        else:
            size = None
            weights = None
        pred_clusters, truth_clusters, _ = matcher.match(
            xi, yi, weights=weights)
        pred_clusters = pred_clusters.astype(str)
        truth_clusters = truth_clusters.astype(str)
        plot_df = event.input_event.copy()
        plot_df['pred_instance_labels'] = event.pred_instance_labels
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
