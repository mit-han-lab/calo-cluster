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


class ClusteringStudy(BaseStudy):

    def __init__(self, experiment, energy_name='energy', clusterer=MeanShift(bandwidth=0.022)) -> None:
        self.energy_name = 'energy'
        self.clusterer = clusterer
        super().__init__(experiment)

    def resolution(self, nevents=100, bin_edges=[0.0, 0.5, 1.0, 3, 5, 10], range_x=(0, 2), splits=('train', 'val'), out_dir='.'):
        out_dir = self.out_dir / out_dir
        out_dir.mkdir(exist_ok=True, parents=True)
        for split in splits:
            events = self.experiment.get_events(split=split, n=nevents)
            plot_df, event_df = F.resolution(events)
            fig = px.histogram(plot_df, x='energy resolution')
            out_path = out_dir / f'{split}_energy_resolution_histogram.png'
            fig.write_image(str(out_path), scale=10)

            mean_resolutions = np.zeros(len(bin_edges) - 1)
            std_resolutions = np.zeros(len(bin_edges) - 1)
            bin_energies = np.zeros(len(bin_edges) - 1)
            for i in range(len(bin_edges) - 1):
                start = bin_edges[i]
                end = bin_edges[i+1]
                data = plot_df[(plot_df['energy'] > start) & (plot_df['energy'] < end)]
                mean_resolutions[i] = data['energy resolution'].mean()
                std_resolutions[i] = data['energy resolution'].std()
                bin_energies[i] = (end + start) / 2

            plot_df_2 = pd.DataFrame({'mean_resolution': mean_resolutions, 'std_resolution': std_resolutions, 'energy': bin_energies})
            fig = px.scatter(plot_df_2, x='energy', y='mean_resolution', title='mean of energy resolution')
            out_path = self.out_dir / f'{split}_mean_energy_resolution.png'
            fig.write_image(str(out_path), scale=10)
            fig = px.scatter(plot_df_2, x='energy', y='std_resolution', title='standard deviation of energy resolution')
            out_path = self.out_dir / f'{split}_std_energy_resolution.png'
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
        xi = event.pred_instance_labels
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
