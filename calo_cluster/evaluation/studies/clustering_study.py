import calo_cluster.evaluation.studies.functional as F
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
from calo_cluster.evaluation.metrics.instance import iou_match
from calo_cluster.evaluation.studies.base_study import BaseStudy
from calo_cluster.evaluation.utils import get_palette
from scipy.optimize import basinhopping
from tqdm.auto import tqdm

from calo_cluster.evaluation.experiments.base_experiment import BaseExperiment

class ClusteringStudy(BaseStudy):

    def __init__(self, experiment: BaseExperiment, energy_name: str = 'energy', clusterer=None) -> None:
        assert experiment.task == 'panoptic' or experiment.task == 'instance'
        self.energy_name = energy_name
        super().__init__(experiment, clusterer)

    def get_clusters(self, nevents, split, match_highest=False):
        events = self.experiment.get_events(split=split, n=nevents)
        truth_clusters = []
        pred_clusters = []
        truth_hits = []
        pred_hits = []
        for event in tqdm(events):
            xi = self.clusterer.cluster(event)
            yi = event.input_event[event.instance_label].values
            df = event.input_event
            df['weta'] = df['eta'] * df['energy']
            df['wphi'] = df['phi'] * df['energy']
            result = df.groupby([event.instance_label])[
                ['energy', 'weta', 'wphi']].agg(['mean', 'count'])
            energy = result[('energy', 'mean')]
            energy.name = 'energy'
            nconstituents = result[('energy', 'count')]
            nconstituents.name = 'nconstituents'
            eta = result[('weta', 'mean')] / energy
            eta.name = 'eta'
            phi = result[('wphi', 'mean')] / energy
            phi.name = 'phi'
            if event.weight_name:
                weights = event.input_event[event.weight_name].values
            else:
                weights = None
            if self.clusterer.use_semantic:
                xs = event.pred_semantic_labels
                ys = event.input_event[event.semantic_label].values
                outputs = (xs, xi)
                targets = (ys, yi)
            else:
                outputs = xi
                targets = yi
            matched_pred, matched_truth, *_ = iou_match(
                outputs, targets, weights=weights, ignore_semantic_labels=(self.clusterer.ignore_semantic_label,), semantic=self.clusterer.use_semantic, match_highest=match_highest, num_classes=self.experiment.num_classes)
            if 1 in matched_pred:
                matched_pred = matched_pred[1]
                matched_truth = matched_truth[1]
            else:
                matched_pred = matched_pred[0]
                matched_truth = matched_truth[0]
            for pred_label, truth_label in zip(matched_pred, matched_truth):
                _truth_hits = event.input_event[yi == truth_label].copy()
                _pred_hits = event.input_event[xi == pred_label].copy()
                c_energy = _truth_hits['energy'].sum()
                _truth_hits['weta'] = _truth_hits['eta'] * _truth_hits['energy']
                _truth_hits['wphi'] = _truth_hits['phi'] * _truth_hits['energy']
                c_eta = _truth_hits['weta'].sum() / c_energy
                c_phi = _truth_hits['wphi'].sum() / c_energy
                c_n_constituents = len(_truth_hits)
                c_pred_energy = _pred_hits['energy'].sum()
                c_response =  c_pred_energy / c_energy
                truth_clusters.append([c_energy, c_eta, c_phi, c_n_constituents, c_pred_energy, c_response])
                _pred_hits['weta'] = _pred_hits['eta'] * _pred_hits['energy']
                _pred_hits['wphi'] = _pred_hits['phi'] * _pred_hits['energy']
                c_pred_eta = _pred_hits['weta'].sum() / c_pred_energy
                c_pred_phi = _pred_hits['wphi'].sum() / c_pred_energy
                c_pred_n_constituents = len(_pred_hits)
                pred_clusters.append([c_pred_energy, c_pred_eta, c_pred_phi, c_pred_n_constituents])
                truth_hits.append(_truth_hits)
                pred_hits.append(_pred_hits)

        truth_clusters = pd.DataFrame(truth_clusters, columns=['energy', 'eta', 'phi', 'n_constituents', 'pred_energy', 'response'])
        pred_clusters = pd.DataFrame(pred_clusters, columns=['energy', 'eta', 'phi', 'n_constituents'])
        return truth_clusters, truth_hits, pred_clusters, pred_hits

    def response(self, nevents=100, nbins=21, lo=0, hi=20, splits=('val',), out_dir='.', match_highest=False):
        out_dir = self.out_dir / out_dir
        if match_highest:
            out_dir = out_dir / 'match_highest'
        else:
            out_dir = out_dir / 'match_threshold'
        out_dir.mkdir(exist_ok=True, parents=True)
        data_dict = {}
        for split in splits:
            data_dict[split] = {}
            events = self.experiment.get_events(split=split, n=nevents)
            cluster_dfs, event_dfs = F.response(
                events, self.clusterer, match_highest, num_classes=self.experiment.num_classes)
            data_dict[split]['cluster_dfs'] = cluster_dfs
            data_dict[split]['event_dfs'] = event_dfs
            for k in cluster_dfs:
                cluster_df = cluster_dfs[k]
                fig = px.histogram(cluster_df, x='energy response')
                out_path = out_dir / \
                    f'{split}_class_{k}_energy_response_histogram.png'
                fig.write_image(str(out_path), scale=10)

                bin_edges = np.linspace(lo, hi, num=nbins)
                means, standard_errors = F.make_bins(
                    cluster_df['energy'], cluster_df['energy response'], bin_edges=bin_edges, use_standard_error=True)
                _, errors = F.make_bins(
                    cluster_df['energy'], cluster_df['energy response'], bin_edges=bin_edges, use_standard_error=False)
                data_dict[split][k] = {}
                data_dict[split][k]['means'] = means
                data_dict[split][k]['standard_error'] = standard_errors
                data_dict[split][k]['rms'] = errors
                data_dict[split][k]['energy'] = bin_edges
                bin_df = pd.DataFrame(
                    {'response': means, 'standard_error': standard_errors, 'rms': errors, 'energy': bin_edges})
                fig = px.scatter(bin_df, x='energy',
                                 y='response', error_y='standard_error')
                out_path = out_dir / \
                    f'{split}_class_{k}_binned_energy_response_histogram.png'
                fig.write_image(str(out_path), scale=10)

                fig = px.scatter(bin_df, x='energy',
                                 y='rms')
                out_path = out_dir / \
                    f'{split}_class_{k}_binned_rms_histogram.png'
                fig.write_image(str(out_path), scale=10)

                matched_mask = cluster_df['energy response'] >= 0.5
                means, errors = F.make_bins(
                    cluster_df['energy'], matched_mask.astype(int), bin_edges=bin_edges, use_standard_error=True)
                count_df = pd.DataFrame(
                    {'matched_frac': means, 'error': errors, 'energy': bin_edges})
                fig = px.scatter(count_df, x='energy',
                                 y='matched_frac', error_y='error')
                out_path = out_dir / \
                    f'{split}_class_{k}_binned_n_matched_histogram.png'
                fig.write_image(str(out_path), scale=10)

                
                fig = px.density_heatmap(cluster_df[(cluster_df['energy response'] < 10) & (cluster_df['energy'] < 100)], x='energy', y='energy response', nbinsx=100, nbinsy=100)
                out_path = out_dir / \
                    f'{split}_class_{k}_density_heatmap_energy_response_histogram.png'
                fig.write_image(str(out_path), scale=10)

        return data_dict

    def pq(self, nevents=100, use_weights=True, ignore_semantic_labels=None, split='val'):
        events = self.experiment.get_events(split=split, n=nevents)
        nevents = len(events)
        results = {}
        for event in tqdm(events):
            pq = F.pq(event, use_weights, self.clusterer,
                      self.experiment.task, ignore_semantic_labels)
            for k in pq:
                if k not in results:
                    results[k] = 0.0
                if type(pq[k]) == np.ndarray:
                    pq[k][pq[k] == -1] = 0.0
                elif pq[k] == -1:
                    pq[k] = 0.0
                results[k] += pq[k] / nevents
        return results

    def _qualitative_plot(self, out_dir, split, i, event):
        xi = self.clusterer.cluster(event)
        yi = event.input_event[event.instance_label].values
        # TODO: change this so it's not hardcoded
        hit_mask = (event.input_event[event.semantic_label] == 1).values
        xi = xi[hit_mask]
        yi = yi[hit_mask]
        if event.weight_name:
            size = event.weight_name
            weights = event.input_event[event.weight_name].values
            weights = weights[hit_mask]
        else:
            size = None
            weights = None
        _pred_clusters, _truth_clusters, *_ = iou_match(
            xi, yi, weights=weights, num_classes=self.experiment.num_classes)
        for k in _pred_clusters:
            pred_clusters = _pred_clusters[k].astype(str)
            truth_clusters = _truth_clusters[k].astype(str)
            plot_df = event.input_event[hit_mask].copy()
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
            out_path = out_dir / \
                f'class_{k}_{split}_{i}_matched_instance_pred.png'
            fig.write_image(str(out_path), scale=10)

            fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='truth_instance_labels',
                                size=size, color_discrete_map=color_discrete_map)
            out_path = out_dir / \
                f'class_{k}_{split}_{i}_matched_instance_truth.png'
            fig.write_image(str(out_path), scale=10)
        return super()._qualitative_plot(out_dir, split, i, event)

    def bandwidth_study(self, clusterer_factory, nevents=-1, out_dir='.', x0=0.02, bounds=(0.001, 1.0), use_weights: bool = True, ignore_semantic_labels: list = None, niter: int = 200, T: float = 0.1, stepsize: float = 0.05):
        out_dir = self.out_dir / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        events = self.experiment.get_events(split='train', n=nevents)

        optimal_bw, optimal_pq, results = self._optimize_bandwidth(
            events, clusterer_factory, x0, bounds, use_weights, ignore_semantic_labels, niter, T, stepsize)
        print(f'optimal bandwidth = {optimal_bw}, for which pq = {optimal_pq}')
        bws = results.keys()
        pqs = results.values()

        bw_fig = px.scatter(x=bws, y=pqs, labels={
                            'x': 'bandwidth', 'y': 'pq'}, title='Bandwidth vs. Panoptic Quality')
        bw_image_path = out_dir / 'bandwidth.png'
        bw_fig.write_image(str(bw_image_path), scale=10)

        data_dict = {}
        data_dict['bws'] = bws
        data_dict['pqs'] = pqs
        data_dict['optimal_bw'] = optimal_bw
        data_dict['optimal_pq'] = optimal_pq

        figs_dict = {}
        figs_dict['bandwidth_vs_pq'] = bw_fig

        return data_dict, figs_dict

    def _optimize_bandwidth(self, events: list, clusterer_factory, x0: float, bounds: tuple, use_weights: bool, ignore_semantic_labels: list, niter: int, T: float, stepsize: float):
        all_results = {}
        n = len(events)

        def f(bw: float):
            bw = bw[0]
            wpq = 0.0
            for event in events:
                clusterer = clusterer_factory(bw)
                results = F.pq(event, use_weights, clusterer,
                               self.experiment.task, ignore_semantic_labels)
                wpq += results['wpq'] / n
            all_results[bw] = wpq
            return -wpq
        bounds = [bounds]
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        results = basinhopping(f, x0=x0, niter=niter,
                               disp=True, T=T, stepsize=stepsize, minimizer_kwargs=minimizer_kwargs)
        return results.x, results.fun, all_results


def main():
    #xi = np.array([0, 0, 1, 1, 1])
    #yi = np.array([2, 2, 2, 1, 1])
    #weights = np.array([10, 1, 1, 1, 1])
    #matcher = IoUMatcher()
    #xclusters, yclusters, ious = matcher.match(xi, yi, weights)
    # print(f'xclusters={xclusters}')
    # print(f'yclusters={yclusters}')
    # print(f'ious={ious}')
    from calo_cluster.evaluation.experiments.simple_experiment import \
        SimpleExperiment
    exp = SimpleExperiment('dhc9f7wl')
    study = ClusteringStudy(exp)
    study.response(nevents=100)


if __name__ == '__main__':
    main()
