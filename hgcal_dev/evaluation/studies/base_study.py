import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from hgcal_dev.clustering.meanshift import MeanShift
from hgcal_dev.metrics.instance import PanopticQuality
from PyPDF2 import PdfFileMerger
from scipy.optimize import basinhopping
from tqdm import tqdm
from ..utils import get_palette


class BaseStudy:
    def __init__(self, experiment) -> None:
        self.experiment = experiment
        self.out_dir = Path('/home/alexj/plots') / self.experiment.wandb_version
        self.out_dir.mkdir(exist_ok=True, parents=True)

    def optimize_bandwidth(events: list, niter=200):
        all_results = {}
        n = len(events)

        def f(bw: float):
            bw = bw[0]
            wpq = 0.0
            for event in events:
                clusterer = MeanShift(use_gpu=True, bandwidth=bw)
                pred_instance_labels = clusterer.cluster(event.embedding)
                results = event.pq(pred_instance_labels)
                wpq += results['wpq'] / n
            all_results[bw] = wpq
            return wpq
        bounds = [(0.001, 1.0)]
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        results = basinhopping(f, x0=0.250, niter=niter,
                               disp=True, minimizer_kwargs=minimizer_kwargs)
        return results.x, results.fun, all_results

    def bandwidth_study(self, experiment, out_dir, nevents=10, niter=200):
        out_path = Path(out_dir)
        events = experiment.get_events(split='train', n=nevents)

        results = {}
        n = len(events)

        def f(bw: float):
            bw = bw[0]
            wpq = 0.0
            for event in events:
                clusterer = MeanShift(use_gpu=True, bandwidth=bw)
                pred_instance_labels = clusterer.cluster(event.embedding)
                results = event.pq(pred_instance_labels)
                wpq += results['wpq'] / n
            results[bw] = wpq
            return wpq
        bounds = [(0.001, 1.0)]
        minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds}
        optimizer_results = basinhopping(f, x0=0.01, niter=niter,
                               disp=True, minimizer_kwargs=minimizer_kwargs)
        optimal_bw = optimizer_results.x
        optimal_pq = optimizer_results.fun
        
        print(f'optimal bandwidth = {optimal_bw}, for which pq = {optimal_pq}')
        bws = results.keys()
        pqs = results.values()

        bw_fig = px.scatter(x=bws, y=pqs, labels={
                            'x': 'bandwidth', 'y': 'pq'}, title='Bandwidth vs. Panoptic Quality')
        bw_image_path = out_path / 'bandwidth.pdf'
        bw_fig.write_image(str(bw_image_path), scale=10)

    def qualitative_cluster_study(self, n=5):
        out_dir = self.out_dir / 'scatter'
        out_dir.mkdir(exist_ok=True)

        for split in ('train', 'val'):
            for i, event in enumerate(self.experiment.get_events(split='train', n=n)):
                plot_df = event.input_event
                if self.experiment.task == 'instance' or self.experiment.task == 'panoptic':
                    plot_df['pred_instance_labels'] = event.pred_instance_labels
                    plot_df['pred_instance_labels'] = plot_df['pred_instance_labels'].astype(str)
                    plot_df['truth_instance_labels'] = plot_df[event.instance_label].astype(str)

                    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='pred_instance_labels', color_discrete_sequence=get_palette(event.pred_instance_labels))
                    out_path = out_dir / f'{split}_{i}_instance_pred.png'
                    fig.write_image(str(out_path), scale=10)

                    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='truth_instance_labels', color_discrete_sequence=get_palette(plot_df[event.instance_label]))
                    out_path = out_dir / f'{split}_{i}_instance_truth.png'
                    fig.write_image(str(out_path), scale=10)

                if self.experiment.task == 'semantic' or self.experiment.task == 'panoptic':
                    plot_df['pred_semantic_labels'] = event.pred_class_labels
                    plot_df['truth_semantic_labels'] = plot_df[event.class_label]

                    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='pred_semantic_labels')
                    out_path = out_dir / f'{split}_{i}_semantic_pred.png'
                    fig.write_image(str(out_path), scale=10)

                    fig = px.scatter_3d(plot_df, x='x', y='y', z='z', color='truth_semantic_labels')
                    out_path = out_dir / f'{split}_{i}_semantic_truth.png'
                    fig.write_image(str(out_path), scale=10)



