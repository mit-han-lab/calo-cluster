import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from calo_cluster.clustering.meanshift import MeanShift
from calo_cluster.metrics.instance import PanopticQuality
from calo_cluster.evaluation.experiment import Experiment
from PyPDF2 import PdfFileMerger
from scipy.optimize import basinhopping
from tqdm.auto import tqdm


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
    results = basinhopping(f, x0=0.250, niter=niter, disp=True, minimizer_kwargs=minimizer_kwargs)
    return results.x, results.fun, all_results


def bandwidth_study(experiment, out_dir, nevents):
    out_path = Path(out_dir)
    events = experiment.get_events(split='train', n=nevents)

    optimal_bw, optimal_pq, results = optimize_bandwidth(events)
    print(f'optimal bandwidth = {optimal_bw}, for which pq = {optimal_pq}')
    bws = results.keys()
    pqs = results.values()

    bw_fig = px.scatter(x=bws, y=pqs, labels={
                        'x': 'bandwidth', 'y': 'pq'}, title='Bandwidth vs. Panoptic Quality')
    bw_image_path = out_path / 'bandwidth.pdf'
    bw_fig.write_image(str(bw_image_path), scale=10)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('wandb_version')
    parser.add_argument('--ckpt_name')
    parser.add_argument('out_dir')
    parser.add_argument('--n_events', '-n', default=100, type=int)
    args = parser.parse_args()
    experiment = Experiment(args.wandb_version, args.ckpt_name)
    bandwidth_study(experiment, args.out_dir, args.n_events)


if __name__ == '__main__':
    main()
