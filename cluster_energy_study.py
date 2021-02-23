import argparse
from pathlib import Path

from hgcal_dev.utils.experiment import Experiment
from hgcal_dev.metrics.instance import PanopticQuality
import plotly.express as px
from scipy.optimize import minimize_scalar
from sklearn import cluster
from PyPDF2 import PdfFileMerger
import pandas as pd
import numpy as np
from tqdm import tqdm

import plotly.io as pio

def top_n_clusters(events: list, n: int):
    out = []
    for event in events:
        out.append(event.top_n_clusters(n))
    return out

def optimize_bandwidth(events: list, max_clusters: int):
    n_clusters = list(range(1, max_clusters+1))
    bws, pqs = np.zeros(len(n_clusters)), np.zeros(len(n_clusters))
    for i, n in enumerate(tqdm(n_clusters)):
        filtered_events = top_n_clusters(events, n)
        # TODO: generalize for non-hcal
        def evaluate(bw: float):
            for event in filtered_events:
                clusterer = cluster.MeanShift(bandwidth=bw, bin_seeding=True, n_jobs=-1)
                pred_instance_labels = clusterer.fit_predict(event.embedding)
                _, _, pq = event.pq(pred_instance_labels)
                return pq

        pq = PanopticQuality(num_classes=2, min_points=2)
        results = minimize_scalar(evaluate, bounds=[0.001, 0.5], method='bounded', options={'maxiter': 10})
        bws[i] = results.x
        pqs[i] = results.fun
    return bws, pqs, n_clusters

def bandwidth_study(run_dir: str, out_dir: str, max_clusters: int, n_events: int, ckpt_path: str):
    run_path = Path(run_dir)
    out_path = Path(out_dir)
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        experiment = Experiment(run_path, ckpt_path)
    else:
        experiment = Experiment(run_path)
    events = experiment.get_events(split='train', n=n_events)
    bws, pqs, n_clusters = optimize_bandwidth(events, max_clusters)

    bw_fig = px.scatter(x=n_clusters, y=bws, labels={'x': 'n_clusters', 'y': 'bandwidth'}, title='Optimal Bandwidth vs. Top k Clusters')
    bw_image_path = out_path / 'bandwidth.pdf'
    bw_fig.write_image(str(bw_image_path), width=1200, height=800)

    pq_fig = px.scatter(x=n_clusters, y=pqs, labels={'x': 'n_clusters', 'y': 'pq'}, title='PQ vs. Top k Clusters')
    pq_image_path = out_path / 'pq.pdf'
    pq_fig.write_image(str(pq_image_path), width=1200, height=800)

    bw_txt_path = out_path / 'bandwidth.txt'
    with bw_txt_path.open('w+') as f:
        f.write(' '.join(bws))

    pq_txt_path = Path.cwd() / 'pq.txt'
    with pq_txt_path.open('w+') as f:
        f.write(' '.join(pqs))
    

def evaluate_pqs(events, max_clusters, bandwidth):
    n_clusters = list(range(1, max_clusters+1))
    pqs = np.zeros(len(n_clusters))
    for i, n in enumerate(tqdm(n_clusters)):
        filtered_events = top_n_clusters(events, n)
        clusterer = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)
        for event in filtered_events:
            pred_instance_labels = clusterer.fit_predict(event.embedding)
            _, _, pq = event.pq(pred_instance_labels)
            pqs[i] = pq
    return pqs, n_clusters

def pq_study(run_dir, out_dir, max_clusters, n_events, bandwidth, ckpt_path):
    run_path = Path(run_dir)
    out_path = Path(out_dir)
    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        experiment = Experiment(run_path, ckpt_path)
    else:
        experiment = Experiment(run_path)
    events = experiment.get_events(split='val', n=n_events)
    pqs, n_clusters = evaluate_pqs(events, max_clusters, bandwidth)

    pq_fig = px.scatter(x=n_clusters, y=pqs, labels={'x': 'n_clusters', 'y': 'pq'}, title='PQ vs. Top k Clusters')
    pq_image_path = out_path / 'pq.pdf'
    pq_fig.write_image(str(pq_image_path), width=1200, height=800)

    pq_txt_path = Path.cwd() / 'pq.txt'
    with pq_txt_path.open('w+') as f:
        f.write(' '.join(pqs))

def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest='func')
    bandwidth_sp = subparsers.add_parser('bandwidth')
    bandwidth_sp.add_argument('wandb_run_dir')
    bandwidth_sp.add_argument('out_dir')
    bandwidth_sp.add_argument('--ckpt_path')
    bandwidth_sp.add_argument('--max_clusters', '-c',
                              required=False, default=10, type=int)
    bandwidth_sp.add_argument(
        '--n_events', '-n', required=False, default=10, type=int)

    pq_sp = subparsers.add_parser('pq')
    pq_sp.add_argument('wandb_run_dir')
    pq_sp.add_argument('--max_clusters', '-c',
                              required=False, default=10, type=int)
    pq_sp.add_argument(
        '--n_events', '-n', required=False, default=100, type=int)
    pq_sp.add_argument('--bandwidth', '-bw', type=float, default=0.225)

    args = parser.parse_args()

    if args.func == 'bandwidth':
        bandwidth_study(args.wandb_run_dir, args.out_dir, args.max_clusters, args.n_events, args.ckpt_path)
    elif args.func == 'pq':
        pq_study(args.wandb_run_dir, args.out_dir, args.max_clusters, args.n_events, args.bandwidth, args.ckpt_path)


if __name__ == '__main__':
    main()
