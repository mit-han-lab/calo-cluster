import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pytorch_metric_learning.utils.loss_and_miner_utils import \
    get_all_pairs_indices
from tqdm import tqdm

from hgcal_dev.clustering.meanshift import MeanShift


def to_pairs(y):
    p1_inds, p2_inds, _, _ = get_all_pairs_indices(torch.as_tensor(y))
    return set(map(lambda x: tuple(sorted(x)), ((float(p1_inds[i]), float(p2_inds[i])) for i in range(len(p1_inds)))))

def efficiency(true_pairs, pred_pairs):
    return sum((p in pred_pairs for p in true_pairs)) / len(true_pairs)

def purity(true_pairs, pred_pairs):
    return sum((p in true_pairs for p in pred_pairs)) / len(pred_pairs)

def match_clusters(x, y_true, y_pred):
    for cluster_id in np.unique(y_pred):
        p_inds = np.where(y_pred == cluster_id)
        true_ids = y_true[p_inds]
        E = x['energy', p_inds]

def cluster_metrics(x, y_true, y_pred):
    true_pairs = to_pairs(y_true)
    pred_pairs = to_pairs(y_pred)
    
    eff = efficiency(true_pairs, pred_pairs)
    p = purity(true_pairs, pred_pairs)

    return eff, p

def eval_preds(prediction_dir, input_dir):
    effs = []
    ps = []
    clusterer = MeanShift()
    for pred_path in tqdm(prediction_dir.glob('*.npz')):
        event_name = pred_path.stem
        input_path = input_dir / f'{event_name}.pkl'
        event_prediction = np.load(pred_path)
        x = pd.read_pickle(input_path)
        x = x.loc[event_prediction['inds']].reset_index(drop=True)

        y_pred = clusterer.cluster(event_prediction['prediction'])
        y_random = clusterer.cluster(np.random.rand(event_prediction['prediction'].shape[0], event_prediction['prediction'].shape[1]))
        y_true = event_prediction['labels']

        eff, p = cluster_metrics(x, y_true, y_pred)
        eff_rand, p_rand = cluster_metrics(x, y_true, y_random)
        effs.append(eff)
        ps.append(p)
        print(f'eff = {eff}')
        print(f'purity = {p}')

        print(f'random eff = {eff_rand}')
        print(f'random purity = {p_rand}')

    print(f'mean efficiency = {sum(effs) / len(effs)}')
    print(f'mean purity = {sum(ps) / len(ps)}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prediction_dir')
    parser.add_argument('input_dir')

    args = parser.parse_args()
    prediction_dir = Path(args.prediction_dir)
    input_dir = Path(args.input_dir)
    eval_preds(prediction_dir, input_dir)

if __name__ == '__main__':
    main()
