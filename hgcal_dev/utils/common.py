import os

import numpy as np
import torch

__all__ = ['get_save_path', 'get_best_arch']


def get_save_path(*configs, prefix='runs'):
    memo = dict()
    for c in configs:
        cmemo = memo
        c = c.replace('configs/', '').replace('.py', '').split('/')
        for m in c:
            if m not in cmemo:
                cmemo[m] = dict()
            cmemo = cmemo[m]

    def get_str(m, p):
        n = len(m)
        if n > 1:
            p += '['
        for i, (k, v) in enumerate(m.items()):
            p += k
            if len(v) > 0:
                p += '.'
            p = get_str(v, p)
            if n > 1 and i < n - 1:
                p += '+'
        if n > 1:
            p += ']'
        return p

    return os.path.join(prefix, get_str(memo, ''))


def get_best_arch(arch_dir, max_iter=None):
    as_folders = [
        x for x in sorted(os.listdir(arch_dir)) if 'arch_search' in x
    ]
    as_folder = as_folders[-1]
    log_files = [
        x for x in sorted(os.listdir(os.path.join(arch_dir, as_folder)))
        if 'population_e' in x
    ]
    if not len(log_files):
        return {}, 0.0
    else:
        best_arch = {}
        best_acc = 0.0
        for log_file in log_files:
            pop_idx = log_file.split('/')[-1].split('.')[0]
            pop_idx = int(pop_idx[12:])
            if max_iter is not None and pop_idx >= max_iter:
                continue
            pop_lis = torch.load(os.path.join(arch_dir, as_folder, log_file))
            arch = [x[0] for x in pop_lis]
            acc = [x[1] for x in pop_lis]
            idx = np.argmax(acc)
            cur_best_acc = acc[idx]
            if cur_best_acc > best_acc:
                best_acc = cur_best_acc
                best_arch = arch[idx]
        return best_arch, best_acc
