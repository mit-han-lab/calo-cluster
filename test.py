import argparse
import datetime
import os
import random
import shutil

import numpy as np
import torch

from utils.comm import *


def prepare():
    from utils.common import get_save_path, get_best_arch
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    parser.add_argument('--train_single', type=str, default='False')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--local_rank', default=0)
    parser.add_argument('--votes', default=5)
    args, opts = parser.parse_known_args()
    try:
        args.local_rank = int(args.local_rank)
    except:
        args.local_rank = 0

    try:
        args.votes = int(args.votes)
    except:
        args.votes = 1

    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    #torch.distributed.init_process_group(backend='nccl',
    #        init_method='tcp://127.0.0.1:23308', rank=args.local_rank,
    #        world_size=len(gpus))
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    configs.num_votes = args.votes
    configs.local_rank = args.local_rank
    configs.global_rank = torch.distributed.get_rank()

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    configs.train.train_single = args.train_single.lower() != 'false'

    # define save path
    configs.train.save_path = get_save_path(*args.configs, prefix='runs')
    configs.train.submit = 'submit' in configs.train.save_path
    if configs.train.submit:
        configs.train.save_path = os.path.join(configs.train.save_path,
                                               'single_network')

    # override configs with args
    configs.update_from_arguments(*opts)
    if len(gpus) == 0:
        configs.device = 'cpu'
        configs.device_ids = []
    else:
        configs.device = 'cuda'
        configs.device_ids = gpus

    metrics = []
    if 'metric' in configs.train and configs.train.metric is not None:
        metrics.append(configs.train.metric)
    if 'metrics' in configs.train and configs.train.metrics is not None:
        for m in configs.train.metrics:
            if m not in metrics:
                metrics.append(m)
    configs.train.metrics = metrics
    configs.train.metric = None if len(metrics) == 0 else metrics[0]

    if 'train_single' in configs.train and configs.train.train_single is True:
        configs.train.save_path = os.path.join(configs.train.save_path,
                                               'single_network')
    else:
        configs.train.train_single = False
    save_path = configs.train.save_path
    configs.train.overwrite_checkpoint_path = args.checkpoint_path
    configs.train.checkpoint_path = os.path.join(save_path, 'best.pth.tar')
    configs.train.checkpoints_path = os.path.join(save_path, 'best',
                                                  'e{}.pth.tar')
    configs.train.best_checkpoint_path = os.path.join(configs.train.save_path,
                                                      'best.pth.tar')
    best_checkpoints_dir = os.path.join(save_path, 'best')
    configs.train.best_checkpoint_paths = {
        m: os.path.join(best_checkpoints_dir,
                        'best.{}.pth.tar'.format(m.replace('/', '.')))
        for m in configs.train.metrics
    }
    os.makedirs(os.path.dirname(configs.train.checkpoints_path), exist_ok=True)
    os.makedirs(best_checkpoints_dir, exist_ok=True)

    return configs


def main():
    from utils.common import get_save_path, get_best_arch

    configs = prepare()

    import numpy as np
    import tensorboardX
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm
    from modules.efficient_minkowski import SparseTensor

    ################################
    # Eval Kernel Functions #
    ################################
    # evaluate kernel
    def evaluate(model, loader, split='test'):
        meters = {}
        for k, meter in configs.train.meters.items():
            meters[k.format(split)] = meter()
        model.eval()

        out_lis = {}
        for k in meters:
            out_lis[k] = {}

        with torch.no_grad():
            for (locs, feats, targets), all_labels, invs in tqdm(loader,
                                                                 desc=split,
                                                                 ncols=0):
                inputs = SparseTensor(feats, coords=locs).to(configs.device)
                targets = targets.to(configs.device, non_blocking=True).long()
                outputs = model(inputs)

                if isinstance(outputs, SparseTensor):
                    outputs = outputs.F

                for idx in range(len(invs)):
                    outputs_mapped = outputs[inputs.C[:, -1] == idx][
                        invs[idx]].argmax(1)
                    targets_mapped = torch.from_numpy(
                        all_labels[idx]).long().to(configs.device)
                    for k, meter in meters.items():
                        meter.update(outputs_mapped, targets_mapped, pred=True)

        synchronize()

        for k, meter in meters.items():
            for attr in dir(meter):
                if isinstance(getattr(meter, attr), np.ndarray):
                    all_attr = accumulate_data_from_multiple_gpus(
                        [getattr(meter, attr)])
                    reduce_attr = np.zeros_like(getattr(meter, attr))
                    if all_attr is not None:
                        for x in all_attr:
                            reduce_attr += x[0]
                    setattr(meters[k], attr, reduce_attr)

        if configs.local_rank == 0:
            for k, meter in meters.items():
                meters[k] = meter.compute()
        else:
            meters = {}

        return meters

    ###########
    # Prepare #
    ###########

    if configs.device == 'cuda':
        torch.cuda.set_device(configs.local_rank)
        cudnn.benchmark = True
        if configs.get('deterministic', False):
            cudnn.deterministic = True
            cudnn.benchmark = False
    if ('seed' not in configs) or (configs.seed is None):
        configs.seed = torch.initial_seed() % (2**32 - 1)

    # all nodes should have the same seed.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(configs)

    #####################################################################
    # Initialize DataLoaders, Model, Criterion, LRScheduler & Optimizer #
    #####################################################################

    print(f'\n==> loading dataset "{configs.dataset}"')
    dataset = configs.dataset()
    loaders = {}
    samplers = {}
    for split in dataset:
        samplers[split] = DistributedSampler(
            dataset[split],
            rank=configs.global_rank,
            shuffle=(split == 'train')) if split != 'train' else None
        loaders[split] = DataLoader(
            dataset[split],
            sampler=samplers[split],
            batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model().to(configs.device)

    if configs.device == 'cuda':
        model = DistributedDataParallel(model,
                                        device_ids=[configs.local_rank],
                                        output_device=configs.local_rank,
                                        find_unused_parameters=False)

    if os.path.exists(configs.train.checkpoint_path):
        print(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path,
                                map_location='cuda:%d' % configs.local_rank)
        print(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))

        meters = checkpoint.get('meters', {})

        del checkpoint

    meters = evaluate(model, loaders['test'], split='test')
    if configs.local_rank == 0:
        print(meters)


if __name__ == '__main__':
    main()
