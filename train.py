import argparse
import copy
import os
import random
import shutil

import hydra
import torch
from omegaconf import DictConfig

from utils.comm import *

def prepare():
    from utils.common import get_save_path
    from utils.config import configs
    from utils.device import set_cuda_visible_devices

    # since PyTorch jams device selection, we have to parse args before import torch (issue #26790)
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    parser.add_argument('--devices', default=None)
    parser.add_argument('--local_rank', default=0)
    args, opts = parser.parse_known_args()
    try:
        args.local_rank = int(args.local_rank)
    except:
        args.local_rank = 0

    if args.devices is not None and args.devices != 'cpu':
        gpus = set_cuda_visible_devices(args.devices)
    else:
        gpus = []

    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    configs.local_rank = args.local_rank
    configs.global_rank = torch.distributed.get_rank()

    print(f'==> loading configs from {args.configs}')
    configs.update_from_modules(*args.configs)
    # define save path
    configs.train.save_path = get_save_path(*args.configs, prefix='runs')

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

    save_path = configs.train.save_path
    configs.train.checkpoint_path = os.path.join(save_path, 'latest.pth.tar')
    configs.train.checkpoints_path = os.path.join(save_path, 'latest',
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

@hydra.main(config_path="configs", config_name="config")
def hydra_main(cfg: DictConfig) -> None:
    print(cfg.pretty())
    model = hydra.utils.instantiate(cfg.model)

def main(cfg: DictConfig) -> None:
    configs = prepare()

    from utils.common import get_best_arch
    import numpy as np
    import tensorboardX
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel
    from tqdm import tqdm
    from modules.efficient_minkowski import SparseTensor

    ################################
    # Train / Eval Kernel Function #
    ################################

    # train kernel
    def train(model, loader, criterion, optimizer, scheduler, current_step,
              writer):
        if scheduler is not None:
            print(scheduler.get_lr())
        model.train()

        for (locs, feats, targets), all_labels, invs in tqdm(loader,
                                                             desc='train',
                                                             ncols=0):
            inputs = SparseTensor(feats, coords=locs).to(configs.device)
            targets = targets.to(configs.device, non_blocking=True).long()
            optimizer.zero_grad()

            outputs = model(inputs)

            if isinstance(outputs, SparseTensor):
                outputs = outputs.F

            loss = criterion(outputs,
                             targets) + 0 * sum(p.sum()
                                                for p in model.parameters())

            if configs.local_rank == 0:
                writer.add_scalar('loss/train', loss.item(), current_step)
            current_step += 1
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

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
    seed = configs.seed + configs.global_rank * configs.data.num_workers * configs.train.num_epochs
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
        samplers[split] = DistributedSampler(dataset[split],
                                             rank=configs.global_rank,
                                             shuffle=(split == 'train'))
        loaders[split] = DataLoader(
            dataset[split],
            sampler=samplers[split],
            batch_size=configs.train.batch_size,
            num_workers=configs.data.num_workers,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn,
            worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))

    print(f'\n==> creating model "{configs.model}"')
    model = configs.model()

    model = model.to(configs.device)

    if configs.device == 'cuda':
        model = DistributedDataParallel(model,
                                        device_ids=[configs.local_rank],
                                        output_device=configs.local_rank,
                                        find_unused_parameters=False)

    criterion = configs.train.criterion().to(configs.device)
    optimizer = configs.train.optimizer(model.parameters())

    last_epoch, best_metrics = -1, {m: None for m in configs.train.metrics}

    if os.path.exists(configs.train.checkpoint_path):
        print(f'==> loading checkpoint "{configs.train.checkpoint_path}"')
        checkpoint = torch.load(configs.train.checkpoint_path,
                                map_location='cuda:%d' % configs.local_rank)
        print(' => loading model')
        model.load_state_dict(checkpoint.pop('model'))
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            print(' => loading optimizer')
            optimizer.load_state_dict(checkpoint.pop('optimizer'))
        last_epoch = checkpoint.get('epoch', last_epoch)
        meters = checkpoint.get('meters', {})
        for m in configs.train.metrics:
            best_metrics[m] = meters.get(m + '_best', best_metrics[m])
        del checkpoint

    if 'scheduler' in configs.train and configs.train.scheduler is not None:
        world_size = get_world_size()
        dataset_size = len(dataset['train'])
        iter_per_epoch = (dataset_size + configs.train.batch_size * world_size
                          - 1) // (configs.train.batch_size * world_size)
        if last_epoch > -1:
            configs.train.scheduler.last_epoch = (last_epoch +
                                                  1) * iter_per_epoch
        else:
            configs.train.scheduler.last_epoch = -1
        print(f'==> creating scheduler "{configs.train.scheduler}"')
        scheduler = configs.train.scheduler(optimizer)
    else:
        scheduler = None

    ############
    # Training #
    ############

    if last_epoch >= configs.train.num_epochs:
        meters = dict()
        for split, loader in loaders.items():
            if split != 'train':
                meters.update(evaluate(model, loader=loader, split=split))
        for k, meter in meters.items():
            print(f'[{k}] = {meter:2f}')
        return

    with tensorboardX.SummaryWriter(configs.train.save_path) as writer:
        for current_epoch in range(last_epoch + 1, configs.train.num_epochs):
            synchronize()

            loaders['train'].worker_init_fn = lambda worker_id: np.random.seed(
                seed + current_epoch * configs.data.num_workers + worker_id)
            samplers['train'].set_epoch(current_epoch)

            steps_per_epoch = (len(dataset['train']) +
                               configs.train.batch_size - 1) // (
                                   configs.train.batch_size * get_world_size())

            current_step = current_epoch * steps_per_epoch

            # train
            print(
                f'\n==> training epoch {current_epoch}/{configs.train.num_epochs}'
            )
            train(model,
                  loader=loaders['train'],
                  criterion=criterion,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  current_step=current_step,
                  writer=writer)
            current_step += steps_per_epoch

            meters = dict()

            # evaluate
            for split, loader in loaders.items():
                if split != 'train':
                    meters.update(evaluate(model, loader=loader, split=split))

            if configs.global_rank != 0:
                continue

            # check whether it is the best
            best = {m: False for m in configs.train.metrics}
            for m in configs.train.metrics:
                if best_metrics[m] is None or best_metrics[m] < meters[m]:
                    best_metrics[m], best[m] = meters[m], True
                meters[m + '_best'] = best_metrics[m]

            # log in tensorboard
            for k, meter in meters.items():
                print(f'[{k}] = {meter:2f}')
                if configs.local_rank == 0:
                    writer.add_scalar(k, meter, current_step)

            # save checkpoint
            torch.save(
                {
                    'epoch': current_epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'meters': meters,
                    'configs': configs,
                }, configs.train.checkpoint_path)
            shutil.copyfile(
                configs.train.checkpoint_path,
                configs.train.checkpoints_path.format(current_epoch))

            for m in configs.train.metrics:
                if best[m]:
                    shutil.copyfile(configs.train.checkpoint_path,
                                    configs.train.best_checkpoint_paths[m])
            if best.get(configs.train.metric, False):
                shutil.copyfile(configs.train.checkpoint_path,
                                configs.train.best_checkpoint_path)
            print(f'[save_path] = {configs.train.save_path}')


if __name__ == '__main__':
    hydra_main() # pylint: disable=no-value-for-parameter
