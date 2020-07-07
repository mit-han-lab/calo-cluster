import torch.nn as nn
import torch.optim as optim

from datasets.hgcal import HGCAL
from meters import Meter
from utils.config import Config, configs

configs.data = Config()
configs.data.num_workers = 8
configs.data.num_classes = 4

configs.dataset = Config(HGCAL)
configs.dataset.root = '/dataset/hgcal/'
configs.dataset.num_points = 80000
configs.dataset.voxel_size = 1.0

configs.train = Config()
configs.train.criterion = Config(nn.CrossEntropyLoss)
configs.train.criterion.ignore_index = 255

configs.train.meters = Config()
configs.train.meters['acc/iou_{}'] = Config(
    Meter, reduction='iou', num_classes=configs.data.num_classes)
configs.train.meters['acc/class_{}'] = Config(
    Meter, reduction='class', num_classes=configs.data.num_classes)

configs.train.metric = 'acc/iou_test'
