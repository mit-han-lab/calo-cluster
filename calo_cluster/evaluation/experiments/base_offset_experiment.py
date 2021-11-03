import math

import numpy as np
import torch
from tqdm.auto import tqdm

from .base_experiment import BaseExperiment

from calo_cluster.models.spvcnn_offset import SPVCNNOffset

class BaseOffsetExperiment(BaseExperiment):
    def _save_predictions(self, model, datamodule, run_prediction_dir, cfg, batch_size=512, num_workers=32):
        datamodule.num_workers = num_workers
        datamodule.batch_size = batch_size
        model.cuda(0)
        model.eval()

        for split, dataloader in zip(('test', 'val', 'train'), (datamodule.test_dataloader(), datamodule.val_dataloader(), datamodule.train_dataloader())):
            if split != 'val':
                continue
            output_dir = run_prediction_dir / split
            output_dir.mkdir(exist_ok=True, parents=True)
            with torch.no_grad():
                for i, batch in tqdm(enumerate(dataloader), total=math.ceil(len(dataloader.dataset.files) / datamodule.batch_size)):
                    features = batch['features'].to(model.device)
                    inverse_map = batch['inverse_map'].F.type(torch.long)
                    subbatch_idx = features.C[..., -1]
                    subbatch_im_idx = batch['inverse_map'].C[..., -
                                                             1].to(model.device)
                    if cfg.task == 'instance':
                        coordinates = batch['coordinates'].F.to(model.device)
                        embedding = (coordinates + model(features)).cpu().numpy()
                    elif cfg.task == 'semantic':
                        labels = torch.argmax(
                            model(features), dim=1).cpu().numpy()
                    elif cfg.task == 'panoptic':
                        out_c, out_e = model(features)
                        labels = torch.argmax(out_c, dim=1).cpu().numpy()
                        coordinates = batch['coordinates'].F.to(model.device)
                        embedding = (coordinates + out_e).cpu().numpy()
                    for j in torch.unique(subbatch_idx):
                        event_path = dataloader.dataset.files[i *
                                                               datamodule.batch_size + int(j.cpu().numpy())]
                        event_name = event_path.stem
                        output_path = output_dir / event_name
                        mask = (subbatch_idx == j).cpu().numpy()
                        im_mask = (subbatch_im_idx == j).cpu().numpy()
                        if cfg.task == 'instance':
                            np.savez_compressed(
                                output_path, embedding=embedding[mask][inverse_map[im_mask]])
                        elif cfg.task == 'semantic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]])
                        elif cfg.task == 'panoptic':
                            np.savez_compressed(
                                output_path, labels=labels[mask][inverse_map[im_mask]], embedding=embedding[mask][inverse_map[im_mask]])