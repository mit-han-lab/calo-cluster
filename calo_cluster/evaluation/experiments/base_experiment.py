import importlib
from pathlib import Path

import hydra
import yaml
from omegaconf import OmegaConf

from calo_cluster.training.config import fix_task


class BaseExperiment():
    def __init__(self, wandb_version, ckpt_name=None, num_workers=8, batch_size=128, aux=None):
        self.wandb_version = wandb_version
            
        self.run_path = self.get_run_path(self.wandb_version)
        cfg_path = self.run_path / 'files' / '.hydra' / 'config.yaml'
        self.cfg = OmegaConf.load(cfg_path)
        fix_task(self.cfg)
        self.ckpt_path, self.ckpt_name = self._get_ckpt(
            self.cfg, ckpt_name)
        self.model = self.load_model(str(self.ckpt_path))

        plots_dir = Path(self.cfg.plots_dir)
        self.plots_dir = plots_dir / self.wandb_version / self.ckpt_name

        self.datamodule = hydra.utils.instantiate(self.cfg.dataset, num_workers=num_workers, batch_size=batch_size, aux=aux)
        self.datamodule.prepare_data()
        self.datamodule.setup(stage=None)

        self.num_classes = self.datamodule.num_classes

        self.plots_dir.mkdir(exist_ok=True, parents=True)

        self.task = self.cfg.task

    def _get_ckpt(self, cfg, ckpt_name):
        ckpt_dir = Path(cfg.outputs_dir) / cfg.wandb.project / \
            cfg.wandb.version / 'checkpoints'
        if ckpt_name is None:
            ckpt_path = ckpt_dir / 'last.ckpt'
            if not ckpt_path.exists():
                ckpt_path = [p for p in sorted(ckpt_dir.glob('*.ckpt'))][-1]
            print(f'no checkpoint name given, using {ckpt_path}.')
        else:
            ckpt_path = ckpt_dir / ckpt_name
            if not ckpt_path.exists():
                raise RuntimeError(f'No checkpoint found at {ckpt_path}!')
        ckpt_name = ckpt_path.stem

        return ckpt_path, ckpt_name

    def get_run_path(self, wandb_version):
        config_path = Path(__file__).parent.parent.parent.parent / \
            'train_configs' / 'config.yaml'
        with config_path.open('r') as f:
            config = yaml.load(f, Loader=yaml.CLoader)
            outputs_dir = Path(config['outputs_dir'])
        wandb_dir = outputs_dir / 'wandb'
        for run_dir in wandb_dir.iterdir():
            if wandb_version in run_dir.stem:
                print(f'run_dir = {run_dir}')
                return run_dir
        raise RuntimeError(
            f'run with wandb_version={wandb_version} not found; is {wandb_dir} correct?')

    def load_model(self, ckpt_path):
        model_str = self.cfg.model.target._target_
        comps = model_str.split('.')
        module = importlib.import_module('.'.join(comps[:-1]))
        cls = getattr(module, comps[-1])
        return cls.load_from_checkpoint(str(ckpt_path))