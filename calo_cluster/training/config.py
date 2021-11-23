from omegaconf import DictConfig, open_dict


def fix_task(cfg: DictConfig) -> None:
    with open_dict(cfg):
        semantic = 'semantic_criterion' in cfg
        instance = 'embed_criterion' in cfg
        if semantic and instance:
            cfg.task = 'panoptic'
        elif semantic:
            cfg.task = 'semantic'
        elif instance:
            cfg.task = 'instance'
        else:
            raise RuntimeError('semantic_criterion and/or embed_criterion must be set!')

        if instance:
            requires_semantic = ('method' not in cfg.embed_criterion) or cfg.embed_criterion.method in ['ignore', 'separate']
            cfg.requires_semantic = requires_semantic

def add_wandb_version(cfg: DictConfig, wandb_version):
    with open_dict(cfg):
        cfg.wandb.version = wandb_version