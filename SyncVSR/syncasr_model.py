from omegaconf import DictConfig, OmegaConf
from .lightning import ModelModule


def create_model(pretrained_path='./Vox+LRS2+LRS3.ckpt', cfg=OmegaConf.load('SyncVSR/config/lrs2.yaml')):
    cfg.ckpt_path = pretrained_path
    model_module = ModelModule.load_from_checkpoint(pretrained_path, cfg=cfg, strict=False)
    return model_module
