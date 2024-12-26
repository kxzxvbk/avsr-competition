from omegaconf import DictConfig, OmegaConf
from lightning import ModelModule


def create_model(pretrained_path='./Vox+LRS2+LRS3.ckpt', cfg=OmegaConf.load('SyncVSR/config/lrs2.yaml')):
    # Set modules and trainer
    model_module = ModelModule(cfg)
    if cfg.trainer.resume_from_checkpoint and cfg.train:
        model_module = model_module.load_from_checkpoint(pretrained_path, cfg=cfg)
        print("Loaded checkpoint from", cfg.trainer.resume_from_checkpoint)
    return model_module
