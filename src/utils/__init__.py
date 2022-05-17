from typing import List

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf

@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionaly saves:
    - number of model parameters
    """

    if not trainer.logger:
        return

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["model"] = config["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # hparams["datamodule"] = config["datamodule"]
    hparams["datamodule"] = {'_target_': config["datamodule"]['_target_'], 'data_dir': config["datamodule"]['data_dir'], 'batch_size': config["datamodule"]['batch_size'], 'train_val_test_split': config["datamodule"]['train_val_test_split'], 'pin_memory': config["datamodule"]['pin_memory']}
    
    hparams["trainer"] = config["trainer"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)