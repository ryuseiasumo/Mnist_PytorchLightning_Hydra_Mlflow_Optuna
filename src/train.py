
import os
from typing import List, Optional

import hydra
from omegaconf import DictConfig

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from src import utils



def train(config: DictConfig) -> Optional[float]:
    """
    Args:
        config (DictConfig): Hydraで設定したconfigを渡す

    Returns:
        Optional[float]: hyperparameter最適化の, Metric score
    """

    # 並列に使用するcpuのユニット数
    #最大数利用する場合 → RuntimeError: Too many open files.となることがある →  torch.multiprocessing.set_sharing_strategy('file_system')で解決
    num_workers = os.cpu_count()
    
    # seed値の設定
    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # ckpt path を相対パスから絶対パスに変換
    ckpt_path = config.trainer.get("resume_from_checkpoint")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.trainer.resume_from_checkpoint = os.path.join(
            hydra.utils.get_original_cwd(), ckpt_path
        )
    
    
    # lightning datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule, num_workers=num_workers)

    # Pytorch model
    model: torch.nn.Module = hydra.utils.instantiate(config.model)
    
    # PytorchLightning model(lightningモジュール)
    lit_model: LightningModule = hydra.utils.instantiate(config.lightning_module, net=model)
    
    # lightning callbacks(ModelCheckpoint, EarlyStopping)
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    # lightning loggers(defaultはmlflow)
    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    # lightning trainer
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, #_convert_="partial"
    )

    # configに設定したパラメータをloggersに送る.
    utils.log_hyperparameters(
        config=config,
        model=lit_model,
        datamodule=datamodule,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )


    # Train the model
    trainer.fit(model=lit_model, datamodule=datamodule)
    best_model_path = trainer.checkpoint_callback.best_model_path

    # hyperparameter最適化の結果（metric score）を取得(Optunaで利用)
    optimized_metric = config.get("optimized_metric")
    if optimized_metric and optimized_metric not in trainer.callback_metrics:
        raise Exception(
            "Metric for hyperparameter optimization not found! "
            "Make sure the `optimized_metric` in `hparams_search` config is correct!"
        )
    score = trainer.callback_metrics.get(optimized_metric)



    # Test the model
    if config.get("do_test"):
        ckpt_path = "best"
        if not config.get("do_train") or config.trainer.get("fast_dev_run"):
            ckpt_path = None
            
        if trainer.global_rank == 0: #最後のプロセス
            # テスト用のtrainerを用意(Single GPU)
            test_trainer = Trainer(gpus=1, logger=logger, max_epochs=1) #trainerと別の場所にlogが取られてしまうため注意
            test_trainer.test(model=lit_model, datamodule=datamodule, ckpt_path=best_model_path)


    # 最もVal Accが高い, モデルのパスを出力
    print(f"Best model ckpt at {best_model_path}")
    

    return score