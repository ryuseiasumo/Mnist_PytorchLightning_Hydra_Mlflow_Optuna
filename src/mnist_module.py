import os
from typing import Any, List

import torch
from torch.optim import Adam, Optimizer
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.simple_dense_net import SimpleDenseNet

#Pytorch Lightningモジュール
class MNISTLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module, #モデル
        optimizer_namelist: List, #optimizerの名前のリスト（追加）
        lr: float = 0.001, #学習率
        weight_decay: float = 0.0005, #重み減衰
    ):
        super().__init__()

        # 引数にバッチサイズなどのパラメータを渡すことで, self.hpparamsに登録されて参照できるようになる
        # self.save_hyperparameters(ignore=['net']) #loggerを設定している場合
        self.save_hyperparameters(logger=False, ignore=['net']) #loggerを設定していない場合

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # train, val, testで別々のマトリックスを使用
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        # valの精度が最大の場合の記録用
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        #ディクショナリに任意のテンソル(lossを含めて)を入れて返す.
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_acc(preds, targets)
        # callback時, ModelCheckpointの引数である"monitor"を"val/acc"に指定するのにも使われる
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  #　現在のepochでのVal Accuracy
        self.val_acc_best.update(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True) #現状で最大のAccuracyを出せるモデルを記録

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # エポックの終わり毎にリセット
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()


    def configure_optimizers(self) -> List[Optimizer]:
        """
        optimizers, learning-rate schedulersの設定
        """
        optimizer_list = []
        for optimizer_name in self.hparams.optimizer_namelist:
            if optimizer_name == "Adam":
                """Initialize Adam optimizer."""
                optimizer = Adam(params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
                optimizer_list.append(optimizer)
        
        return optimizer_list