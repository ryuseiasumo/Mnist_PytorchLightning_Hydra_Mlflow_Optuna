# Mnist_PytorchLightning_Hydra_Mlflow_Optuna
PytorchLightning, Hydra, MLflow, Optunaの個人的な練習用に作成しました。Mnistを例に用いています。

参考： [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)。


# Version
タグに載せたバージョン毎に、以下の仕様になっています（高いバージョンの使用を推奨）。
* v1.x : PytorchLightning + Hydra
    * 1.1 : Single GPUとDP(Data Parallel)、DDP(Distributed Data Parallel)が可能。
    * 1.0 : Single GPUとDDPのみ可能(非推奨)。
* v2.x : PytorchLightning + Hydra + MLflow
    * 2.1 : Single GPUとDP、DDPが可能。
    * 2.0 : Single GPUとDDPのみ可能(非推奨)。
* v3.x : PytorchLightning + Hydra + MLflow + Optuna
    * 3.1 : Single GPUとDPが可能。
    * 3.0 : Single GPUのみ可能(非推奨)。

# 開発環境
* Ubuntu 21.04
* CUDA v11.5.50
* singularityイメージ: [nvcr.io/nvidia/pytorch:21.06-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-06.html)

# Requirement
* CUDA v11.x
* singularity-ce version 3.9.0

# Quickstart
<!-- ## singularityイメージの作成 -->
```bash
$ git clone https://github.com/ryuseiasumo/Mnist_PytorchLightning_Hydra_Mlflow_Optuna
$ cd Mnist_PytorchLightning_Hydra_Mlflow_Optuna/
$ singularity build --fakeroot pytorch_lit.sif ./def_files/pytorch_lit.def
$ singularity shell --nv pytorch_lit.sif
```


# Note
* singularityが利用できない場合は、requirements.txtを参考にしてみてください。

* multirunすると、1回の実行でlogのディレクトリが2つ作られるので気をつけてください(2022/05/20時点では原因不明、知っている方いれば教えて下さい)。
    * 具体的には、「"2022-05-19_12-45-40"と"2022-05-19_12-45-43"」など。checkpointは前者に作られます。
    * mlflowの方では、1回の実行あたり1つのディレクトリが作られます。
<!-- DDPで実行する場合、multirunするとエラーになるので気をつけてください(2022/05/17時点では原因不明、知っている方いれば教えて下さい)。 -->

* multirunの際、以下のように実行してしまうとプログラムが暴走するので、trainerでのmultirunは行わないでください(2022/05/20時点では原因不明、知っている方いれば教えて下さい)。
```bash
$ python main.py --multirun trainer=ddp.yaml,dp.yaml
```