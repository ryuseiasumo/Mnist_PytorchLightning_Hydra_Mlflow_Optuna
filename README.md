# Mnist_PytorchLightning_Hydra_Mlflow_Optuna
PytorchLightning, Hydra, MLflow, Optunaの個人的な練習用に作成しました。Mnistを例に用いています。参考： [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)。

<u>__[English version](./README_EN.md)__</u>

# Version
タグに載せたバージョン毎に、以下の仕様となっています（__下線部付きバージョンの使用を推奨__）。mainブランチに上げているのはv3.1です。
* __v1.x : PytorchLightning + Hydra__
    * <u>v1.1 : Single GPUとDP(Data Parallel)、DDP(Distributed Data Parallel)が可能(推奨)。</u>
    * v1.0 : Single GPUとDDPのみ可能(非推奨)。
* __v2.x : PytorchLightning + Hydra + MLflow__
    * <u>v2.1 : Single GPUとDP、DDPが可能(推奨)。</u>
    * v2.0 : Single GPUとDDPのみ可能(非推奨)。
* __v3.x : PytorchLightning + Hydra + MLflow + Optuna__
    * <u>v3.1 : Single GPUとDP、DDPが可能(推奨)。</u>
    * v3.0 : Single GPUのみ可能(非推奨)。

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
$ python main.py



# multirunを使用する場合は、上記の代わりに、以下の例を参考に実行してください。
$ python main.py --multirun batchsize=32,64,128

# また、Hydraの設定を次のように上書きすることも可能です。
$ python main.py trainer=dp.yaml logger=csv.yaml batch_size=128

# v3.xでOptunaを用いる場合は、以下のように実行してください。
$ python main.py --multirun hparams_search=mnist_optuna.yaml
```


# Note
* singularityが利用できない場合は、requirements.txtを参考にして環境構築をしてみてください。


* multirunすると、1回の実行でlogのディレクトリが2つ作られるので気をつけてください(2022/05/20時点では原因不明、知っている方いれば教えて下さい)。
    * 具体的には、「"2022-05-19_12-45-40"と"2022-05-19_12-45-43"」など。checkpointは前者に作られます。
    * mlflowの方では、1回の実行あたり1つのディレクトリが作られます。
<!-- DDPで実行する場合、multirunするとエラーになるので気をつけてください(2022/05/17時点では原因不明、知っている方いれば教えて下さい)。 -->


* multirunの際、以下のように実行してしまうとプログラムが止まらなくなるので、trainerでのmultirunは行わないでください(2022/05/20時点では原因不明、知っている方いれば教えて下さい)。
    ```bash
    $ python main.py --multirun trainer=ddp.yaml,dp.yaml
    ```


* "src/utils/\_\_init\_\_.py"の、以下の部分（40行目辺り）は、"configs/datamodule/mnist.yaml"の内容に依存しているので、別のコードで利用する場合は注意してください。
    ```python
    # save hyper parameters of "datamodule"
    hparams["datamodule"] = {'_target_': config["datamodule"]['_target_'], 'data_dir': config["datamodule"]['data_dir'], 'batch_size': config["datamodule"]['batch_size'], 'train_val_test_split': config["datamodule"]['train_val_test_split'], 'pin_memory': config["datamodule"]['pin_memory']}
    ```
    