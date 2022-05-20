# Mnist_PytorchLightning_Hydra_Mlflow_Optuna
Created for personal practice with PytorchLightning, Hydra, MLflow and Optuna, using Mnist as an example, inspired by [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)。

<u>__[日本語版](./README.md)__</u>


# Version
The specifications are as follows for each version listed in the tags (__underlined versions are recommended__). v3.1 is the one I have on the main branch.
* __v1.x : PytorchLightning + Hydra__
    * <u>v1.1 : Single GPU and DP (Data Parallel) and DDP (Distributed Data Parallel) are available (recommended).</u>
    * v1.0 : Only Single GPU and DDP are available (deprecated).
* __v2.x : PytorchLightning + Hydra + MLflow__
    * <u>v2.1 : Single GPU and DP and DDP are available (recommended).</u>
    * v2.0 : Only Single GPU and DDP are available (deprecated).
* __v3.x : PytorchLightning + Hydra + MLflow + Optuna__
    * <u>v3.1 : Single GPU and DP and DDP are available (recommended).</u>
    * v3.0 : Only Single GPU is available (deprecated).

# Development environment
* Ubuntu 21.04
* CUDA v11.5.50
* Singularity image: [nvcr.io/nvidia/pytorch:21.06-py3](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_21-06.html)

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



# If you want to use multirun, please refer to the following example and execute it instead of the above.
$ python main.py --multirun batchsize=32,64,128

# It is also possible to override the Hydra settings as follows
$ python main.py trainer=dp.yaml logger=csv.yaml batch_size=128

# To use Optuna in v3.x, execute as follows
$ python main.py --multirun hparams_search=mnist_optuna.yaml
```


# Note
* If singularity is not available, please refer to requirements.txt to build your environment.


* Please note that multirun will create two log directories in one run (unknown cause as of 2022/05/20, if you know the cause, please let me know).
    * For example, "2022-05-19_12-45-40" and "2022-05-19_12-45-43". checkpoint is created for the former.
    * For mlflow, one directory is created per run.
<!-- DDPで実行する場合、multirunするとエラーになるので気をつけてください(2022/05/17時点では原因不明、知っている方いれば教えて下さい)。 -->


* When multirun, do not do multirun with trainer because the program will run out of control if you do the following (cause unknown as of 2022/05/20, if you know the cause, please let me know).
    ```bash
    $ python main.py --multirun trainer=ddp.yaml,dp.yaml
    ```


* The following part of "src/utils/\_init\_\_.py" (around line 40) depends on the contents of "configs/datamodule/mnist.yaml", so please be careful when using it in other codes.
    ```python
    # save hyper parameters of "datamodule"
    hparams["datamodule"] = {'_target_': config["datamodule"]['_target_'], 'data_dir': config["datamodule"]['data_dir'], 'batch_size': config["datamodule"]['batch_size'], 'train_val_test_split': config["datamodule"]['train_val_test_split'], 'pin_memory': config["datamodule"]['pin_memory']}
    ```