# Mnist_PytorchLightning_Hydra_Mlflow_Optuna
Created for personal practice with PytorchLightning, Hydra, MLflow and Optuna, using Mnist as an example, inspired by [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template)。

## __[日本語版](./README.md)__
<br/>

# Version
The specifications are as follows for each version listed in the tags (__bold versions are recommended__). v3.1 is the one I have on the main branch.
* ### v1.x : PytorchLightning + Hydra
    * __v1.1 : Single GPU and DP (Data Parallel) and DDP (Distributed Data Parallel) are available (recommended).__
    * v1.0 : Only Single GPU and DDP are available (deprecated).
* ### v2.x : PytorchLightning + Hydra + MLflow
    * __v2.1 : Single GPU and DP and DDP are available (recommended).__
    * v2.0 : Only Single GPU and DDP are available (deprecated).
* ### v3.x : PytorchLightning + Hydra + MLflow + Optuna
    * __v3.1 : Single GPU and DP and DDP are available (recommended).__
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



# Confirmation of experimental results on a web browser by mlflow.
$ cd logs/mlflow/ # Go to the directory where the "mlruns directory" is located.
$ mlflow ui # Connect on port 5000. If you want to change the port, -p port number.
# -> http://localhost:5000/
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


* The following part of "src/utils/\_\_init\_\_.py" (around line 40) depends on the contents of "configs/datamodule/mnist.yaml", so please be careful when using it in other codes.
    ```python
    # save hyper parameters of "datamodule"
    hparams["datamodule"] = {'_target_': config["datamodule"]['_target_'], 'data_dir': config["datamodule"]['data_dir'], 'batch_size': config["datamodule"]['batch_size'], 'train_val_test_split': config["datamodule"]['train_val_test_split'], 'pin_memory': config["datamodule"]['pin_memory']}
    ```

* Please note that if you use ```$ mlflow ui``` to view the results of an experiment run on the server from a client's web browser, you must open the port to view the results. If you have set up ssh in VS Code, it is easier to run ```$ mlflow ui``` in a terminal on VS Code.

<br/>

# Reference Sites
## Pytorch Lightning
* [lightning-hydra-template (github)](https://github.com/ashleve/lightning-hydra-template)
* [Pytorch Lightning](https://www.pytorchlightning.ai/)
* [Let's study the PyTorch Lightning API](https://qiita.com/ground0state/items/c1d705ca2ee329cdfae4)
* [About DataParallel](https://pytorch-lightning.readthedocs.io/en/1.5.8/advanced/multi_gpu.html)

## Hydra
* [Hydra](https://hydra.cc/docs/intro/)
* [How to manage Python and machine learning parameters using Hydra](https://zenn.dev/kwashizzz/articles/ml-hydra-param)
* [Problems with working directory changes in Hydra](https://zenn.dev/ken7/articles/149becf3bea910)

## Mlflow
* [MLFlowLogger PytorchLightning](https://pytorch-lightning.readthedocs.io/en/stable/extensions/generated/pytorch_lightning.loggers.MLFlowLogger.html#pytorch_lightning.loggers.MLFlowLogger)
* [Notes on using MLflow](https://zenn.dev/currypurin/articles/15bd449da18807b08f89)
* [Cloud Engineer's Notebook](https://tmyoda.hatenablog.com/entry/20210422/1619085282#Runs)

## Optuna
* [Automatic Tuning of Hyperparameters with Optuna -Pytorch Lightning Edition-](https://cpp-learning.com/optuna-pytorch/#Optuna_-Pytorch_Lightning)
