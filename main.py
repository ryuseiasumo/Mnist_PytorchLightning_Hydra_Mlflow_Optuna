import os
import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# 環境変数があれば `.env` ファイルから読み込む
# 作業ディレクトリから始まるすべてのフォルダの `.env` を再帰的に検索
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="main.yaml")
def main(config: DictConfig):
    print("original working directry:", config.original_work_dir) # mainファイルが本来あるディレクトリ
    print("current directry:", os.getcwd()) # hydraを用いるとディレクトリが移動する
    
    # print("---View parameter configuration---")
    # print(OmegaConf.to_yaml(config))  #hydraの設定を表示
    # print("----------------------------------")
    
    if config.do_train: #学習を行う場合
        # trainの処理
        from src.train import train
            
        # Train model
        return train(config)



if __name__ == "__main__":
    #python main.py do_train=False #コマンドライン引数でhydraで与えた変数の値の変更も可能
    #python main.py --multirun model=mnist, cnn, ... #いろんな条件でやりたい時
    #python main.py --multirun #Optuna利用時は, "--multirun"が必須
    
    main()
