import faulthandler, signal
faulthandler.register(signal.SIGUSR1)

import os,sys
from pathlib import Path

# ---- [Mac対応] 実行ディレクトリと baseLib の場所を固定する --------------------------
# このファイルをどこから実行しても、相対パス(dataset/ , runs/ など)が dataset_train 基準になる。
# 別の場所を使いたい場合は環境変数 WORK_DIR で上書きできる。
_here=Path(__file__).resolve().parent
WORK_DIR=Path(os.environ.get("WORK_DIR",_here if (_here/"baseLib.py").exists() else _here.parent/"dataset_train"))
os.chdir(WORK_DIR)
sys.path.insert(0,str(WORK_DIR))

# MPS 未対応の演算が出たときに CPU へ自動フォールバックさせる(torch import 前に設定する)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")

import torch
from ultralytics import YOLO
import json

print("CUDA available:",torch.cuda.is_available())
print("MPS available:",torch.backends.mps.is_available())
if(torch.cuda.is_available()):
    print("cudnn version:",torch.backends.cudnn.version())
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

# ---- [Mac対応] 学習に使うデバイスの自動選択 ---------------------------------------
# 環境変数 DEVICE ("0" / "mps" / "cpu") で明示指定も可能
def selectDevice():
    env_device=os.environ.get("DEVICE")
    if(env_device):
        return env_device
    if(torch.cuda.is_available()):
        return 0
    if(torch.backends.mps.is_available()):
        return "mps"
    return "cpu"

DEVICE=selectDevice()
print("Train device:",DEVICE)

# 環境変数で切り替え可能。デフォルトは先輩の結果が残っている v20251125_merged / yolo12s
BASE_MODEL_NAME=os.environ.get("BASE_MODEL_NAME","yolo12s")
#BASE_MODEL_NAME="yolo26s"

DATASET_NAME=os.environ.get("DATASET_NAME","v20251125_merged")
#DATASET_NAME="v20260105_merged"

# [Mac対応] dataset/ 内のシンボリックリンクは Linux の絶対パス(/home/e205767/...)を指していて Mac では壊れているため、
# リンクを相対パスで作り直した dataset_mac/ を使う(元の dataset/ は変更していない)
DATASET_ROOT=os.environ.get("DATASET_ROOT","dataset_mac")
DATASET_DIR=Path(f"{DATASET_ROOT}/{DATASET_NAME}/")
DATASET_DATA_PATH=Path(f"{DATASET_DIR}/data.yaml")

EPOCHS=int(os.environ.get("EPOCHS","50"))
#FREEZE=10

# batch=-1 (AutoBatch) は CUDA 専用。CUDA 以外では固定値を使う
BATCH=int(os.environ.get("BATCH","-1" if torch.cuda.is_available() else "16"))

CONFIG_NAME=f"e{EPOCHS}"

# [再現用] 先輩の結果(runs/detect/)を上書き・衝突させないよう、出力先を runs/reproduce/ に分ける
PROJECT_DIR=Path(os.environ.get("PROJECT_DIR","runs/reproduce/"))
TRAIN_NAME=f"{DATASET_NAME}_{CONFIG_NAME}_{BASE_MODEL_NAME}"
RESUME_MODE=os.environ.get("RESUME_MODE","0")=="1"

SETTINGS={
    "epochs": EPOCHS,
    "data": str(DATASET_DATA_PATH),
    "batch": BATCH,
    "workers": 0,
    "imgsz": 640,
    "device": DEVICE,

    "lr0": 0.01,
    "lrf": 0.01,
    
    #"mosaic":1.0,
    #"mixup": 0.5,
    #"hsv_h":0.1,
    #"hsv_s":0.8,
    #"hsv_v":0.5,
    #"flipud":0.0,
    #"fliplr":0.5,
    #"degrees":5,
    #"translate":0.2,
    #"scale":0.75,
    #"shear":1.0,
    #"perspective":0.0005,
    
    #"freeze":FREEZE,
    #"close_mosaic":5,
    "project": str(PROJECT_DIR),
    "name": TRAIN_NAME,
    "amp": False,
    "deterministic": False,
}

train_dir=Path(f"{PROJECT_DIR}/{TRAIN_NAME}")

model=None
if(RESUME_MODE):
    resume_model_path=Path(train_dir/"weights/last.pt")
    if(not resume_model_path.exists()):
        raise FileNotFoundError(f"Cannot find model weights to resume from: {str(resume_model_path)}")

    model=YOLO(str(resume_model_path))
else:
    if(train_dir.exists()):
        raise FileExistsError(f"Training directory already exists: {str(train_dir)}. To resume training, set RESUME_MODE=True.")
    
    model=YOLO(BASE_MODEL_NAME)
    train_dir.mkdir(parents=True,exist_ok=True)

    settings_path=train_dir/"settings.json"
    save_setting_data={
        "base_model_name": BASE_MODEL_NAME,
        "train_dir": str(train_dir),
    }
    save_setting_data.update(SETTINGS)
    
    with open(settings_path,mode="w",encoding="utf-8") as f:
        json.dump(save_setting_data,f,indent=4,ensure_ascii=False)
    del save_setting_data

    SETTINGS["exist_ok"]=True

# ---- [補完] 元ファイルはここで途中終了していたため、役割から学習の実行部分を補った ----------
# 新規学習: SETTINGS をそのまま渡して学習
# 再開学習: last.pt から読み込んだモデルに resume=True を渡す(ハイパーパラメータは保存済みの値が使われる)
if(RESUME_MODE):
    results=model.train(resume=True)
else:
    results=model.train(**SETTINGS)

print("Training finished.")
print("Result directory:",train_dir)
