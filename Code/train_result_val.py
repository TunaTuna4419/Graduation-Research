import faulthandler, signal
faulthandler.register(signal.SIGUSR1)

import os,sys
from pathlib import Path

# ---- [Mac対応] 実行ディレクトリと baseLib の場所を固定する --------------------------
_here=Path(__file__).resolve().parent
WORK_DIR=Path(os.environ.get("WORK_DIR",_here if (_here/"baseLib.py").exists() else _here.parent/"dataset_train"))
os.chdir(WORK_DIR)
sys.path.insert(0,str(WORK_DIR))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")

import torch
import baseLib

print("CUDA available:",torch.cuda.is_available())
print("MPS available:",torch.backends.mps.is_available())
if(torch.cuda.is_available()):
    print("cudnn version:",torch.backends.cudnn.version())
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

# train.py と同じ環境変数・同じデフォルト値を使う
BASE_MODEL_NAME=os.environ.get("BASE_MODEL_NAME","yolo12s")

DATASET_NAME=os.environ.get("DATASET_NAME","v20251125_merged")

# [Mac対応] 壊れたシンボリックリンクの代わりに、相対リンクで作り直した dataset_mac/ を使う
DATASET_ROOT=os.environ.get("DATASET_ROOT","dataset_mac")
DATASET_DIR=Path(f"{DATASET_ROOT}/{DATASET_NAME}/")
DATASET_DATA_PATH=Path(f"{DATASET_DIR}/data.yaml")

EPOCHS=int(os.environ.get("EPOCHS","50"))

CONFIG_NAME=f"e{EPOCHS}"

# [修正] モデルが実際に保存されている runs/detect/ をデフォルトに変更
PROJECT_DIR=Path(os.environ.get("PROJECT_DIR","runs/detect/"))
TRAIN_NAME=f"{DATASET_NAME}_{CONFIG_NAME}_{BASE_MODEL_NAME}"

yolo_model_dir=f"{PROJECT_DIR}/{TRAIN_NAME}/"

r=baseLib.imgLib.cocoDatasetLib.datasetInfo.resultYOLOTrain(
    mode=baseLib.imgLib.cocoDatasetLib.datasetInfo.resultYOLOTrain.RESULT_YOLO_TRAIN_EVALUATION_MODE_MODEL_DIRECTORY,
    yolo_model_dir=yolo_model_dir,
    dataset_data_yaml_path=DATASET_DATA_PATH
)
r.saveEvaluationJson()

r_df=r.getEvaluationsDataFrame(
    drop_list=[
        # [修正] dorp_args -> drop_args のタイポを修正
        {"drop_key":"class_name","drop_args":{"axis":0}},
        {"drop_key":"TP","drop_args":{"axis":0}},
        {"drop_key":"FP","drop_args":{"axis":0}},
        {"drop_key":"FN","drop_args":{"axis":0}},
        {"drop_key":"TN","drop_args":{"axis":0}},
    ]
).T

# 先輩の結果と同じ形式(クラス名 x precision/recall/f1)で保存する
r_df.to_csv(f"{r.getResultsValSaveDir()}/class_evaluation.csv")
print(r_df)
print("Saved to:",r.getResultsValSaveDir())