import os,sys
from pathlib import Path as _Path

# ---- [Mac対応] 実行ディレクトリと baseLib の場所を固定する --------------------------
# このファイルをどこから実行しても、相対パス(dataset/ , runs/ , ../base_dataset/ など)が
# dataset_train 基準になり、dataset_train の baseLib.py を import できる。
# 別の場所を使いたい場合は環境変数 WORK_DIR で上書きできる。
_here=_Path(__file__).resolve().parent
WORK_DIR=_Path(os.environ.get("WORK_DIR",_here if (_here/"baseLib.py").exists() else _here.parent/"dataset_train"))
os.chdir(WORK_DIR)
sys.path.insert(0,str(WORK_DIR))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
# ---------------------------------------------------------------------------------

import baseLib
DataFrameExLib=baseLib.pyExLib.DataFrameExLib
resultYOLOTrain=baseLib.imgLib.cocoDatasetLib.datasetInfo.resultYOLOTrain
YOLOANN=baseLib.imgLib.YOLOANN
BBimgJson=baseLib.imgLib.BBimgJson
ensembleModel=baseLib.imgLib.ensembleModel
ExtendedJSONEncoder=baseLib.IOLib.JSONLib.ExtendedJSONEncoder

from pathlib import Path
from ultralytics import YOLO
import numpy as np
import pandas as pd
import json

yolo_model_dir=Path("runs/detect/v20251125_merged_e50_yolo12s")
yolo_model_path=Path(f"{yolo_model_dir}/weights/best.pt")

# [Mac対応] dataset/ 内のリンクは Linux の絶対パスを指していて壊れているため、相対リンクで作り直した dataset_mac/ を使う
dataset_dir=Path("dataset_mac/v20251125_merged/")
dataset_yaml_path=Path(f"{dataset_dir}/data.yaml").absolute()
dataset_val_images_dir=Path(f"{dataset_dir}/images/val/")
dataset_val_images_files=list(dataset_val_images_dir.glob("*.jpg"))

# [動作確認用] 環境変数 MAX_IMAGES=50 のように指定すると、先頭からその枚数だけで試せる(未指定=全画像)
_max_images=int(os.environ.get("MAX_IMAGES","0"))
if(_max_images>0):
    dataset_val_images_files=sorted(dataset_val_images_files)[:_max_images]
print("Number of val images:",len(dataset_val_images_files))

# [再現用] 先輩の結果(out/test_cm/)を上書きしないよう出力先を分ける
out_dir=Path(f"out/test_cm_reproduce/")
out_dir.mkdir(parents=True, exist_ok=True)

model=YOLO(str(yolo_model_path))

"""
#### run val to confusion matrix (`baseLib.imgLib.cocoDatasetLib.datasetInfo.resultYOLOTrain.resultYOLOTrain.getEvaluation` like algorithm) 
val_results=model.val(
    data=str(dataset_yaml_path),
    project=str(out_dir),
    name="val",
    exist_ok=True,
)

val_evaluation=resultYOLOTrain.modelEvaluation4ConfusionMatrix(
    cm_mat=val_results.confusion_matrix.matrix,
    names=val_results.names,
    is_transpose=True,
    NEAR_ZERO=resultYOLOTrain.DEFAULT_NEAR_ZERO
)

with open(str(out_dir/"val_evaluation.json"),mode="w",encoding="utf-8") as f:
    json.dump(val_evaluation,f,indent=4)
"""
    
## run predict to calculate confusion matrix (`baseLib.imgLib.YOLOModelLib.AllImagesResultBBImgJsonCombinationsModel.getEvaluationDict` like algorithm)
predict_bb_img_results=[]
for _i,img_file in enumerate(dataset_val_images_files):
    if(_i%200==0):
        print(f"Predicting: {_i}/{len(dataset_val_images_files)}",flush=True)
    ann_file=Path(f"{dataset_dir}/labels/val/{img_file.stem}.txt")
    if(not ann_file.exists()):
        continue

    yolo_ann=YOLOANN(
        img_mode=YOLOANN.IMG_MODE_FILE,
        ann_data_arg=str(ann_file),
        arg=str(img_file),
    )
    tmp_predict_results=yolo_ann.procYOLOPredict(
        models=[model],
        mode=ensembleModel.ENSEMBLE_MODE_ONE_MODEL,
        iou_threshold=ensembleModel.DEFAULT_IOU_THRESHOLD,
        multi_class_mode=ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_ALL_CLASS_TOGETHER,

    )
    tmp_bb_img_json=tmp_predict_results[0].getResultBBimgJsonObj()
    predict_bb_img_results.append(tmp_bb_img_json)
    del yolo_ann

predict_cm=BBimgJson.BuildConfusionMatrixFromBBimgJsonList(
    predict_bb_img_results,
    iou_threshold=BBimgJson.DEFAULT_CONFUSION_MATRIX_IOU_THRESHOLD,
    include_background=True,
    background_label=BBimgJson.DEFAULT_CONFUSION_MATRIX_BACKGROUND_LABEL,
    orientation=BBimgJson.BUILD_CONFUSION_MATRIX_FROM_BB_IMG_JSON_LIST_ORIENTATION_ULTRALYTICS
)

predict_cm_np=DataFrameExLib.DataFrame2NdarraySafe(
    predict_cm,
    dtype=np.int64,
    is_fillna_zero=False,
)
predict_cm_cls_names={}
for i,cls_name in enumerate(predict_cm.index):
    predict_cm_cls_names[float(i)]=cls_name

predict_model_evaluation=resultYOLOTrain.modelEvaluation4ConfusionMatrix(
    predict_cm_np,
    names_dict=predict_cm_cls_names,
    is_transpose=True,
    NEAR_ZERO=resultYOLOTrain.DEFAULT_NEAR_ZERO
)

with open(str(out_dir/"predict_evaluation.json"),mode="w",encoding="utf-8") as f:
    json.dump(predict_model_evaluation,f,indent=4,cls=ExtendedJSONEncoder)
