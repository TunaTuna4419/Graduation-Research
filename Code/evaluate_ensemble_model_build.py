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

import os
import baseLib
from pathlib import Path
import glob
import re

def autoNameFunc(model_dir_obj:Path):
    model_dir_obj_name=model_dir_obj.name
    return f"{model_dir_obj_name}"

model_obj_list=baseLib.imgLib.YOLOModelLib.readYOLOModelList.createGlobReadYOLOModelList(
    [
        #"runs/detect/*"
        "runs/detect/v20251108_merged_e50_yolov8s/",
        "runs/detect/v20251125_merged_e50_yolo12n/",
        "runs/detect/v20251125_merged_e50_yolo12s/",
        "runs/detect/v20251125_merged_e50_yolov8n/",
        "runs/detect/v20251125_merged_e50_yolov8s/",
        "runs/detect/v20251125_merged_or_e50y_yolo12x/",
        "runs/detect/v20251125_merged_or_e100y_yolo12s/",
        "runs/detect/v20251125_merged_or_e100y_yolo12x/",
    ],
    recursive=True,
    is_auto_name=True,
    auto_name_func=autoNameFunc
)

predict_config=[
    {
        "combinations_r":1,
        "config_for_each_combination":[
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_ONE_MODEL,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            }
        ],
    },
    {
        "combinations_r":3,
        "config_for_each_combination":[
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_NMS,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_WBF,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_MAJORITY_RULE_NMS,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
        ]
    },
    {
        "combinations_r":5,
        "config_for_each_combination":[
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_NMS,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_WBF,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
            {
                "mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MODE_MAJORITY_RULE_NMS,
                "iou_threshold":baseLib.imgLib.ensembleModel.DEFAULT_IOU_THRESHOLD,
                "multi_class_mode":baseLib.imgLib.ensembleModel.ENSEMBLE_MULTI_CLASS_MODE_EACH_CLASS,
                "ensemble_args":{},
                "bb_img_json_mode":True,
                "proc_ensemble_model_name_func":baseLib.imgLib.YOLOModelLib.YOLOModelPredict.DEFAULT_PROC_ENSEMBLE_MODEL_NAME
            },
        ]
    }
]

# [Mac対応] dataset/ 内のリンクは Linux の絶対パスを指していて壊れているため、相対リンクで作り直した dataset_mac/ を使う
img_dirs=[
    #Path("dataset_mac/v20251125_merged/images/val/"),
    Path("dataset_mac/v20251125_merged/images/test/"),
]
label_dirs=[
    #Path("dataset_mac/v20251125_merged/labels/val/"),
    Path("dataset_mac/v20251125_merged/labels/test/"),
]
out_save_dirs=[
    #Path("out/evaluate_ensemble_model/20251208/val/"),
    Path("out/evaluate_ensemble_model/20251208/test/"),
]

for img_dir,label_dir,out_save_dir in zip(img_dirs,label_dirs,out_save_dirs):
    ann_files=[]
    name_re_list=[
        r"^A_img[0-9]*$",
        r"^A_movie_[a-zA-Z0-9#$]{11}_[0-9]*$",
        r"^B_MVI_[0-9]*_[0-9]*$",
        r"^C_[0-9]*_[0-9]*_IMG_[0-9]*$",
        r"^D_rugby_[0-9]*_[0-9]*$",
    ]
    for ann_file in list(label_dir.rglob("*.txt")):
        ann_name=ann_file.stem
        matched_flag=False
        for name_re in name_re_list:
            if(re.match(name_re,ann_name)):
                matched_flag=True
                break
        if(matched_flag):
            ann_files.append(str(ann_file))

    all_imgs_results=baseLib.imgLib.YOLOModelLib.AllProcPredictCombinationsModel(
        ann_files,
        glob.glob(f"{img_dir}/*.jpg"),
        model_obj_list,
        predict_config,

        batch_size=25,
        # [Mac対応] gpu_flag は CUDA autocast を使うかどうかのフラグ。CUDA が無い環境では False にする
        gpu_flag=baseLib.torch.cuda.is_available(),
        save_bb_img_json_each_img=True,
        save_bb_img_json_each_img_out_dir_format=f"{out_save_dir}/data/",
            save_bb_img_json_each_img_img=False,

        cache_mode=False
    )

    """
    all_imgs_results.saveAllData(
        out_save_dir,
        save_bb_img_json=True,
        save_all_iou_df=False,
        save_evaluation_json=False,
        save_evaluation_excel=False,
    )
    """