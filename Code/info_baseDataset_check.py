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
from pathlib import Path

dataset_version="v20260105"
bundle_zip_path=Path(f"../base_dataset/info/baseDataset_info_{dataset_version}.zip")
meta_str="baseDataset_info"

info=baseLib.IOLib.FileStore.loadZipBundle(
    bundle_zip_path,
    meta_str,
)

if(not isinstance(info,baseLib.imgLib.YOLODatasetInfo)):
    raise ValueError("Loaded object is not YOLODatasetInfo.")

all_counter=0
invalid_counter=0
for img_idx,info_i in enumerate(info.dataInfoGenerator()):
    invalid_flag=False

    if((info_i["img_is_duplication"]) and  (not info_i["img_is_valid"])):
        print(f"Invalid : {info_i['img_name']}")
        invalid_flag=True

    img_file_path=Path(info_i["img_file_path"])
    label_file_path=Path(info_i["label_file_path"])
    if(not img_file_path.exists()):
        print(f"Image file not found: {img_file_path}")
        invalid_flag=True
    if(not label_file_path.exists()):
        print(f"Label file not found: {label_file_path}")
        invalid_flag=True
    
    if(invalid_flag):
        invalid_counter+=1
    all_counter+=1
print(f"Total images: {all_counter}")
print(f"Invalid images: {invalid_counter}")