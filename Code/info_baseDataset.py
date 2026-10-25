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

from pathlib import Path
import baseLib

dataset_root_dir=Path("../base_dataset/baseDataset/")

dataset_version="v20260105"
bundle_zip_path=Path(f"../base_dataset/info/baseDataset_info_{dataset_version}.zip")
meta_str="baseDataset_info"

data_types=[
    "train",
    "val",
    "test"
]

img_name_tag_regex_dict={
    "A":[r"^A_img[0-9]*$",r"^A_movie_[a-zA-Z0-9#$]{11}_[0-9]*$"],
    "A_aug":[r"^A_img[0-9]*_aug_v1_[0-9]*$",r"^A_movie_[a-zA-Z0-9#$]{11}_[0-9]*_aug_v1_[0-9]*$"],
    "B":[r"^B_MVI_[0-9]*_[0-9]*$"],
    "B_aug":[r"^B_MVI_[0-9]*_[0-9]*_aug_v1_[0-9]*$"],
    "C":[r"^C_[0-9]*_[0-9]*_IMG_[0-9]*$"],
    "C_aug":[r"^C_[0-9]*_[0-9]*_IMG_[0-9]*_aug_v1_[0-9]*$"],
    "D":[r"^D_rugby_[0-9]*_[0-9]*$"],
    "D_aug":[r"^D_rugby_[0-9]*_[0-9]*_aug_v1_[0-9]*$"],
    "E":[r"^E_[0-9]{8}_CAM[0-9]+_(2K|4K)_[0-9]*__[0-9]*$"],
    "E_aug":[r"^E_[0-9]{8}_CAM[0-9]+_(2K|4K)_[0-9]*__[0-9]*_aug_v1_[0-9]*$"],
}

info=baseLib.imgLib.YOLODatasetInfo(
    dataset_root_dir=dataset_root_dir,
    data_types=data_types,
    img_name_tag_regex_dict=img_name_tag_regex_dict,
    print_log=True,
)

baseLib.IOLib.FileStore.dumpZipBundle(
    info,
    bundle_zip_path,
    meta_str,
)

restored=baseLib.IOLib.FileStore.loadZipBundle(
    bundle_zip_path,
    meta_str,
)
print(type(restored))