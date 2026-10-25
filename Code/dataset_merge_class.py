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
import os

dataset_version="v20251108"
bundle_zip_path=Path(f"../base_dataset/info/baseDataset_info_{dataset_version}.zip")
meta_str="baseDataset_info"

info=baseLib.IOLib.FileStore.loadZipBundle(
    bundle_zip_path,
    meta_str,
)

if(not isinstance(info,baseLib.imgLib.YOLODatasetInfo)):
    raise ValueError("Loaded object is not YOLODatasetInfo.")

dataset_root_dir_label=Path(f"{str(info.dataset_root_dir)}/labels/")
merged_dataset_dir_label=Path(f"{str(info.dataset_root_dir)}/merged_labels/")

others_class_num=info.class_names.index("others")

l=set()
counter=0
for img_idx,info_i in enumerate(info.filteredDataInfoGenerator(
    is_valid=True,
    is_not_duplication=True,
)):
    img_name=info_i["img_name"]
    img_path=Path(info_i["img_file_path"])
    label_file_path=Path(info_i["label_file_path"])
    print(img_name)

    new_label_path=baseLib.IOLib.changeDirectoryPath(label_file_path,dataset_root_dir_label,merged_dataset_dir_label)
    new_label_path.parent.mkdir(parents=True,exist_ok=True)

    with open(label_file_path,mode="r") as f:
        with open(new_label_path,mode="w") as f_new:
            for line in f.readlines():
                split_line=line.strip().split(" ")
                if(len(split_line)==0):
                    continue
                before_class_num=int(split_line[0])
                new_class_num=None

                if(before_class_num!=others_class_num):
                    new_class_num=0 # abnormalityクラス
                else:
                    new_class_num=1  # normalityクラス

                new_line=" ".join([str(new_class_num)]+split_line[1:])+" \n"
                f_new.write(new_line)

    counter+=1

print(f"Total copied items: {counter}")
