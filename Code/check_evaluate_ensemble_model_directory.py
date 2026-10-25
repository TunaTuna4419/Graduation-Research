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

dir=Path("out/evaluate_ensemble_model/20251201/data/")

counter_dict={}

dir_list=list(dir.glob("*/"))
for img_dir_path in dir_list:
    img_dir_files=list(img_dir_path.glob("*.json"))
    file_num=len(img_dir_files)
    file_num_key=f"{file_num}"
    
    tmp=counter_dict.get(file_num_key,{"count":0,"img_dirs":[]})
    tmp["count"]+=1
    tmp["img_dirs"].append(str(img_dir_path))
    counter_dict[file_num_key]=tmp

for counter_key in counter_dict.keys():
    print("File Num:",counter_key,"Count:",counter_dict[counter_key]["count"])
    print("Example Dirs:",counter_dict[counter_key]["img_dirs"][0])
    print()
