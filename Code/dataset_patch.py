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
import math
import json
import tempfile
import zipfile

patch_str="patch_20260105"
patch_zip=f"../base_dataset/{patch_str}.zip"

new_dataset_dir=Path("../base_dataset/baseDataset/")

data_expanding_img_reshape_log_list = []

# 1) ボックスブラー × ガウシアンノイズ（ごま塩イメージの荒れも代替）
for k in [1, 3, 5]:  # 1は実質無変更、3/5で軽・中ブラー
    for sigma in [15, 25, 35]:
        data_expanding_img_reshape_log_list.append([
            {
                "mode": "ImgTransFuncRun",
                "args": {
                    "func_str": "cv2.blur",
                    "args_list": [[k, k]],
                    "args_dict": {},
                    "result_iterate": True,
                    "iterate_key": 0,
                    "stack_index": -1
                }
            },
            {
                "mode": "ImgTransFuncRun",
                "args": {
                    "func_str": "baseLib.imgLib.addGaussianNoiseImg",
                    "args_list": [],
                    "args_dict": {"mu": 0, "sigma": sigma},
                    "result_iterate": True,
                    "iterate_key": 0,
                    "stack_index": -1
                }
            }
        ])

# 1b) メディアンブラー（点ノイズ耐性）
for k in [3, 5]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "cv2.medianBlur",
                "args_list": [k],  # medianBlurはint ksize
                "args_dict": {},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        }
    ])

# 1c) バイラテラルフィルタ（エッジ保持）
for d, sc, ss in [(5, 30, 30), (9, 60, 60)]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "cv2.bilateralFilter",
                "args_list": [d, sc, ss],
                "args_dict": {},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        }
    ])

# 2) 明るさ・コントラスト（線形：convertScaleAbs）
for alpha in [0.85, 1.00, 1.15]:   # コントラスト
    for beta in [-20, 0, 20]:      # 明るさ
        if(alpha == 1.00 and beta == 0):
            continue  # 恒等変換はスキップ
        data_expanding_img_reshape_log_list.append([
            {
                "mode": "ImgTransFuncRun",
                "args": {
                    "func_str": "cv2.convertScaleAbs",
                    "args_list": [],
                    "args_dict": {"alpha": float(alpha), "beta": int(beta)},
                    "result_iterate": True,
                    "iterate_key": 0,
                    "stack_index": -1
                }
            }
        ])

# 3) 自動コントラスト（ヒストグラムクリップ版）
for clip in [1.0, 2.5, 5.0]:
    for per_channel in [False, True]:
        data_expanding_img_reshape_log_list.append([
            {
                "mode": "ImgTransFuncRun",
                "args": {
                    "func_str": "baseLib.imgLib.AutoContrast",
                    "args_list": [],
                    "args_dict": {"clip_hist_percent": float(clip), "per_channel": bool(per_channel)},
                    "result_iterate": True,
                    "iterate_key": 0,
                    "stack_index": -1
                }
            }
        ])

# 4) 照明むら補正（モルフォロジーで背景推定→正規化）
for r in [15, 31, 61]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "baseLib.imgLib.NormalizeIllumination",
                "args_list": [],
                "args_dict": {"radius": int(r)},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        }
    ])

# 5) 軽い平行移動（座標も追従：WarpPerspective）
for tx in [-8, 8]:
    for ty in [-8, 8]:
        data_expanding_img_reshape_log_list.append([
            {
                "mode": "WarpPerspective",
                "args": {
                    "M": [[1.0, 0.0, float(tx)],
                          [0.0, 1.0, float(ty)],
                          [0.0, 0.0, 1.0]]
                    # dsize/borderは省略（デフォルト：元サイズ・黒埋め）
                }
            }
        ])

# 6) 微回転（原点回り・±5度）
for deg in [-5, 5]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    M = [[ float(c),  float(s), 0.0],
         [ -float(s), float(c), 0.0],
         [ 0.0,       0.0,      1.0]]
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {"M": M}
        }
    ])

# 7) 軽いシアー（X方向のみ）
for shx in [-0.05, 0.05]:
    M = [[1.0, float(shx), 0.0],
         [0.0, 1.0,        0.0],
         [0.0, 0.0,        1.0]]
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {"M": M}
        }
    ])

# 8) コントラスト→ノイズ（順序効果も見る）
for (alpha, beta, sigma) in [(1.15, 0, 10), (0.85, 0, 20)]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "cv2.convertScaleAbs",
                "args_list": [],
                "args_dict": {"alpha": float(alpha), "beta": int(beta)},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        },
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "baseLib.imgLib.addGaussianNoiseImg",
                "args_list": [],
                "args_dict": {"mu": 0, "sigma": int(sigma)},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        }
    ])

# 9) 等方スケール（原点基準）
for s in [0.90, 0.95, 1.05, 1.10]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {
                "M": [
                    [float(s), 0.0,      0.0],
                    [0.0,      float(s), 0.0],
                    [0.0,      0.0,      1.0]
                ]
            }
        }
    ])

# 10) 弱い異方スケール（原点基準）
for sx, sy in [(1.05, 0.95), (0.95, 1.05)]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {
                "M": [
                    [float(sx), 0.0,       0.0],
                    [0.0,       float(sy), 0.0],
                    [0.0,       0.0,       1.0]
                ]
            }
        }
    ])

# 11) ズーム + 微平行移動（原点基準スケールの見かけ片寄りを軽補正）
zoom_translate_specs = [
    (1.05, -8, -8),  # ほんの少しズームイン＋左上寄りを戻すイメージ
    (0.95,  8,  8),  # 少しズームアウト＋右下方向へ軽く寄せる
]
for s, tx, ty in zoom_translate_specs:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {
                "M": [
                    [float(s), 0.0,      0.0],
                    [0.0,      float(s), 0.0],
                    [0.0,      0.0,      1.0]
                ]
            }
        },
        {
            "mode": "WarpPerspective",
            "args": {
                "M": [
                    [1.0, 0.0, float(tx)],
                    [0.0, 1.0, float(ty)],
                    [0.0, 0.0, 1.0]
                ]
            }
        }
    ])

# 12) ズーム + 微ノイズ（実機でのゲイン上げ・再圧縮などの見かけを模擬）
for s, sigma in [(1.05, 8), (0.95, 8)]:
    data_expanding_img_reshape_log_list.append([
        {
            "mode": "WarpPerspective",
            "args": {
                "M": [
                    [float(s), 0.0,      0.0],
                    [0.0,      float(s), 0.0],
                    [0.0,      0.0,      1.0]
                ]
            }
        },
        {
            "mode": "ImgTransFuncRun",
            "args": {
                "func_str": "baseLib.imgLib.addGaussianNoiseImg",
                "args_list": [],
                "args_dict": {"mu": 0, "sigma": int(sigma)},
                "result_iterate": True,
                "iterate_key": 0,
                "stack_index": -1
            }
        }
    ])

print("Total augmentation methods: ", len(data_expanding_img_reshape_log_list))

def merged_class(label_file_path,new_label_path,others_class_num):
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

tempfile_dir=tempfile.TemporaryDirectory()
unzip_patch_dir=Path(tempfile_dir.name)
unzip_patch_dir.mkdir(parents=True,exist_ok=True)
zipfile.ZipFile(patch_zip,"r").extractall(path=unzip_patch_dir)

json_path=Path(unzip_patch_dir/f"{patch_str}/dataset.json")
if(not json_path.exists()):
    raise FileNotFoundError(f"JSON file not found in the patch zip: {str(json_path)}")

"""json format example:
[
    {
        "name": "A_img00001",
        "img_path": "train/images/A_img00001.jpg",
        "label_path": "train/labels/A_img00001.txt"
    },
    {
        "name": "A_img00002",
        "img_path": "train/images/A_img00002.jpg",
        "label_path": "train/labels/A_img00002.txt"
    },
    ...
]
"""
with open(json_path,mode="r") as f:
    patch_data=json.load(f)

new_data_paths=[]

# original + augmentations copy
counter=0
for img_idx,dataset_data in enumerate(patch_data):
    img_name=dataset_data["name"]
    img_path=Path(unzip_patch_dir/dataset_data["image_path"])
    label_file_path=Path(unzip_patch_dir/dataset_data["label_path"])

    if(not img_path.exists()):
        print(f"  Warning: Image file not found: {str(img_path)}. Skipping.")
        continue
    if(not label_file_path.exists()):
        print(f"  Warning: Label file not found: {str(label_file_path)}. Skipping.")
        continue

    yolo_ann=baseLib.imgLib.YOLOANN(
        img_mode=baseLib.imgLib.YOLOANN.IMG_MODE_FILE,
        ann_data_arg=str(label_file_path),
        arg=str(img_path),
    )

    new_img_path=baseLib.IOLib.changeDirectoryPath(img_path,unzip_patch_dir/Path(patch_str),new_dataset_dir)
    new_img_path.parent.mkdir(parents=True,exist_ok=True)

    new_label_path=baseLib.IOLib.changeDirectoryPath(label_file_path,unzip_patch_dir/Path(patch_str),new_dataset_dir)
    new_label_path.parent.mkdir(parents=True,exist_ok=True)

    new_merged_label_path=Path(str(new_label_path).replace("/labels/","/merged_labels/",1))
    new_merged_label_path.parent.mkdir(parents=True,exist_ok=True)

    print(f"Processing: {img_name} ({img_idx+1}/{len(patch_data)})")
    if(not isinstance(yolo_ann,baseLib.imgLib.YOLOANN)):
        print("  Warning: Annotation is not YOLOANN instance. Skipping.")
        continue

    # Copy original image and label
    baseLib.IOLib.copySafe(img_path,new_img_path)
    baseLib.IOLib.copySafe(label_file_path,new_label_path)
    merged_class(label_file_path,new_merged_label_path,others_class_num=5)

    new_data_paths.append((new_img_path,new_label_path,new_merged_label_path))

    counter+=1

    # Generate augmentations
    for idx,new_yolo_ann in enumerate(yolo_ann.augmentationGenerator(data_expanding_img_reshape_log_list)):
        print(f"    {img_name} : Augmentation ({idx+1}/{len(data_expanding_img_reshape_log_list)})")

        new_augmentation_img_path=new_img_path.with_name(new_img_path.stem+"_aug_v1_{:03d}".format(idx)+new_img_path.suffix)
        new_augmentation_label_path=new_label_path.with_name(new_label_path.stem+"_aug_v1_{:03d}".format(idx)+new_label_path.suffix)
        new_augmentation_merged_label_path=new_merged_label_path.with_name(new_merged_label_path.stem+"_aug_v1_{:03d}".format(idx)+new_merged_label_path.suffix)

        new_yolo_ann.saveImgAndANNTXT(
            img_file_path=new_augmentation_img_path,
            ann_txt_file_path=new_augmentation_label_path,
            img_is_polygon_mode=False,
            img_draw_shape_flag=False,
            img_label_flag=False,
        )
        merged_class(new_augmentation_label_path,new_augmentation_merged_label_path,others_class_num=5)
        
        new_data_paths.append((new_augmentation_img_path,new_augmentation_label_path,new_augmentation_merged_label_path))
        counter+=1
    print()

print("Total files processed (including augmentations): ",counter)
