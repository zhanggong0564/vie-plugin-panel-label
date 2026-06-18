#!/usr/bin/env python3
"""
生成两个训练数据集（图片均来自 crop_ocr/images/）：

1. 方向分类数据集  →  train/cls/
   crop_cls/0 有记录 → label 0（正向），存入 images/0/
   crop_cls/1 有记录 → label 1（反向），存入 images/1/
   无记录 → 跳过

2. OCR 识别数据集  →  train/ocr/all/
   crop_ocr 图 + JSON 文字标签，cls1 旋转 180°。
   无 crop_cls 方向记录的，用方向模型(TextLineOrientationClassification)预测 0/1
   作为兜底（USE_ORIENT_PRED=True 时；置 False 则退回旧行为：直接跳过）。
   1图1条，不做额外增强

均按 6/4 划分 train / val，保留原始文件名。
"""

import os

# WSL2/headless 下 Paddle/OpenCV-Qt 无法连 X11，须在 import paddleocr 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import json
import random
import shutil
from pathlib import Path

import cv2

DATA_ROOT = Path("/mnt/d/workspace/mobile_vision/data/annotated/line_marker/data")
TRAIN_ROOT = Path("/mnt/d/workspace/mobile_vision/data/annotated/line_marker/train")

CLS_OUT = TRAIN_ROOT / "cls"
REC_OUT = TRAIN_ROOT / "ocr" / "all"

# 方向模型（与生产 config 同款），给 crop_cls 缺失的 crop 预测方向 0/1
USE_ORIENT_PRED = True
ORIENT_MODEL_DIR = "/mnt/d/workspace/WSL/VisInferEngine/mobile_vision/weights/panel_label/v2/textline_ori_lcnet_v2"
ORIENT_BATCH = 64

# 增量模式：已在 train.txt/val.txt 里的 crop 直接跳过（不读json/不跑预测/不重写），
# 旧切分原样保留，新增的按 VAL_RATIO 追加。置 False 则全量重建（重新洗牌切分）。
INCREMENTAL = True

VAL_RATIO = 0.4
random.seed(42)

_orient_model = None


def _get_orient_model():
    """惰性加载方向分类模型（仅在确有 crop 需要预测时才加载）。"""
    global _orient_model
    if _orient_model is None:
        from paddleocr import TextLineOrientationClassification

        _orient_model = TextLineOrientationClassification(
            model_name="PP-LCNet_x1_0_textline_ori",
            model_dir=ORIENT_MODEL_DIR,
        )
    return _orient_model


def predict_orientations(img_paths: list[Path], batch: int = ORIENT_BATCH) -> list[int]:
    """批量预测方向，返回与输入等长的 0/1 列表（0=正向，1=反向需旋转180°）。"""
    model = _get_orient_model()
    out: list[int] = []
    for i in range(0, len(img_paths), batch):
        chunk = [str(p) for p in img_paths[i : i + batch]]
        results = model.predict(chunk)
        out.extend(int(r["class_ids"][0]) for r in results)
    return out


def write_list(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for img_rel, label in records:
            f.write(f"{img_rel}\t{label}\n")


def read_list(path: Path) -> list[tuple[str, str]]:
    """读已有清单 'img_rel\\tlabel'，文件不存在则返回空。"""
    if not path.exists():
        return []
    out: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            img_rel, label = line.split("\t", 1)
            out.append((img_rel, label))
    return out


def split_records(records: list, val_ratio: float) -> tuple[list, list]:
    random.shuffle(records)
    n = int(len(records) * (1 - val_ratio))
    return records[:n], records[n:]


def build_cls_index(data_root: Path) -> dict[str, int]:
    """返回 {文件名stem: 0或1}，从所有 crop_cls/{0,1}/ 收集。"""
    index: dict[str, int] = {}
    for label in (0, 1):
        for img in data_root.rglob(f"crop_cls/{label}/*.jpg"):
            index[img.stem] = label
    return index


# ── 方向分类数据集 ──────────────────────────────────────────────────────────────


def build_cls():
    out_img = CLS_OUT / "images"
    (out_img / "0").mkdir(parents=True, exist_ok=True)
    (out_img / "1").mkdir(parents=True, exist_ok=True)

    cls_index = build_cls_index(DATA_ROOT)

    existing_train = read_list(CLS_OUT / "train.txt") if INCREMENTAL else []
    existing_val = read_list(CLS_OUT / "val.txt") if INCREMENTAL else []
    done_names = {Path(img_rel).name for img_rel, _ in existing_train + existing_val}

    new_records: list[tuple[str, str]] = []
    skip_no_cls = skip_done = 0

    for src in sorted(DATA_ROOT.rglob("crop_ocr/images/*.jpg")):
        if src.name in done_names:
            skip_done += 1
            continue
        label = cls_index.get(src.stem)
        if label is None:
            skip_no_cls += 1
            continue

        dst = out_img / str(label) / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        new_records.append((f"images/{label}/{src.name}", str(label)))

    new_train, new_val = split_records(new_records, VAL_RATIO)
    train_r = existing_train + new_train
    val_r = existing_val + new_val
    write_list(CLS_OUT / "train.txt", train_r)
    write_list(CLS_OUT / "val.txt", val_r)

    print(f"[cls] 新增 {len(new_records)}（train+{len(new_train)} val+{len(new_val)}）  已跳过(已处理):{skip_done}")
    print(f"      总计 {len(train_r) + len(val_r)}  train={len(train_r)}  val={len(val_r)}  跳过(无cls)={skip_no_cls}")


# ── OCR 识别数据集 ─────────────────────────────────────────────────────────────


def build_rec():
    out_img = REC_OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    cls_index = build_cls_index(DATA_ROOT)

    # 增量模式：以已写入 train.txt/val.txt 的 crop 为「已处理」基准
    existing_train = read_list(REC_OUT / "train.txt") if INCREMENTAL else []
    existing_val = read_list(REC_OUT / "val.txt") if INCREMENTAL else []
    done_names = {Path(img_rel).name for img_rel, _ in existing_train + existing_val}

    # 第一遍：收集有效条目（文字非空、图存在），label 为 None 表示 crop_cls 无方向记录
    pending: list[tuple[Path, str, "int | None"]] = []
    skip_done = skip_no_text = skip_no_img = 0

    for json_path in sorted(DATA_ROOT.rglob("crop_ocr/jsons/*.json")):
        # 已处理的直接跳过：连 json 都不读、方向模型也不跑
        if f"{json_path.stem}.jpg" in done_names:
            skip_done += 1
            continue

        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not shapes or not shapes[0].get("description", "").strip():
            skip_no_text += 1
            continue
        text = shapes[0]["description"].strip()

        src = json_path.parent.parent / "images" / f"{json_path.stem}.jpg"
        if not src.exists():
            skip_no_img += 1
            continue

        pending.append((src, text, cls_index.get(json_path.stem)))

    # 方向预测兜底：crop_cls 缺失的，用方向模型批量预测 0/1
    need_pred = [src for src, _, label in pending if label is None]
    n_pred = len(need_pred)
    pred_map: dict[Path, int] = {}
    if need_pred and USE_ORIENT_PRED:
        print(f"[rec] crop_cls 缺失 {n_pred} 张，调用方向模型预测中...")
        angles = predict_orientations(need_pred)
        pred_map = dict(zip(need_pred, angles))

    # 第二遍：落盘（反向 label==1 旋转 180°），只处理新增条目
    new_records: list[tuple[str, str]] = []
    skip_no_cls = 0
    for src, text, label in pending:
        if label is None:
            if not USE_ORIENT_PRED:
                skip_no_cls += 1
                continue
            label = pred_map[src]

        dst = out_img / src.name
        if not dst.exists():
            img = cv2.imread(str(src))
            if img is None:
                skip_no_img += 1
                continue
            if label == 1:
                img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(dst), img)

        new_records.append((f"images/{src.name}", text))

    # 新增按 6/4 切分后，追加到旧切分之后（旧切分原样保留，避免重新洗牌）
    new_train, new_val = split_records(new_records, VAL_RATIO)
    train_r = existing_train + new_train
    val_r = existing_val + new_val
    write_list(REC_OUT / "train.txt", train_r)
    write_list(REC_OUT / "val.txt", val_r)

    # 字典基于全量（旧+新）标签重算，保证新增字符也并入
    chars = sorted({ch for _, lbl in (train_r + val_r) for ch in lbl})
    with open(REC_OUT / "dict.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(chars) + "\n")

    fill_msg = f"模型预测补方向:{n_pred}" if USE_ORIENT_PRED else f"无cls(已跳过):{skip_no_cls}"
    print(f"[rec] 新增 {len(new_records)}（train+{len(new_train)} val+{len(new_val)}）" f"  已跳过(已处理):{skip_done}")
    print(f"      总计 {len(train_r) + len(val_r)}  train={len(train_r)}  val={len(val_r)}")
    print(f"      {fill_msg}  空标签:{skip_no_text}  缺图:{skip_no_img}")
    print(f"      字典字符数: {len(chars)}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"数据源: {DATA_ROOT}")
    print(f"分类输出: {CLS_OUT}")
    print(f"识别输出: {REC_OUT}\n")
    build_cls()
    build_rec()
    print("\n完成。")
