#!/usr/bin/env python3
"""
生成两个训练数据集（图片均来自 crop_ocr/images/）：

1. 方向分类数据集  →  train/cls/
   crop_cls/0 有记录 → label 0（正向），存入 images/0/
   crop_cls/1 有记录 → label 1（反向），存入 images/1/
   无记录 → 跳过

2. OCR 识别数据集  →  train/ocr/all/
   crop_ocr 图 + JSON 文字标签，cls1 旋转 180°，无 cls 记录的跳过
   1图1条，不做额外增强

均按 6/4 划分 train / val，保留原始文件名。
"""

import json
import random
import shutil
from pathlib import Path

import cv2

DATA_ROOT  = Path("/mnt/d/workspace/mobile_vision/data/annotated/line_marker/data")
TRAIN_ROOT = Path("/mnt/d/workspace/mobile_vision/data/annotated/line_marker/train")

CLS_OUT = TRAIN_ROOT / "cls"
REC_OUT = TRAIN_ROOT / "ocr" / "all"

VAL_RATIO = 0.4
random.seed(42)


def write_list(path: Path, records: list[tuple[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for img_rel, label in records:
            f.write(f"{img_rel}\t{label}\n")


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
    records: list[tuple[str, str]] = []
    skip_no_cls = 0

    for src in sorted(DATA_ROOT.rglob("crop_ocr/images/*.jpg")):
        label = cls_index.get(src.stem)
        if label is None:
            skip_no_cls += 1
            continue

        dst = out_img / str(label) / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
        records.append((f"images/{label}/{src.name}", str(label)))

    train_r, val_r = split_records(records, VAL_RATIO)
    write_list(CLS_OUT / "train.txt", train_r)
    write_list(CLS_OUT / "val.txt", val_r)

    print(f"[cls] 总计 {len(records)}  train={len(train_r)}  val={len(val_r)}  跳过(无cls)={skip_no_cls}")


# ── OCR 识别数据集 ─────────────────────────────────────────────────────────────

def build_rec():
    out_img = REC_OUT / "images"
    out_img.mkdir(parents=True, exist_ok=True)

    cls_index = build_cls_index(DATA_ROOT)
    records: list[tuple[str, str]] = []
    skip_no_cls = skip_no_text = skip_no_img = 0

    for json_path in sorted(DATA_ROOT.rglob("crop_ocr/jsons/*.json")):
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        if not shapes or not shapes[0].get("description", "").strip():
            skip_no_text += 1
            continue
        text = shapes[0]["description"].strip()

        label = cls_index.get(json_path.stem)
        if label is None:
            skip_no_cls += 1
            continue

        src = json_path.parent.parent / "images" / f"{json_path.stem}.jpg"
        if not src.exists():
            skip_no_img += 1
            continue

        dst = out_img / src.name
        if not dst.exists():
            img = cv2.imread(str(src))
            if img is None:
                skip_no_img += 1
                continue
            if label == 1:
                img = cv2.rotate(img, cv2.ROTATE_180)
            cv2.imwrite(str(dst), img)

        records.append((f"images/{src.name}", text))

    train_r, val_r = split_records(records, VAL_RATIO)
    write_list(REC_OUT / "train.txt", train_r)
    write_list(REC_OUT / "val.txt", val_r)

    chars = sorted({ch for _, lbl in records for ch in lbl})
    with open(REC_OUT / "dict.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(chars) + "\n")

    print(f"[rec] 总计 {len(records)}  train={len(train_r)}  val={len(val_r)}")
    print(f"      跳过 - 无cls:{skip_no_cls}  空标签:{skip_no_text}  缺图:{skip_no_img}")
    print(f"      字典字符数: {len(chars)}")


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"数据源: {DATA_ROOT}")
    print(f"分类输出: {CLS_OUT}")
    print(f"识别输出: {REC_OUT}\n")
    build_cls()
    build_rec()
    print("\n完成。")
