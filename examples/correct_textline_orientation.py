'''
@Description : 用方向分类模型(PP-LCNet_x1_0_textline_ori)矫正 OCR 识别数据集中方向反(180°)的样本。

背景：line_marker 识别数据集里有部分裁剪小图是上下颠倒的，混进识别训练会让 loss 难收敛。
本脚本对 images/ 下每张图做方向分类，class_id==1(180_degree) 且置信度达标的，原地旋转 180° 覆盖。

注意：
  * 识别标签(train.txt/val.txt 的文本)与方向无关，旋转图片后标签不变，无需改动标签文件。
  * 原地覆盖是破坏性操作，默认会先把被矫正的原图备份到 <images>/../orient_backup/，
    并写出 orient_correction_manifest.csv 审计清单(可据此回滚)。
  * 建议先 --dry-run 看会矫正多少张，确认后再去掉 --dry-run 实际执行。

用法示例：
  # 1) 先 dry-run 预览(不改任何文件)
  python correct_textline_orientation.py \
      --images-dir /mnt/d/workspace/mobile_vision/data/annotated/line_marker/train/ocr/all/images \
      --dry-run

  # 2) 确认后实际矫正(默认带备份)
  python correct_textline_orientation.py \
      --images-dir /mnt/d/workspace/mobile_vision/data/annotated/line_marker/train/ocr/all/images
'''

import argparse
import csv
import os
import shutil
import sys
import time
from glob import glob

import cv2

# 跳过 paddle 联网检查模型源(离线本地权重)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import TextLineOrientationClassification  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REVERSED_CLASS_ID = 1  # 0 -> 0_degree(正向), 1 -> 180_degree(倒置)


def find_default_model_dir(start_dir: str) -> str:
    """从脚本所在仓库向上找 weights/panel_label/ 下版本号最大的 textline_ori_lcnet_* 目录。"""
    cur = os.path.abspath(start_dir)
    for _ in range(8):
        cand = os.path.join(cur, "weights", "panel_label")
        if os.path.isdir(cand):
            dirs = sorted(glob(os.path.join(cand, "textline_ori_lcnet_v*")))
            if dirs:
                return dirs[-1]  # 版本号最大(字典序对 v2<v3<v4 成立)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("未自动找到 textline_ori_lcnet 权重目录，请用 --model-dir 指定")


def list_images(images_dir: str) -> list:
    files = []
    for name in sorted(os.listdir(images_dir)):
        if name.lower().endswith(IMG_EXTS):
            files.append(os.path.join(images_dir, name))
    return files


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    parser = argparse.ArgumentParser(description="用方向分类模型矫正 OCR 数据集中方向反的图片")
    parser.add_argument(
        "--images-dir", required=True,
        help="待矫正图片目录(如 .../train/ocr/all/images)",
    )
    parser.add_argument(
        "--model-dir", default=None,
        help="方向分类模型导出目录，默认自动定位 weights/panel_label/textline_ori_lcnet_v*(最新版)",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.7,
        help="判为 180° 的最低置信度，低于此值不矫正(默认 0.7)",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="批量推理大小(默认 64)")
    parser.add_argument("--dry-run", action="store_true", help="只统计不改文件")
    parser.add_argument("--no-backup", action="store_true", help="不备份被矫正的原图(谨慎)")
    parser.add_argument(
        "--backup-dir", default=None,
        help="原图备份目录，默认 <images-dir>/../orient_backup",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="审计清单 csv 路径，默认 <images-dir>/../orient_correction_manifest.csv",
    )
    args = parser.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        sys.exit(f"[错误] 图片目录不存在: {images_dir}")

    model_dir = args.model_dir or find_default_model_dir(os.path.dirname(os.path.abspath(__file__)))
    if not os.path.isdir(model_dir):
        sys.exit(f"[错误] 模型目录不存在: {model_dir}")

    parent = os.path.dirname(images_dir)
    backup_dir = args.backup_dir or os.path.join(parent, "orient_backup")
    manifest_path = args.manifest or os.path.join(parent, "orient_correction_manifest.csv")

    images = list_images(images_dir)
    if not images:
        sys.exit(f"[错误] 目录内未找到图片: {images_dir}")

    print(f"[信息] 图片目录 : {images_dir}")
    print(f"[信息] 图片总数 : {len(images)}")
    print(f"[信息] 模型目录 : {model_dir}")
    print(f"[信息] 置信阈值 : {args.score_thresh}")
    print(f"[信息] 模式     : {'DRY-RUN(不改文件)' if args.dry_run else '实际矫正'}")
    if not args.dry_run:
        print(f"[信息] 备份     : {'关闭' if args.no_backup else backup_dir}")
        print(f"[信息] 审计清单 : {manifest_path}")

    model = TextLineOrientationClassification(
        model_name="PP-LCNet_x1_0_textline_ori",
        model_dir=model_dir,
    )

    if not args.dry_run and not args.no_backup:
        os.makedirs(backup_dir, exist_ok=True)

    corrected = []       # (path, score)
    low_conf_skipped = 0  # 判 180° 但置信度不足而跳过
    failed = []          # 读写失败
    t0 = time.time()
    done = 0

    for batch in batched(images, args.batch_size):
        for res in model.predict(batch):
            path = res["input_path"]
            class_id = int(res["class_ids"][0])
            score = float(res["scores"][0])
            done += 1

            if class_id == REVERSED_CLASS_ID:
                if score < args.score_thresh:
                    low_conf_skipped += 1
                else:
                    corrected.append((path, score))
                    if not args.dry_run:
                        ok = rotate_in_place(path, backup_dir, args.no_backup)
                        if not ok:
                            failed.append(path)
                            corrected.pop()

        if done % (args.batch_size * 10) == 0 or done >= len(images):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  进度 {done}/{len(images)}  已矫正 {len(corrected)}  "
                  f"低置信跳过 {low_conf_skipped}  ({rate:.1f} img/s)")

    # 写审计清单
    if not args.dry_run and corrected:
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image", "score", "action", "backup"])
            for path, score in corrected:
                bk = "" if args.no_backup else os.path.join(backup_dir, os.path.basename(path))
                w.writerow([path, f"{score:.4f}", "rotate_180", bk])

    print("\n========== 汇总 ==========")
    print(f"扫描图片   : {len(images)}")
    print(f"矫正(180°) : {len(corrected)}")
    print(f"低置信跳过 : {low_conf_skipped} (判为反向但 score < {args.score_thresh})")
    if failed:
        print(f"读写失败   : {len(failed)}")
        for p in failed[:10]:
            print(f"  - {p}")
    if args.dry_run:
        print("\n[DRY-RUN] 未修改任何文件。去掉 --dry-run 即按上述结果实际矫正。")
    else:
        print(f"\n已原地矫正 {len(corrected)} 张。")
        if not args.no_backup:
            print(f"原图备份   : {backup_dir}")
        if corrected:
            print(f"审计清单   : {manifest_path}")
        print("提示：识别标签与方向无关，train.txt/val.txt 无需改动。")


def rotate_in_place(path: str, backup_dir: str, no_backup: bool) -> bool:
    """旋转 180° 后原地覆盖，覆盖前先备份原图。失败返回 False。"""
    img = cv2.imread(path)
    if img is None:
        # 兼容中文路径
        import numpy as np
        try:
            data = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            img = None
    if img is None:
        print(f"  [警告] 无法读取，跳过: {path}")
        return False

    if not no_backup:
        try:
            shutil.copy2(path, os.path.join(backup_dir, os.path.basename(path)))
        except Exception as e:
            print(f"  [警告] 备份失败({e})，跳过: {path}")
            return False

    rotated = cv2.rotate(img, cv2.ROTATE_180)
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, rotated)
    if not ok:
        print(f"  [警告] 编码失败，跳过: {path}")
        return False
    buf.tofile(path)  # 兼容中文路径的原地写
    return True


if __name__ == "__main__":
    main()
