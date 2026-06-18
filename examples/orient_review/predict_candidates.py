# orient_review/predict_candidates.py
"""用方向分类模型扫全量图片，把判为反向(180°)的挑成候选，写 candidates.json。

只挑 class_id==1(180_degree) 且 score>=阈值 的，给网页工具逐张人工复核。
不修改任何图片——改不改由复核工具点击决定。

用法:
  python predict_candidates.py                 # 用 review_io 里默认 IMAGES_DIR
  python predict_candidates.py --images-dir X --score-thresh 0.7 --model-dir Y
"""

import argparse
import os
import sys
import time
from glob import glob

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from paddleocr import TextLineOrientationClassification  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import review_io as rio  # noqa: E402

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
REVERSED_CLASS_ID = 1


def find_default_model_dir(start_dir):
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        cand = os.path.join(cur, "weights", "panel_label")
        if os.path.isdir(cand):
            dirs = sorted(glob(os.path.join(cand, "textline_ori_lcnet_v*")))
            if dirs:
                return dirs[-1]
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError("未自动找到 textline_ori_lcnet 权重，请用 --model-dir 指定")


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default=rio.IMAGES_DIR)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--score-thresh", type=float, default=0.7)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default=rio.CANDIDATES_JSON)
    ap.add_argument(
        "--flush-every", type=int, default=5, help="每多少个 batch 刷一次 candidates.json/scan_status.json(边扫边看)"
    )
    args = ap.parse_args()

    images_dir = os.path.abspath(args.images_dir)
    if not os.path.isdir(images_dir):
        sys.exit(f"[错误] 图片目录不存在: {images_dir}")
    model_dir = args.model_dir or find_default_model_dir(os.path.dirname(os.path.abspath(__file__)))

    images = [os.path.join(images_dir, n) for n in sorted(os.listdir(images_dir)) if n.lower().endswith(IMG_EXTS)]
    if not images:
        sys.exit(f"[错误] 目录内无图片: {images_dir}")

    print(f"[信息] 图片目录 : {images_dir}")
    print(f"[信息] 图片总数 : {len(images)}")
    print(f"[信息] 模型目录 : {model_dir}")
    print(f"[信息] 置信阈值 : {args.score_thresh}")

    model = TextLineOrientationClassification(model_name="PP-LCNet_x1_0_textline_ori", model_dir=model_dir)

    total = len(images)

    def flush(done, low_conf, finished):
        # 候选按分数排序后原子写出；复核工具轮询这两个文件即可边扫边看
        candidates.sort(key=lambda c: c["score"], reverse=True)
        rio.save_candidates(candidates, args.out)
        rio.save_scan(
            {
                "done": done,
                "total": total,
                "found": len(candidates),
                "low_conf": low_conf,
                "finished": finished,
                "updated_at": time.time(),
            }
        )

    candidates = []
    low_conf = 0
    t0 = time.time()
    done = 0
    flush(0, 0, False)  # 先落一个空状态，复核页面立刻能显示"扫描中 0/N"
    for bi, batch in enumerate(batched(images, args.batch_size), 1):
        for res in model.predict(batch):
            done += 1
            if int(res["class_ids"][0]) == REVERSED_CLASS_ID:
                score = float(res["scores"][0])
                if score >= args.score_thresh:
                    candidates.append({"path": res["input_path"], "score": round(score, 4)})
                else:
                    low_conf += 1
        if bi % args.flush_every == 0:
            flush(done, low_conf, False)
            rate = done / max(time.time() - t0, 1e-6)
            print(
                f"  进度 {done}/{total}  候选 {len(candidates)}  " f"低置信跳过 {low_conf}  ({rate:.1f} img/s)",
                flush=True,
            )

    flush(done, low_conf, True)
    print(f"\n候选已写出: {args.out}")
    print(f"判为反向(180°)候选 : {len(candidates)}  (score>={args.score_thresh})")
    print(f"低置信跳过        : {low_conf}")
    print("复核工具已可全程使用: python review_tool.py -> http://127.0.0.1:5001")


if __name__ == "__main__":
    main()
