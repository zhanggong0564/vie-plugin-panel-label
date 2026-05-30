"""panel_label 插件用法示例（线标 OCR 检测）。

单图模式（默认）：
    python plugins/vie-plugin-panel-label/examples/run.py <图片路径> [产品型号] [front|back|all]

批量评测模式（遍历目录、逐图检测、输出各型号正确率汇总）：
    python plugins/vie-plugin-panel-label/examples/run.py --batch <数据目录> [--rule all]
    数据目录下每个子目录名即产品型号，内含该型号的 *.jpg。

前置：已 `pip install -e plugins/vie-plugin-panel-label`；OCR 模型权重就位。
"""
import os
import sys
import json
import argparse
from pathlib import Path

# WSL2/headless 下 Paddle/OpenCV-Qt 无法连 X11，须在所有 import 之前设置
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np

# 让示例在任意 cwd 下都能 import 框架（services/schemas 在仓库根，未作为包安装）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import vie_plugin_panel_label.plugin  # noqa: E402,F401  导入即触发 @detection_factory.register("panel_label")
from vie_plugin_panel_label import PRODUCT_guideline  # noqa: E402
from services.api import detection_factory  # noqa: E402
from schemas.data_base import InputParamsBusiness  # noqa: E402


def visualize(image, out, product_type):
    """画 guideline 参考框（绿）+ 归一化 8 点检测框（绿=通过 红=异常）+ 识别文本。"""
    h, w = image.shape[:2]
    if product_type in PRODUCT_guideline:
        gx, gy, gw, gh = PRODUCT_guideline[product_type]
        cv2.rectangle(image, (int(gx * w), int(gy * h)),
                      (int((gx + gw) * w), int((gy + gh) * h)), (0, 255, 0), 2)
    for item in out.get("detailList", []):
        coord = item.get("coordinate", [])
        if len(coord) != 8:
            continue
        pts = np.array([[int(coord[i] * w), int(coord[i + 1] * h)] for i in range(0, 8, 2)], np.int32)
        color = (0, 255, 0) if item.get("status") == "true" else (0, 0, 255)
        cv2.polylines(image, [pts], True, color, 2)
        cv2.putText(image, item.get("name") or "", (pts[0][0], pts[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return image


def run_single(detector, image_path, product_type, rule):
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    out = detector.detect(InputParamsBusiness(image=image, product_type=product_type, rule=rule)).to_dict()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    save_path = "panel_label_result.jpg"
    cv2.imwrite(save_path, visualize(image, out, product_type))
    print(f"可视化结果已保存: {save_path}")


def run_batch(detector, data_dir, rule, vis_dir):
    data_dir = Path(data_dir)
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    product_types = [d.name for d in sorted(data_dir.iterdir())
                     if d.is_dir() and next(d.glob("*.jpg"), None) is not None]
    if not product_types:
        raise SystemExit(f"未在 {data_dir} 下发现含 *.jpg 的型号子目录")
    print(f"检测到 {len(product_types)} 个型号: {', '.join(product_types)}")

    summary = {}
    for pt in product_types:
        imgs = sorted((data_dir / pt).glob("*.jpg"))
        positive = 0
        for ip in imgs:
            image = cv2.imread(str(ip))
            if image is None:
                print(f"  无法读取: {ip}")
                continue
            out = detector.detect(InputParamsBusiness(image=image, product_type=pt, rule=rule)).to_dict()
            ok = out["status"] == "true"
            positive += ok
            cv2.imwrite(str(vis_dir / f"{pt}_{ip.stem}_res.jpg"), visualize(image, out, pt))
            if not ok:
                print(f"  FAIL: {ip}")
        summary[pt] = (positive, len(imgs))
        print(f"[{pt}] 正确 {positive}/{len(imgs)} = {positive / max(len(imgs), 1):.2%}")

    print(f"\n{'=' * 50}\n各型号正确率汇总（共 {len(summary)} 个）\n{'=' * 50}")
    print(f"{'型号':<20}{'正确/总数':>14}{'正确率':>12}")
    print("-" * 50)
    tot_p = tot_t = 0
    for pt, (p, t) in sorted(summary.items()):
        print(f"{pt:<20}{f'{p}/{t}':>14}{p / max(t, 1):>11.2%}")
        tot_p += p
        tot_t += t
    print("-" * 50)
    print(f"{'总计':<20}{f'{tot_p}/{tot_t}':>14}{tot_p / max(tot_t, 1):>11.2%}")
    print(f"可视化结果输出目录: {vis_dir}")


def main():
    ap = argparse.ArgumentParser(description="panel_label 插件演示 / 批量评测")
    ap.add_argument("image", nargs="?", default="test.jpg", help="单图模式：图片路径")
    ap.add_argument("product_type", nargs="?", default="", help="单图模式：产品型号")
    ap.add_argument("rule", nargs="?", default="all", choices=["front", "back", "all"], help="字符比较规则")
    ap.add_argument("--batch", metavar="DIR", help="批量评测：数据目录（子目录名=型号）")
    ap.add_argument("--vis-dir", default="output/panel_label_vis", help="批量模式可视化输出目录")
    args = ap.parse_args()

    detector = detection_factory.get_scenarios("panel_label")
    if args.batch:
        run_batch(detector, args.batch, args.rule, args.vis_dir)
    else:
        run_single(detector, args.image, args.product_type, args.rule)


if __name__ == "__main__":
    main()
