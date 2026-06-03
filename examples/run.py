"""panel_label 插件用法示例（线标 OCR 检测）。

单图模式（默认）：
    python plugins/vie-plugin-panel-label/examples/run.py <图片路径> [产品型号] [front|back|all]

批量评测模式（遍历目录、逐图检测、输出各型号正确率汇总）：
    python plugins/vie-plugin-panel-label/examples/run.py --batch <数据目录> [--rule all]
    自动识别两种目录结构：
      - 多型号父目录：每个子目录名即产品型号，内含该型号的 *.jpg；
      - 单型号目录：*.jpg 直接位于该目录下，目录名即产品型号。

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
from tqdm import tqdm

# 让示例在任意 cwd 下都能 import 框架（services/schemas 在仓库根，未作为包安装）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import vie_plugin_panel_label.plugin  # noqa: E402,F401  导入即触发 @detection_factory.register("panel_label")
from vie_plugin_panel_label import PRODUCT_guideline  # noqa: E402
from vie_plugin_panel_label.product_type import PRODUCT_TYPE  # noqa: E402
from services.api import detection_factory  # noqa: E402
from schemas.data_base import InputParamsBusiness  # noqa: E402


def detect(detector, image, product_type, rule):
    """手动复刻生产 detect() 流程，返回 (item, result)。

    与 BusinessLogicBase.detect 等价（单次推理），但额外暴露 ctx.raw_result：
        result = ctx.result      生产响应 MoMResult（坐标已归一化，等同 API 输出）
        item   = ctx.raw_result  PanellabelItem（原图像素坐标，含 text_det_points 供可视化）

    生产环境里标准顺序(standard_result)与引导框(guideline)随请求下发；示例为方便
    测试仍从本地 PRODUCT_TYPE / PRODUCT_guideline 词典读取并经 ctx.extra 注入。
    型号未在本地词典登记时返回 (item, None)，仅画框不判定。
    """
    standard_result = PRODUCT_TYPE.get(product_type)
    guideline = PRODUCT_guideline.get(product_type)
    extra = {"standard_result": standard_result, "guideline": guideline}
    params = InputParamsBusiness(image=image, product_type=product_type, rule=rule, extra=extra)
    ctx = detector.build_context(params)
    detector.preprocess_hook(ctx)
    ctx.raw_result = detector.detector.infer(ctx.image)
    if standard_result is None or guideline is None:
        return ctx.raw_result, None
    detector.business_post_process(ctx)
    if detector.should_normalize(ctx):
        detector.normalize_hook(ctx)
    detector.finalize_hook(ctx)
    return ctx.raw_result, ctx.result


def visualize(image, item, result, product_type):
    """画 guideline 参考框（绿）+ 线标框（绿=通过 红=异常）+ 文本检测框（蓝）+ 识别文本。

    线标框/文本框均用 item 的原图像素坐标；状态与文本取自 result.detailList（与 item 逐项对齐）。
    """
    h, w = image.shape[:2]
    if product_type in PRODUCT_guideline:
        gx, gy, gw, gh = PRODUCT_guideline[product_type]
        cv2.rectangle(image, (int(gx * w), int(gy * h)), (int((gx + gw) * w), int((gy + gh) * h)), (0, 255, 0), 2)

    details = result.detailList if result is not None else []

    for i, coord in enumerate(item.Points):
        # 线标框（YOLO minAreaRect，8 值像素坐标）
        if len(coord) == 8:
            pts = np.array([[int(coord[j]), int(coord[j + 1])] for j in range(0, 8, 2)], np.int32)
            status = details[i].status if i < len(details) else True
            name = details[i].name if i < len(details) else (item.texts[i] or "")
            color = (0, 255, 0) if status else (0, 0, 255)
            cv2.polylines(image, [pts], True, color, 2)
            cv2.putText(image, name or "", (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        # 文本检测框（PaddleOCR Stage1，反映射回原图，蓝色）
        tdp = item.text_det_points[i] if i < len(item.text_det_points) else None
        if tdp is not None and len(tdp) >= 3:
            tpts = np.array(tdp, np.int32).reshape(-1, 2)
            cv2.polylines(image, [tpts], True, (255, 0, 0), 2)
    return image


def save_rec_hard_samples(out_dir, src_stem, item, result, product_type):
    """把识别错误的文本行按 PPOCR rec 格式落盘，返回保存条数。

    仅当 result.message=='mismatch'（observed 与 standard 数量对齐）时处理：
    对 status=False 的第 i 行，把 item.text_crops[i] 写到
    <out_dir>/images/<src_stem>_line{i}.png，并向 <out_dir>/label.txt
    追加 'images/xxx.png\\t<standard[i]>'（standard=PRODUCT_TYPE[product_type]）。
    """
    if result is None or result.message != "mismatch":
        return 0
    standard = PRODUCT_TYPE.get(product_type, [])
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    lines = []
    for i, d in enumerate(result.detailList):
        if d.status:
            continue
        crop = item.text_crops[i] if i < len(item.text_crops) else None
        if crop is None or i >= len(standard):
            continue
        name = f"{src_stem}_line{i}.png"
        img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_dir / name), crop)
        lines.append(f"images/{name}\t{standard[i]}")
    if lines:
        with open(out_dir / "label.txt", "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    return len(lines)


def run_single(detector, image_path, product_type, rule, save_rec_hard=None):
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    item, result = detect(detector, image, product_type, rule)
    if result is not None:
        # 生产响应（MoMResult，等同 API 输出）
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        # observed/standard 对照摘要（从 result 派生，便于人工核对）
        summary = {
            "status": result.status,
            "message": result.message,
            "observed": [d.name for d in result.detailList],
            "standard": PRODUCT_TYPE.get(product_type, []),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"产品型号 '{product_type}' 未注册，仅可视化检测框不做判定。")
    save_path = "panel_label_result.jpg"
    cv2.imwrite(save_path, visualize(image, item, result, product_type))
    print(f"可视化结果已保存: {save_path}")
    if save_rec_hard:
        n = save_rec_hard_samples(save_rec_hard, Path(image_path).stem, item, result, product_type)
        if n:
            print(f"已落盘 {n} 条识别错误样本到 {save_rec_hard}")


def run_batch(detector, data_dir, rule, vis_dir, save_rec_hard=None):
    data_dir = Path(data_dir)
    vis_dir = Path(vis_dir)
    vis_dir.mkdir(parents=True, exist_ok=True)
    # 自动识别：若 data_dir 下直接有 *.jpg，视作单个型号（型号名=目录名）；
    # 否则按「父目录」遍历各子目录为不同型号。
    if next(data_dir.glob("*.jpg"), None) is not None:
        img_lists = {data_dir.name: sorted(data_dir.glob("*.jpg"))}
    else:
        product_types = [
            d.name for d in sorted(data_dir.iterdir()) if d.is_dir() and next(d.glob("*.jpg"), None) is not None
        ]
        if not product_types:
            raise SystemExit(f"未在 {data_dir} 下发现 *.jpg，也未发现含 *.jpg 的型号子目录")
        img_lists = {pt: sorted((data_dir / pt).glob("*.jpg")) for pt in product_types}
    product_types = list(img_lists.keys())
    total_imgs = sum(len(v) for v in img_lists.values())
    print(f"检测到 {len(product_types)} 个型号，共 {total_imgs} 张图: {', '.join(product_types)}", flush=True)

    summary = {}
    # 外层按型号、内层按图片各一条进度条；tqdm 写 stderr 且自带刷新，管道下也实时可见
    for pt in tqdm(product_types, desc="型号", unit="型号", position=0):
        imgs = img_lists[pt]
        positive = 0
        bar = tqdm(imgs, desc=f"  {pt}", unit="img", position=1, leave=False)
        for ip in bar:
            image = cv2.imread(str(ip))
            if image is None:
                tqdm.write(f"  无法读取: {ip}")
                continue
            item, result = detect(detector, image, pt, rule)
            ok = bool(result.status) if result is not None else False
            positive += ok
            vis_dst_path = str(vis_dir / f"{pt}_{ip.stem}_res.jpg")
            cv2.imwrite(vis_dst_path, visualize(image, item, result, pt))
            if save_rec_hard:
                save_rec_hard_samples(save_rec_hard, f"{pt}_{ip.stem}", item, result, pt)
            if not ok:
                tqdm.write(f"  FAIL: {ip} ,vis:{vis_dst_path}")
            bar.set_postfix_str(f"OK {positive}/{bar.n + 1}")
        summary[pt] = (positive, len(imgs))
        tqdm.write(f"[{pt}] 正确 {positive}/{len(imgs)} = {positive / max(len(imgs), 1):.2%}")

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
    ap.add_argument(
        "rule_pos",
        nargs="?",
        default=None,
        choices=["front", "back", "all"],
        metavar="rule",
        help="单图模式字符比较规则（位置参数）",
    )
    ap.add_argument("--batch", metavar="DIR", help="批量评测：数据目录（子目录名=型号）")
    ap.add_argument(
        "--rule",
        dest="rule_opt",
        default=None,
        choices=["front", "back", "all"],
        help="字符比较规则（选项写法，批量/单图均可用）",
    )
    ap.add_argument("--vis-dir", default="output/panel_label_vis", help="批量模式可视化输出目录")
    ap.add_argument(
        "--save-rec-hard",
        metavar="DIR",
        default=None,
        help="把识别错误(mismatch)的文本行按 PPOCR rec 格式落盘到 DIR（images/ + label.txt）",
    )
    args = ap.parse_args()

    # --rule 选项优先，其次单图位置参数，最后默认 all
    rule = args.rule_opt or args.rule_pos or "all"

    detector = detection_factory.get_scenarios("panel_label")
    if args.batch:
        run_batch(detector, args.batch, rule, args.vis_dir, save_rec_hard=args.save_rec_hard)
    else:
        run_single(detector, args.image, args.product_type, rule, save_rec_hard=args.save_rec_hard)


if __name__ == "__main__":
    main()
