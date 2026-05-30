"""panel_label 插件用法示例（线标 OCR 检测）。

运行（从仓库根目录）：
    python plugins/vie-plugin-panel-label/examples/run.py <图片路径> [产品型号] [front|back|all]

前置：已 `pip install -e plugins/vie-plugin-panel-label`；OCR 模型权重就位。
"""
import os
import sys
import json

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


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test.jpg"
    product_type = sys.argv[2] if len(sys.argv) > 2 else ""
    rule = sys.argv[3] if len(sys.argv) > 3 else "all"
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"无法读取图片: {image_path}")
    h, w = image.shape[:2]

    detector = detection_factory.get_scenarios("panel_label")
    result = detector.detect(InputParamsBusiness(image=image, product_type=product_type, rule=rule))
    out = result.to_dict()
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # 可视化：先画 guideline 参考框（绿），再画归一化 8 点检测框（绿=通过 红=异常）
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

    save_path = "panel_label_result.jpg"
    cv2.imwrite(save_path, image)
    print(f"可视化结果已保存: {save_path}")


if __name__ == "__main__":
    main()
