'''
@Description : panel_label 场景内部领域数据结构（推理输出 / 判定结果 / 状态枚举）。

集中管理本场景的内部领域模型，与 schemas.py（Pydantic API 请求契约）职责分离。
'''

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class ErrorType(str, Enum):
    MISSING = "missing"
    EXTRA = "extra"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    OK = "ok"


@dataclass
class PanellabelItem:
    """直送 OCR 管线的逐行输出，各 List 字段按行逐项对齐。"""
    Points: List[np.ndarray] = field(default_factory=list)
    index: List[int] = field(default_factory=list)
    class_id: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    # 识别模型实际输入的文本行小图（rotated_crop，与 texts 逐项对齐；无识别处为 None）。供数据回流落盘用。
    text_crops: List = field(default_factory=list)

    def save_img(self, image, save_path):
        for i, point in enumerate(self.Points):
            image = cv2.polylines(image, [point], True, (0, 255, 0), 2)
            image = cv2.putText(
                image, self.texts[i], (point[0][0], point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
            )
        cv2.imwrite(save_path, image)


@dataclass
class PanelInfo:
    """单张图的线标判定结果。"""
    result: bool = False
    product_type: str = ""
    # 标准ocr结果
    standard_result: list[str] = field(default_factory=list)
    # 观察到的ocr结果
    observed_result: list[str] = field(default_factory=list)
    # 观察到的ocr结果的点坐标
    observed_result_points: list[list[float]] = field(default_factory=list)
    message: str = ErrorType.UNKNOWN.value
    error_indexs: list[int] = field(default_factory=list)
    class_id: List[int] = field(default_factory=list)
    confidence: list[float] = field(default_factory=list)
