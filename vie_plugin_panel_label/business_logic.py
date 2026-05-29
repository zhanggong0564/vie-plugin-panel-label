'''
@Author       : gongzhang4
@Date         : 2026-03-02 03:48:53
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 07:48:57
@FilePath     : business_logic.py
@Description  :
'''

from .panel_label_detect import OCRPipeline, PanellabelItem
from schemas import MoMResult, DetectionItem
from schemas.exceptions import ProductNotRegisteredError, ModelInferenceError
from services.api import detection_factory
from services.base import BusinessLogicBase
from utils import vision_logger
from .product_type import PRODUCT_TYPE, PRODUCT_guideline
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum
from dataclasses import field
from .utils import rect_contains


class ErrorType(str, Enum):
    MISSING = "missing"
    EXTRA = "extra"
    MISMATCH = "mismatch"
    UNKNOWN = "unknown"
    OK = "ok"


@dataclass
class PanelInfo:
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


@detection_factory.register("panel_label")
class PanelLabelJudgeApi(BusinessLogicBase):

    def __init__(self, settings):
        super().__init__(settings)
        self.class_name = {
            0: "line",
            1: "QFU",
        }

    def _initialize_model(self, settings):
        from .config import PanelLabelConfig
        cfg = PanelLabelConfig()
        try:
            self.detector = OCRPipeline(
                cfg.model_path,
                cfg.orient_model_path,
                cfg.text_recognition_model_path,
                cfg.confThreshold,
                cfg.nmsThreshold,
                cfg.text_rec_score_thresh,
                cfg.text_rec_input_shape,
                cfg.text_det_limit_side_len,
                cfg.text_det_limit_type,
                cfg.text_det_thresh,
                cfg.text_det_box_thresh,
                cfg.text_det_unclip_ratio,
                cfg.text_det_input_shape,
            )
        except Exception as e:
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError(
                "panel_label 模型加载失败",
                scenario="panel_label",
                original_error=e,
            )

    def guideline_filter(self, results: PanellabelItem, product_type: str, img_w: int, img_h: int):
        norm_rect = PRODUCT_guideline[product_type]
        x_norm, y_norm, w_norm, h_norm = norm_rect
        rect = (int(x_norm * img_w), int(y_norm * img_h), int(w_norm * img_w), int(h_norm * img_h))
        boxes = results.Points
        keep_indices = []
        for i, box in enumerate(boxes):
            all_points_inside = True
            for j in range(0, len(box), 2):
                px, py = box[j], box[j + 1]
                if not rect_contains(rect, (px, py)):
                    all_points_inside = False
                    break
            if all_points_inside:
                keep_indices.append(i)
        filtered_results = PanellabelItem(
            Points=[results.Points[i] for i in keep_indices],
            index=[results.index[i] for i in keep_indices],
            class_id=[results.class_id[i] for i in keep_indices],
            texts=[results.texts[i] for i in keep_indices],
            confidence=[results.confidence[i] for i in keep_indices],
        )
        return filtered_results

    def business_post_process(self, ctx):
        product_type = ctx.product_type
        if product_type not in PRODUCT_TYPE or product_type not in PRODUCT_guideline:
            raise ProductNotRegisteredError(
                f"产品型号 '{product_type}' 未在 panel_label PRODUCT_TYPE 中注册",
                product_type=product_type,
                scenario="panel_label",
            )
        results = self.guideline_filter(ctx.raw_result, product_type, ctx.w, ctx.h)
        panel_info = self.analyze(results, product_type, ctx.rule)
        mom_result = MoMResult()
        mom_result.status = panel_info.result
        mom_result.message = panel_info.message
        data_list = []
        for i, observed_item in enumerate(panel_info.observed_result):
            status = panel_info.result or i not in panel_info.error_indexs
            data_list.append(
                DetectionItem(
                    status=status,
                    scene=self.class_name[panel_info.class_id[i]],
                    coordinate=panel_info.observed_result_points[i],
                    accuracy=panel_info.confidence[i],
                    name=observed_item,
                )
            )
        mom_result.detailList = data_list
        ctx.result = mom_result

    @staticmethod
    def _fix_slash_misrecognition(text: str) -> str:
        """将不成对的括号修正为 / ，解决OCR将 / 误识别成 ( 或 ) 的问题"""
        if text is None:
            return None
        left_count = text.count("(")
        right_count = text.count(")")
        if left_count == right_count:
            return text
        if left_count > right_count:
            excess = left_count - right_count
            chars = list(text)
            for i in range(len(chars) - 1, -1, -1):
                if chars[i] == "(":
                    chars[i] = "/"
                    excess -= 1
                    if excess == 0:
                        break
            return "".join(chars)
        else:
            excess = right_count - left_count
            chars = list(text)
            for i in range(len(chars)):
                if chars[i] == ")":
                    chars[i] = "/"
                    excess -= 1
                    if excess == 0:
                        break
            return "".join(chars)

    @staticmethod
    def _compare_key(text: str, rule: str) -> str:
        if text is None:
            return None
        parts = text.split("/", 1)
        if rule == "front":
            return parts[0].lower()
        elif rule == "back":
            return parts[-1].lower()
        else:  # "all"
            return text.lower()

    def analyze(self, observed_result: PanellabelItem, product_type: str, rule: str = "all") -> PanelInfo:
        standard_result = PRODUCT_TYPE[product_type]
        corrected_texts = [self._fix_slash_misrecognition(t) for t in observed_result.texts]
        panel_info = PanelInfo(
            standard_result=standard_result,
            observed_result=corrected_texts,
            observed_result_points=observed_result.Points,
            class_id=observed_result.class_id,
            confidence=observed_result.confidence,
        )
        panel_info.result = True
        panel_info.message = ErrorType.OK.value

        observed_count = len(panel_info.observed_result)
        standard_count = len(standard_result)

        if observed_count < standard_count:
            panel_info.message = ErrorType.MISSING.value
            panel_info.result = False
            return panel_info
        elif observed_count > standard_count:
            panel_info.message = ErrorType.EXTRA.value
            panel_info.result = False
            return panel_info

        for i, item in enumerate(panel_info.observed_result):
            if self._compare_key(item, rule) != self._compare_key(standard_result[i], rule):
                panel_info.message = ErrorType.MISMATCH.value
                panel_info.result = False
                panel_info.error_indexs.append(i)
        return panel_info
