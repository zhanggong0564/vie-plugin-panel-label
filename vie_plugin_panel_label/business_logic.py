'''
@Author       : gongzhang4
@Date         : 2026-03-02 03:48:53
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 07:48:57
@FilePath     : business_logic.py
@Description  :
'''

from .panel_label_detect import OCRPipeline
from .models import ErrorType, PanelInfo, PanellabelItem
from schemas import MoMResult, DetectionItem
from schemas.exceptions import InvalidParamsError, ModelInferenceError
from services.api import detection_factory
from services.base import BusinessLogicBase
from utils import vision_logger
from .utils import rect_contains
from .ordering import order_panel_item
from .product_type import get_sort_mode


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
                cfg.text_det_model_path,
                cfg.text_det_limit_side_len,
                cfg.text_det_limit_type,
                cfg.text_det_thresh,
                cfg.text_det_box_thresh,
                cfg.text_det_unclip_ratio,
                cfg.text_det_input_shape,
                dedup_overlap_thresh=cfg.dedup_overlap_thresh,
            )
        except Exception as e:
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError(
                "panel_label 模型加载失败",
                scenario="panel_label",
                original_error=e,
            )

    def guideline_filter(self, results: PanellabelItem, norm_rect, img_w: int, img_h: int):
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
            text_det_points=[results.text_det_points[i] for i in keep_indices] if results.text_det_points else [],
            text_crops=[results.text_crops[i] for i in keep_indices] if results.text_crops else [],
        )
        return filtered_results

    def business_post_process(self, ctx):
        # 标准顺序与引导框由请求经 ctx.extra 下发，不再从本地词典读取。
        standard_result = ctx.extra.get("standard_result")
        norm_rect = ctx.extra.get("guideline")
        if not standard_result or not norm_rect:
            raise InvalidParamsError(
                "panel_label 缺少 line_order 或 guideline_coordinates 参数",
                product_type=ctx.product_type,
                scenario="panel_label",
            )
        results = self.guideline_filter(ctx.raw_result, norm_rect, ctx.w, ctx.h)
        # 按型号固定排序模式对线标重排（消除运行时猜布局/调阈值）。
        ctx.raw_result = order_panel_item(results, get_sort_mode(ctx.product_type))
        panel_info = self.analyze(ctx.raw_result, standard_result, ctx.rule)
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

    def analyze(self, observed_result: PanellabelItem, standard_result, rule: str = "all") -> PanelInfo:
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
