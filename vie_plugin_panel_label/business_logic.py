'''
@Author       : gongzhang4
@Date         : 2026-03-02 03:48:53
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 07:48:57
@FilePath     : business_logic.py
@Description  :
'''

from pathlib import Path

from .config import PanelLabelConfig
from .panel_label_detect import OCRPipeline
from .models import ErrorType, PanelInfo, PanellabelItem
from schemas import MoMResult, DetectionItem
from schemas.exceptions import InvalidParamsError, ModelInferenceError
from services.scenario_registry import scenario_registry
from services.base import BusinessLogicBase
from services.inference import (
    OnnxRuntimeOptions,
    RunnerSpec,
    create_inference_runner,
)
from utils import vision_logger
from .utils import polygon_overlap_ratio
from .ordering import order_panel_item
from .product_type import get_sort_mode


@scenario_registry.register("panel_label")
class PanelLabelJudgeApi(BusinessLogicBase):

    def __init__(self, settings):
        super().__init__(settings)
        cfg = PanelLabelConfig()
        self.enable_guideline_filter = cfg.enable_guideline_filter
        self.guideline_overlap_thresh = cfg.guideline_overlap_thresh
        self.class_name = {
            0: "line",
            1: "QFU",
        }

    def _initialize_model(self, settings):
        cfg = PanelLabelConfig()
        created_runners = []
        try:
            onnx_options = OnnxRuntimeOptions.from_settings(settings)
            detection_runner = create_inference_runner(
                RunnerSpec(
                    scenario="panel_label",
                    onnx_path=cfg.model_path,
                ),
                onnx_options,
            )
            created_runners.append(detection_runner)
            orientation_runner = create_inference_runner(
                RunnerSpec(
                    scenario="panel_label",
                    onnx_path=cfg.orient_model_path,
                ),
                onnx_options,
            )
            created_runners.append(orientation_runner)
            recognition_runner = create_inference_runner(
                RunnerSpec(
                    scenario="panel_label",
                    onnx_path=cfg.text_recognition_model_path,
                ),
                OnnxRuntimeOptions.from_settings(
                    settings, execution_mode="sequential"
                ),
            )
            created_runners.append(recognition_runner)
            orientation_metadata_path = (
                Path(cfg.orient_model_path).with_suffix("") / "inference.yml"
            )
            recognition_metadata_path = (
                Path(cfg.text_recognition_model_path).with_suffix("")
                / "inference.yml"
            )
            self.detector = OCRPipeline(
                str(orientation_metadata_path),
                str(recognition_metadata_path),
                cfg.confThreshold,
                cfg.nmsThreshold,
                cfg.text_rec_score_thresh,
                cfg.text_orient_score_thresh,
                cfg.text_rec_input_shape,
                dedup_overlap_thresh=cfg.dedup_overlap_thresh,
                detection_runner=detection_runner,
                orientation_runner=orientation_runner,
                recognition_runner=recognition_runner,
            )
        except Exception as e:
            for runner in created_runners:
                try:
                    runner.close()
                except Exception as close_error:
                    vision_logger.warning(
                        f"panel_label 初始化回滚清理失败: {close_error}"
                    )
            vision_logger.error(f"initialize model failed, error: {e}")
            raise ModelInferenceError(
                "panel_label 模型加载失败",
                scenario="panel_label",
                original_error=e,
            )

    def guideline_filter(self, results: PanellabelItem, norm_rect, img_w: int, img_h: int):
        # 按下发值长度区分引导区域：4 值=轴对齐矩形（旧），8 值=四边形（新）。
        boxes = results.Points
        if len(norm_rect) == 8:
            roi_poly = [
                norm_rect[i] * (img_w if i % 2 == 0 else img_h)
                for i in range(8)
            ]
        else:
            x_norm, y_norm, w_norm, h_norm = norm_rect
            x = x_norm * img_w
            y = y_norm * img_h
            w = w_norm * img_w
            h = h_norm * img_h
            roi_poly = [
                x, y,
                x + w, y,
                x + w, y + h,
                x, y + h,
            ]
        keep_indices = [
            i for i, box in enumerate(boxes)
            if polygon_overlap_ratio(box, roi_poly) >= self.guideline_overlap_thresh
        ]
        filtered_results = PanellabelItem(
            Points=[results.Points[i] for i in keep_indices],
            index=[results.index[i] for i in keep_indices],
            class_id=[results.class_id[i] for i in keep_indices],
            texts=[results.texts[i] for i in keep_indices],
            confidence=[results.confidence[i] for i in keep_indices],
            text_crops=[results.text_crops[i] for i in keep_indices] if results.text_crops else [],
        )
        return filtered_results

    def business_post_process(self, ctx):
        # 标准顺序与引导框由请求经 ctx.extra 下发，不再从本地词典读取。
        standard_result = ctx.extra.get("standard_result")
        norm_rect = ctx.extra.get("guideline")
        if not standard_result:
            raise InvalidParamsError(
                "panel_label 缺少 line_order 参数",
                product_type=ctx.product_type,
                scenario="panel_label",
            )
        if self.enable_guideline_filter:
            # 开关开启时 guideline 仍为必要参数；关闭时跳过 ROI 过滤，参数可缺省。
            if not norm_rect:
                raise InvalidParamsError(
                    "panel_label 缺少 guideline_coordinates 参数",
                    product_type=ctx.product_type,
                    scenario="panel_label",
                )
            results = self.guideline_filter(ctx.raw_result, norm_rect, ctx.w, ctx.h)
        else:
            results = ctx.raw_result
        # 按型号固定排序模式对线标重排（消除运行时猜布局/调阈值）。
        ctx.raw_result = order_panel_item(results, get_sort_mode(ctx.product_type))
        panel_info = self.analyze(ctx.raw_result, standard_result, ctx.rule)
        mom_result = MoMResult()
        mom_result.status = panel_info.result
        mom_result.message = panel_info.message
        observed_count = len(panel_info.observed_result)
        standard_count = len(standard_result)
        if observed_count != standard_count:
            vision_logger.warning(
                "panel_label line_order count mismatch, product_type={}, observed_count={}, standard_count={}, "
                "standard_result={}, observed_result={}",
                ctx.product_type,
                observed_count,
                standard_count,
                standard_result,
                panel_info.observed_result,
            )
        data_list = []
        for i, observed_item in enumerate(panel_info.observed_result):
            status = panel_info.result or i not in panel_info.error_indexs
            item_name = observed_item
            if item_name is None:
                expected_name = standard_result[i] if i < len(standard_result) else None
                vision_logger.warning(
                    "panel_label observed text is None, fallback detailList.name to empty string, "
                    "product_type={}, idx={}, expected_name={}, coordinate={}, confidence={}",
                    ctx.product_type,
                    i,
                    expected_name,
                    panel_info.observed_result_points[i] if i < len(panel_info.observed_result_points) else None,
                    panel_info.confidence[i] if i < len(panel_info.confidence) else None,
                )
                item_name = ""
            elif not isinstance(item_name, str):
                vision_logger.warning(
                    "panel_label observed text is not str, convert detailList.name to string, "
                    "product_type={}, idx={}, name_type={}, name={}",
                    ctx.product_type,
                    i,
                    type(item_name).__name__,
                    item_name,
                )
                item_name = str(item_name)
            data_list.append(
                DetectionItem(
                    status=status,
                    scene=self.class_name[panel_info.class_id[i]],
                    coordinate=panel_info.observed_result_points[i],
                    accuracy=panel_info.confidence[i],
                    name=item_name,
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
            key = parts[0]
        elif rule == "back":
            key = parts[-1]
        else:  # "all"
            key = text
        # 线标字体下 OCR 区分不了字母 O 与数字 0（TCU-DO1 常读成 TCU-D01），统一归 0 比对
        return key.lower().replace("o", "0")

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
