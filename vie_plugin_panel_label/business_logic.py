'''
@Author       : gongzhang4
@Date         : 2026-03-02 03:48:53
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 07:48:57
@FilePath     : business_logic.py
@Description  :
'''

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
            self.detector = OCRPipeline(
                cfg.orient_metadata_path,
                cfg.text_recognition_metadata_path,
                cfg.confThreshold,
                cfg.nmsThreshold,
                cfg.text_rec_score_thresh,
                cfg.text_orient_score_thresh,
                cfg.text_rec_input_shape,
                dedup_overlap_thresh=cfg.dedup_overlap_thresh,
                cpu_fast_path=cfg.cpu_fast_path,
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

    def guideline_filter(
        self,
        results: PanellabelItem,
        norm_rect,
        img_w: int,
        img_h: int,
    ):
        # 按下发值长度区分引导区域：4 值=轴对齐矩形（旧），8 值=四边形（新）。
        boxes = results.Points
        roi_poly = self._guideline_polygon(norm_rect, img_w, img_h)
        keep_indices = [
            index
            for index, box in enumerate(boxes)
            if polygon_overlap_ratio(box, roi_poly)
            >= self.guideline_overlap_thresh
        ]
        return PanellabelItem(
            Points=[results.Points[index] for index in keep_indices],
            index=[results.index[index] for index in keep_indices],
            class_id=[results.class_id[index] for index in keep_indices],
            texts=[results.texts[index] for index in keep_indices],
            confidence=[results.confidence[index] for index in keep_indices],
            tokens=(
                [results.tokens[index] for index in keep_indices]
                if results.tokens
                else []
            ),
            text_crops=(
                [results.text_crops[index] for index in keep_indices]
                if results.text_crops
                else []
            ),
        )

    @staticmethod
    def _guideline_polygon(norm_rect, img_w: int, img_h: int):
        if len(norm_rect) == 8:
            return [
                value * (img_w if index % 2 == 0 else img_h)
                for index, value in enumerate(norm_rect)
            ]

        x_norm, y_norm, width_norm, height_norm = norm_rect
        x = x_norm * img_w
        y = y_norm * img_h
        width = width_norm * img_w
        height = height_norm * img_h
        return [
            x,
            y,
            x + width,
            y,
            x + width,
            y + height,
            x,
            y + height,
        ]

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
        standard_candidates = self._normalize_standard_candidates(standard_result)
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
        panel_info = self.analyze(ctx.raw_result, standard_candidates, ctx.rule)
        mom_result = MoMResult()
        mom_result.status = panel_info.result
        mom_result.message = panel_info.message
        observed_count = len(panel_info.observed_result)
        standard_counts = [len(candidate) for candidate in standard_candidates]
        if observed_count not in standard_counts:
            vision_logger.warning(
                "panel_label line_order count mismatch, product_type={}, observed_count={}, "
                "candidate_count={}, standard_counts={}, selected_standard_result={}, observed_result={}",
                ctx.product_type,
                observed_count,
                len(standard_candidates),
                standard_counts,
                panel_info.standard_result,
                panel_info.observed_result,
            )
        data_list = []
        for i, observed_item in enumerate(panel_info.observed_result):
            status = panel_info.result or i not in panel_info.error_indexs
            item_name = observed_item
            if item_name is None:
                expected_name = (
                    panel_info.standard_result[i]
                    if i < len(panel_info.standard_result)
                    else None
                )
                vision_logger.warning(
                    "panel_label observed text is None, fallback detailList.name to empty string, "
                    "product_type={}, idx={}, expected_name={}, coordinate={}, confidence={}",
                    ctx.product_type,
                    i,
                    expected_name,
                    (
                        panel_info.observed_result_points[i]
                        if i < len(panel_info.observed_result_points)
                        else None
                    ),
                    (
                        panel_info.confidence[i]
                        if i < len(panel_info.confidence)
                        else None
                    ),
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

    @staticmethod
    def _normalize_standard_candidates(standard_result):
        """兼容内部调用仍传入单个一维标准顺序。"""
        if standard_result and isinstance(standard_result[0], str):
            return [standard_result]
        return standard_result

    def analyze(
        self,
        observed_result: PanellabelItem,
        standard_result,
        rule: str = "all",
    ) -> PanelInfo:
        corrected_texts = [
            self._fix_slash_misrecognition(text)
            for text in observed_result.texts
        ]
        standard_candidates = self._normalize_standard_candidates(standard_result)
        observed_count = len(corrected_texts)
        selected_candidate = standard_candidates[0]
        selected_error_indexs = []
        selected_score = None
        matched = False

        for candidate_index, candidate in enumerate(standard_candidates):
            error_indexs = [
                index
                for index, (observed_item, standard_item) in enumerate(
                    zip(corrected_texts, candidate)
                )
                if self._compare_key(observed_item, rule)
                != self._compare_key(standard_item, rule)
            ]
            count_gap = abs(observed_count - len(candidate))
            if count_gap == 0 and not error_indexs:
                selected_candidate = candidate
                selected_error_indexs = []
                matched = True
                break

            score = (
                0 if count_gap == 0 else 1,
                count_gap,
                len(error_indexs),
                candidate_index,
            )
            if selected_score is None or score < selected_score:
                selected_candidate = candidate
                selected_error_indexs = error_indexs
                selected_score = score

        panel_info = PanelInfo(
            standard_result=selected_candidate,
            observed_result=corrected_texts,
            observed_result_points=observed_result.Points,
            class_id=observed_result.class_id,
            confidence=observed_result.confidence,
        )
        if matched:
            panel_info.result = True
            panel_info.message = ErrorType.OK.value
            return panel_info

        standard_count = len(selected_candidate)
        if observed_count < standard_count:
            panel_info.message = ErrorType.MISSING.value
            return panel_info
        if observed_count > standard_count:
            panel_info.message = ErrorType.EXTRA.value
            return panel_info

        panel_info.message = ErrorType.MISMATCH.value
        panel_info.error_indexs = selected_error_indexs
        return panel_info
