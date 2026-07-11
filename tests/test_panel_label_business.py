"""panel_label 业务逻辑单元测试"""
import sys
import types
import numpy as np
import pytest
from unittest.mock import patch
from schemas.exceptions import InvalidParamsError
from schemas.inference_context import InferenceContext

paddleocr = types.ModuleType("paddleocr")
paddleocr.TextDetection = object
paddleocr.TextLineOrientationClassification = object
paddleocr.TextRecognition = object
paddlex = types.ModuleType("paddlex")
paddlex_inference = types.ModuleType("paddlex.inference")
paddlex_pipelines = types.ModuleType("paddlex.inference.pipelines")
paddlex_components = types.ModuleType("paddlex.inference.pipelines.components")
paddlex_components.CropByPolys = object
sys.modules.setdefault("paddleocr", paddleocr)
sys.modules.setdefault("paddlex", paddlex)
sys.modules.setdefault("paddlex.inference", paddlex_inference)
sys.modules.setdefault("paddlex.inference.pipelines", paddlex_pipelines)
sys.modules.setdefault("paddlex.inference.pipelines.components", paddlex_components)


@pytest.fixture
def api_instance(monkeypatch):
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
    monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
    monkeypatch.setitem(sys.modules, "paddlex", paddlex)
    monkeypatch.setitem(sys.modules, "paddlex.inference", paddlex_inference)
    monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines", paddlex_pipelines)
    monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines.components", paddlex_components)
    with patch("vie_plugin_panel_label.business_logic.OCRPipeline"):
        from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
        from config import settings
        yield PanelLabelJudgeApi(settings)


def _make_ctx(result, product_type, w=1000, h=1000, rule="all", extra=None):
    ctx = InferenceContext(image=np.zeros((h, w, 3), dtype=np.uint8), h=h, w=w,
                           product_type=product_type, rule=rule, extra=extra or {})
    ctx.raw_result = result
    return ctx


def test_model_initialization_uses_direct_ocr_contract():
    from config import settings
    from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi

    with patch("vie_plugin_panel_label.business_logic.OCRPipeline") as pipeline:
        PanelLabelJudgeApi(settings)

    pipeline.assert_called_once_with(
        "./weights/panel_label/v2/best.onnx",
        "./weights/panel_label/v2/textline_ori_lcnet_v2",
        "./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr",
        0.6,
        0.8,
        0.7,
        0.9,
        None,
        dedup_overlap_thresh=0.6,
    )


class TestRequestParamsValidation:
    def test_missing_line_order_and_guideline_raises(self, api_instance):
        """standard_result / guideline 由请求下发，缺失时报参数错误"""
        from vie_plugin_panel_label.models import PanellabelItem
        ctx = _make_ctx(PanellabelItem(), "TK2")  # extra 为空，未携带判定基准
        with pytest.raises(InvalidParamsError) as exc_info:
            api_instance.business_post_process(ctx)
        assert "line_order" in exc_info.value.error_msg
        assert exc_info.value.context.get("scenario") == "panel_label"

    def test_guideline_coordinates_required_in_schema(self):
        """默认契约下调用方必须传 guideline_coordinates。"""
        from pydantic import ValidationError
        from vie_plugin_panel_label.schemas import ModelParams

        with pytest.raises(ValidationError):
            ModelParams(product_type="TK2", line_order="TK2-2,TK2-1")

        mp = ModelParams(
            product_type="TK2",
            line_order="TK2-2,TK2-1",
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )
        assert mp.guideline_coordinates == (0.1, 0.2, 0.3, 0.4)

    def test_guideline_8_values_parsed(self):
        """8 值四边形可解析为长度 8 元组"""
        from vie_plugin_panel_label.schemas import ModelParams
        mp = ModelParams(
            product_type="SCUJ2",
            line_order="A,B",
            guideline_coordinates="0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9",
        )
        assert mp.guideline_coordinates == (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9)

    def test_guideline_4_values_still_parsed(self):
        """回归：4 值矩形仍照常解析"""
        from vie_plugin_panel_label.schemas import ModelParams
        mp = ModelParams(
            product_type="TK2",
            line_order="TK2-2,TK2-1",
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )
        assert mp.guideline_coordinates == (0.1, 0.2, 0.3, 0.4)

    def test_guideline_invalid_length_raises(self):
        """非 4/8 长度（如 6 值）报校验错误"""
        from pydantic import ValidationError
        from vie_plugin_panel_label.schemas import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(
                product_type="SCUJ2",
                line_order="A,B",
                guideline_coordinates="0.1,0.2,0.3,0.4,0.5,0.6",
            )


class TestCompareKeyNormalization:
    def test_zero_o_confusion_not_mismatch(self, api_instance):
        """线标字体下 OCR 区分不了 O/0（TCU-DO1 常读成 TCU-D01），比对须归一"""
        key = api_instance._compare_key
        assert key("TCU-D01-2", "all") == key("TCU-DO1-2", "all")
        assert key("TCU-reader-GND", "all") == key("TCU-Reader-GND", "all")

    def test_front_back_rules_still_split_on_slash(self, api_instance):
        key = api_instance._compare_key
        assert key("QF2-1/PE1-J1", "front") == "qf2-1"
        assert key("FU34-2/KM1-O1", "back") == "km1-01"


class TestDetailNameFallback:
    def test_unrecognized_ocr_text_does_not_emit_none_name(self, api_instance):
        from vie_plugin_panel_label.models import PanellabelItem

        raw = PanellabelItem(
            Points=[[0, 0, 10, 0, 10, 10, 0, 10]],
            index=[0],
            class_id=[0],
            texts=[None],
            confidence=[0.8],
        )
        ctx = _make_ctx(
            raw,
            "TK2",
            extra={"standard_result": ["TK2-1"], "guideline": (0.0, 0.0, 1.0, 1.0)},
        )

        api_instance.business_post_process(ctx)

        assert ctx.result.detailList[0].name == ""
        assert isinstance(ctx.result.detailList[0].name, str)
