"""panel_label 业务逻辑单元测试"""
import sys
import types
import numpy as np
import pytest
from unittest.mock import patch
from schemas.exceptions import InvalidParamsError
from schemas.inference_context import InferenceContext


@pytest.fixture
def api_instance(monkeypatch):
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
    paddleocr = types.ModuleType("paddleocr")
    paddleocr.TextDetection = object
    paddleocr.TextLineOrientationClassification = object
    paddleocr.TextRecognition = object
    paddlex = types.ModuleType("paddlex")
    paddlex_inference = types.ModuleType("paddlex.inference")
    paddlex_pipelines = types.ModuleType("paddlex.inference.pipelines")
    paddlex_components = types.ModuleType("paddlex.inference.pipelines.components")
    paddlex_components.CropByPolys = object
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


class TestRequestParamsValidation:
    def test_missing_line_order_and_guideline_raises(self, api_instance):
        """standard_result / guideline 由请求下发，缺失时报参数错误"""
        from vie_plugin_panel_label.models import PanellabelItem
        ctx = _make_ctx(PanellabelItem(), "TK2")  # extra 为空，未携带判定基准
        with pytest.raises(InvalidParamsError) as exc_info:
            api_instance.business_post_process(ctx)
        assert "line_order" in exc_info.value.error_msg
        assert exc_info.value.context.get("scenario") == "panel_label"

    def test_guideline_coordinates_optional_in_schema(self):
        """关闭 guideline 过滤的部署下调用方可不传 guideline_coordinates"""
        from vie_plugin_panel_label.schemas import ModelParams
        mp = ModelParams(product_type="TK2", line_order="TK2-2,TK2-1")
        assert mp.guideline_coordinates is None
        mp2 = ModelParams(
            product_type="TK2",
            line_order="TK2-2,TK2-1",
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )
        assert mp2.guideline_coordinates == (0.1, 0.2, 0.3, 0.4)


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


class TestOCRPipelineCompatibility:
    def test_infer_accepts_legacy_two_value_points_to_mask(self, monkeypatch):
        """兼容旧版 Points_to_Mask(return_maps=True) 仍只返回 roi 和排序索引。"""
        import sys
        import types

        paddleocr = types.ModuleType("paddleocr")
        paddleocr.TextDetection = object
        paddleocr.TextLineOrientationClassification = object
        paddleocr.TextRecognition = object
        paddlex = types.ModuleType("paddlex")
        paddlex_inference = types.ModuleType("paddlex.inference")
        paddlex_pipelines = types.ModuleType("paddlex.inference.pipelines")
        paddlex_components = types.ModuleType("paddlex.inference.pipelines.components")
        paddlex_components.CropByPolys = object
        monkeypatch.setitem(sys.modules, "paddleocr", paddleocr)
        monkeypatch.setitem(sys.modules, "paddlex", paddlex)
        monkeypatch.setitem(sys.modules, "paddlex.inference", paddlex_inference)
        monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines", paddlex_pipelines)
        monkeypatch.setitem(sys.modules, "paddlex.inference.pipelines.components", paddlex_components)

        from vie_plugin_panel_label import panel_label_detect as detect_mod
        from vie_plugin_panel_label.panel_label_detect import OCRPipeline
        from schemas.data_base import DetectResult

        image = np.zeros((20, 20, 3), dtype=np.uint8)
        polygon = [[0, 0], [10, 0], [10, 10], [0, 10]]
        pipeline = OCRPipeline.__new__(OCRPipeline)
        pipeline.dedup_overlap_thresh = 1.0
        pipeline.detect_model = type(
            "Detect",
            (),
            {
                "infer": lambda self, img: DetectResult(
                    boxes=[[0, 0, 10, 10]],
                    scores=[0.9],
                    class_ids=[0],
                    class_names=["line"],
                    mask_polygons=[polygon],
                )
            },
        )()
        pipeline.text_orient_model = type(
            "Orient",
            (),
            {"predict": lambda self, crops: [{"class_ids": [0]} for _ in crops]},
        )()
        pipeline.text_rec_model = type(
            "Rec",
            (),
            {"predict": lambda self, crops: [{"rec_text": "TK2-1", "rec_score": 0.95} for _ in crops]},
        )()
        pipeline.text_rec_score_thresh = 0.7

        monkeypatch.setattr(
            detect_mod,
            "Points_to_Mask",
            lambda img, pts, return_maps=False: ([image], np.array([0], dtype=np.int64)),
        )

        result = pipeline.infer(image)

        assert result.texts == ["TK2-1"]
        assert len(result.Points) == 1
        assert len(result.Points[0]) == 8


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
