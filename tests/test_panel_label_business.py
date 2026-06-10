"""panel_label 业务逻辑单元测试"""
import numpy as np
import pytest
from unittest.mock import patch
from schemas.exceptions import InvalidParamsError
from schemas.inference_context import InferenceContext


@pytest.fixture
def api_instance():
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
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
