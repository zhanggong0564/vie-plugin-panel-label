"""panel_label 业务逻辑单元测试"""
import numpy as np
import pytest
from unittest.mock import patch
from schemas.exceptions import ProductNotRegisteredError
from schemas.inference_context import InferenceContext


@pytest.fixture
def api_instance():
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
    with patch("vie_plugin_panel_label.business_logic.OCRPipeline"):
        from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
        from config import settings
        yield PanelLabelJudgeApi(settings)


def _make_ctx(result, product_type, w=1000, h=1000, rule="all"):
    ctx = InferenceContext(image=np.zeros((h, w, 3), dtype=np.uint8), h=h, w=w,
                           product_type=product_type, rule=rule)
    ctx.raw_result = result
    return ctx


class TestProductTypeValidation:
    def test_unregistered_product_type_raises(self, api_instance):
        from vie_plugin_panel_label.panel_label_detect import PanellabelItem
        ctx = _make_ctx(PanellabelItem(), "NON_EXISTENT_TYPE_999")
        with pytest.raises(ProductNotRegisteredError) as exc_info:
            api_instance.business_post_process(ctx)
        assert "NON_EXISTENT_TYPE_999" in exc_info.value.error_msg
        assert exc_info.value.context.get("product_type") == "NON_EXISTENT_TYPE_999"
        assert exc_info.value.context.get("scenario") == "panel_label"
