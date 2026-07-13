"""线标数据回流命名规则单元测试

线标专属落盘命名随框架插件化下沉到本插件 Router：
  - 场景目录 = 原始文件名按 '-' 分割的第一段
  - 型号目录 = 请求 product_type，空则取 AICameraModel.AIParameterValue，仍空回退 _unknown_model
  - 落盘文件名 = 原始文件名最后一个 '-' 后的片段（通常为时间戳）
框架基类只保留场景无关的默认落盘命名。
"""
import pytest

from routers.base_router import UNKNOWN_MODEL_DIR
from vie_plugin_panel_label.plugin import PanelLabelRouter
from vie_plugin_panel_label.schemas import PanelLabelRequest


@pytest.fixture
def router():
    return PanelLabelRouter(
        router_name="panel_router",
        api_path="/panel_label_detect",
        summary="t",
        description="t",
        detector_type="panel_label",
    )


@pytest.mark.parametrize(
    "filename, expected_scene, expected_stem",
    [
        # 顶层场景取 '-' 首段，文件名取最后一段时间戳
        ("AI-集中式-1764780181920.jpg", "AI", "1764780181920"),
        ("集中式-中压线标检验-1764780181920.png", "集中式", "1764780181920"),
        (
            "集中式-SG1100UD-AI拍照-交流侧-6-1-1782886643866.jpg",
            "集中式",
            "1782886643866",
        ),
        # 无 '-' 时整名即首段
        ("纯中文.jpg", "纯中文", "纯中文"),
    ],
)
def test_resolve_target_scene_and_stem(router, filename, expected_scene, expected_stem):
    """场景目录取首段、文件名取末段（型号由 product_type 决定）。"""
    target = router.resolve_backflow_target(filename, fallback_product_type="TK2")
    assert target.scene_dir == expected_scene
    assert target.save_stem == expected_stem
    assert target.model_dir == "TK2"


def test_resolve_target_model_unknown_when_no_product_type(router):
    """既无 product_type 兜底时型号目录回退 _unknown_model。"""
    target = router.resolve_backflow_target("AI-集中式-123.jpg", fallback_product_type=None)
    assert target.model_dir == UNKNOWN_MODEL_DIR


def _make_request(product_type="", ai_param_value=None):
    payload = {
        "product": "p",
        "type": "t",
        "modelParams": {
            "product_type": product_type,
            "line_order": "TK2-1,TK2-2",
            "guideline_coordinates": [0, 0, 100, 100],
        },
    }
    if ai_param_value is not None:
        payload["AICameraModel"] = [
            {"AIParameterValue": "", "Id": "1"},
            {"AIParameterValue": ai_param_value, "Id": "2"},
        ]
    return PanelLabelRequest(**payload)


def test_extract_product_type_prefers_product_type():
    """product_type 非空时直接用之。"""
    req = _make_request(product_type="QF2", ai_param_value="SHOULD_NOT_USE")
    assert PanelLabelRouter._extract_product_type(req) == "QF2"


def test_extract_product_type_falls_back_to_ai_parameter_value():
    """product_type 为空字符时取首个非空 AIParameterValue（AICameraModel 经 extra 透传）。"""
    req = _make_request(product_type="", ai_param_value="ABB-100")
    assert PanelLabelRouter._extract_product_type(req) == "ABB-100"


def test_extract_product_type_none_when_both_missing():
    """product_type 空且无 AICameraModel 时返回 None，交框架回退 _unknown_model。"""
    req = _make_request(product_type="")
    assert PanelLabelRouter._extract_product_type(req) is None
