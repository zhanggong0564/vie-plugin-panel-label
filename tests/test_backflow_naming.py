"""线标数据回流命名规则单元测试

线标专属的文件名解析（AI- 前缀剥离、型号图片序号去除、中文场景/型号/时间戳切分）
随框架插件化下沉到本插件 Router；框架基类只保留场景无关的默认落盘命名。
"""
import pytest

from routers.base_router import UNKNOWN_MODEL_DIR
from vie_plugin_panel_label.plugin import PanelLabelRouter


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
    "filename, expected",
    [
        # AI- 前缀剥掉、型号序号 -1 去掉
        ("AI-中压线标检验TK2-1-1764780181920.jpg", ("中压线标检验", "TK2", "1764780181920")),
        # 无 AI- 前缀，效果一致
        ("中压线标检验TK2-1-1764780181920.jpg", ("中压线标检验", "TK2", "1764780181920")),
        # 旧式：型号尾部非数字（-A）不应被当序号去掉
        ("1+X线标检验PE1-A-1779526099406.jpg", ("1+X线标检验", "PE1-A", "1779526099406")),
        # 型号本身无序号后缀
        ("中压线标检验TK2-1764780181920.jpg", ("中压线标检验", "TK2", "1764780181920")),
        # AI- 大小写不敏感
        ("ai-中压线标检验TK2-2-1764780181920.png", ("中压线标检验", "TK2", "1764780181920")),
    ],
)
def test_parse_filename(filename, expected):
    """文件名解析：AI- 前缀剥离、型号图片序号去除、场景/型号/时间戳切分。"""
    assert PanelLabelRouter._parse_filename(filename) == expected


@pytest.mark.parametrize("filename", ["random.jpg", "noscene-123.jpg", "纯中文.jpg"])
def test_parse_filename_unparseable(filename):
    """不符合规则的文件名返回三元 None，由调用方走兜底。"""
    assert PanelLabelRouter._parse_filename(filename) == (None, None, None)


def test_resolve_target_uses_parsed_scene_and_model(router):
    """解析成功：场景目录取中文场景名、型号目录取聚合型号、文件名用时间戳。"""
    target = router.resolve_backflow_target(
        "AI-中压线标检验TK2-1-1764780181920.jpg", fallback_product_type="ignored"
    )
    assert target.scene_dir == "中压线标检验"
    assert target.model_dir == "TK2"
    assert target.save_stem == "1764780181920"


def test_resolve_target_falls_back_to_framework_default(router):
    """文件名不可解析时回退框架默认：场景=detector_type，型号取 product_type 兜底。"""
    target = router.resolve_backflow_target("random.jpg", fallback_product_type="FU211")
    assert target.scene_dir == "panel_label"
    assert target.model_dir == "FU211"
    assert target.save_stem == "random"


def test_resolve_target_fallback_unknown_model(router):
    """既不可解析又无 product_type 时型号目录回退 _unknown_model。"""
    target = router.resolve_backflow_target("random.jpg", fallback_product_type=None)
    assert target.model_dir == UNKNOWN_MODEL_DIR
