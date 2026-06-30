"""panel_label utils 工具函数单元测试"""
import sys
import importlib.util
from pathlib import Path

# 直接加载 utils.py 模块，避免 __init__.py 的依赖问题
utils_path = Path(__file__).parent.parent / "vie_plugin_panel_label" / "utils.py"
spec = importlib.util.spec_from_file_location("utils", utils_path)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

polygon_contains = utils.polygon_contains


# 顺时针四角矩形：左上(100,100) 右上(300,100) 右下(300,300) 左下(100,300)
SQUARE = [100, 100, 300, 100, 300, 300, 100, 300]


class TestPolygonContains:
    def test_point_inside_kept(self):
        assert polygon_contains(SQUARE, (200, 200)) is True

    def test_point_outside_excluded(self):
        assert polygon_contains(SQUARE, (50, 50)) is False
        assert polygon_contains(SQUARE, (350, 200)) is False

    def test_border_included_by_default(self):
        assert polygon_contains(SQUARE, (100, 200)) is True

    def test_border_excluded_when_flag_false(self):
        assert polygon_contains(SQUARE, (100, 200), include_border=False) is False

    def test_accepts_point_pairs_shape(self):
        poly = [(100, 100), (300, 100), (300, 300), (100, 300)]
        assert polygon_contains(poly, (200, 200)) is True

    def test_non_axis_aligned_quad(self):
        # 平行四边形：(100,100)(300,150)(280,320)(80,270)
        quad = [100, 100, 300, 150, 280, 320, 80, 270]
        assert polygon_contains(quad, (190, 210)) is True
        assert polygon_contains(quad, (305, 150)) is False
