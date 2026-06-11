"""panel_label 线标排序（按型号固定 sort_mode）单元测试。

回归点：识别正确但顺序错（斜排被误并成一行、二维多列布局、列内方向反常）。
直接按文件加载 ordering/product_type 两个纯模块，避免触发插件包 __init__
（其会 import services 等重依赖）。
"""
import importlib.util
import os
import random

import pytest

_PLUGIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "vie_plugin_panel_label",
)


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PLUGIN_DIR, filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_ordering = _load("pl_ordering", "ordering.py")
_pt = _load("pl_pt", "product_type.py")
compute_order = _ordering.compute_order
get_sort_mode = _pt.get_sort_mode


def _box(x, y, w=30, h=20):
    """以 (x,y) 为中心造一个展平的四点框 [x1,y1,...x4,y4]。"""
    return [x - w / 2, y - h / 2, x + w / 2, y - h / 2, x + w / 2, y + h / 2, x - w / 2, y + h / 2]


def _order_names(named_pts, sort_mode):
    """named_pts: [(name, x, y), ...]，返回按 sort_mode 排序后的名字序。"""
    names = [n for n, _, _ in named_pts]
    pts = [_box(x, y) for _, x, y in named_pts]
    perm = compute_order(pts, sort_mode)
    return [names[i] for i in perm]


def test_kvm_diagonal_real_coords_default_linear():
    """KVM 扇形斜排（log 真实坐标）：默认 linear 按上→下排成 [4,2,1]。"""
    pts = [
        [0.3273, 0.52275, 0.36433, 0.50475, 0.53533, 0.703, 0.498, 0.721],   # K2-1
        [0.35033, 0.51875, 0.388, 0.4955, 0.59866, 0.6865, 0.56066, 0.70975],  # K2-2
        [0.37066, 0.4975, 0.38666, 0.4645, 0.647, 0.536, 0.63066, 0.569],    # K2-4
    ]
    names = ["K2-1", "K2-2", "K2-4"]
    assert get_sort_mode("KVM") == "linear"
    perm = compute_order(pts, get_sort_mode("KVM"))
    assert [names[i] for i in perm] == ["K2-4", "K2-2", "K2-1"]


def test_qf2_rows_real_coords():
    """QF2：rows:2 —— EXIF 旋转后模型坐标系是上下两行（非左右两列）。

    坐标取自真实推理 log（demo/.../QF2/1776223816260.jpg），两束线标在 y 上分成
    上行(y≈880)、下行(y≈2800)，行内按 x 左→右即为 standard 顺序。历史误设
    columns:2:rowrev 会按 x 把每行劈成两半导致错排。
    """
    named_pts = [
        ("QF2-1/PE1-J1", 762, 876), ("QF2-3/PE1-J3", 931, 874), ("QF2-5/PE1-J5", 1126, 885),
        ("FU34-2/KM1-1", 1448, 949), ("FU35-2/KM1-3", 1708, 896), ("FU36-2/KM1-5", 1998, 887),
        ("QF2-2/T1-38V-a", 604, 2863), ("QF2-4/T1-38V-b", 845, 2881), ("QF2-6/T1-38V-c", 1001, 2894),
        ("FU34-1/QS2-OUT+2", 1457, 2720), ("FU35-1/QS2-OUT-3", 1629, 2698), ("FU36-1/QS2-OUT+3", 1974, 2767),
    ]
    random.seed(1)
    random.shuffle(named_pts)
    assert get_sort_mode("QF2") == "rows:2"
    assert _order_names(named_pts, get_sort_mode("QF2")) == _pt.PRODUCT_TYPE["QF2"]


def test_tcu_rows_top_pair_height_jitter():
    """TCU：rows:2 —— 顶部电源 2 根 + 底部 Reader/DO 8 根。

    坐标取自真实误报图（FCU1202，原图 3000x4000）：顶部右侧 12V- 标签比左侧
    12V+ 悬挂高约 27px。linear 主轴近竖直(u≈(0.16,0.99))，x 差 89px 仅折合
    14px 投影差，被 y 高低差压过 → 12V- 排到 12V+ 前面产生误报；rows:2 行内
    按 x 排，不受悬挂高低影响。
    """
    named_pts = [
        ("TCU-12V+", 859, 330), ("TCU-12V-", 948, 303),
        ("TCU-Reader-RX", 878, 3290), ("TCU-Reader-TX", 932, 3262),
        ("TCU-Reader-GND", 992, 3244), ("TCU-Reader-5V", 1059, 3230),
        ("TCU-DO1-2", 1567, 3392), ("TCU-DO1-1", 1629, 3383),
        ("TCU-DO2-2", 1699, 3370), ("TCU-DO2-1", 1772, 3357),
    ]
    random.seed(3)
    random.shuffle(named_pts)
    assert get_sort_mode("TCU") == "rows:2"
    assert _order_names(named_pts, get_sort_mode("TCU")) == _pt.PRODUCT_TYPE["TCU"]
    # 回归锁定：默认 linear 在该布局下顶部两根会翻转（这是登记 rows:2 的原因）
    ordered_linear = _order_names(sorted(named_pts), "linear")
    assert ordered_linear[:2] == ["TCU-12V-", "TCU-12V+"]


def _two_rows(xs):
    """上(T, y=100)/下(B, y=400)两行各取 xs 横坐标，名字形如 T100/B400。"""
    return [(f"{r}{x}", x, y) for r, y in [("T", 100), ("B", 400)] for x in xs]


def test_rows_two_bands_top_first_left_to_right():
    """rows:2 —— 按 y 间隙分上/下两行（上行先），行内左→右。"""
    items = _two_rows(range(100, 700, 100))
    random.seed(2)
    random.shuffle(items)
    assert _order_names(items, "rows:2") == [
        "T100", "T200", "T300", "T400", "T500", "T600",
        "B100", "B200", "B300", "B400", "B500", "B600",
    ]


def test_rows_rowrev_bottom_first():
    """rows:2:rowrev —— 行序反向（下行先），行内仍左→右。"""
    items = _two_rows((100, 200, 300))
    assert _order_names(items, "rows:2:rowrev") == ["B100", "B200", "B300", "T100", "T200", "T300"]


def test_rows_colrev_right_to_left_in_band():
    """rows:2:colrev —— 行内反向（右→左），行序仍上→下。"""
    items = _two_rows((100, 200, 300))
    assert _order_names(items, "rows:2:colrev") == ["T300", "T200", "T100", "B300", "B200", "B100"]


def test_d1_columns_top_to_bottom():
    """D1 二维散布：columns:2 —— 左列(D1+/D1-) 再 中列(D1-1/D1-3)，列内上→下。"""
    d1 = [("D1-1", 0.48, 0.36), ("D1-3", 0.50, 0.78), ("D1+", 0.19, 0.55), ("D1-", 0.18, 0.66)]
    assert get_sort_mode("D1") == "columns:2"
    assert _order_names(d1, get_sort_mode("D1")) == ["D1+", "D1-", "D1-1", "D1-3"]


def test_columns_colrev_right_first():
    """columns:2:colrev —— 右列先、列内上→下。"""
    items = [(f"{c}{y}", x, y) for c, x in [("L", 100), ("R", 400)] for y in (100, 200, 300)]
    assert _order_names(items, "columns:2:colrev") == ["R100", "R200", "R300", "L100", "L200", "L300"]


def test_default_linear_single_row_left_to_right():
    """未登记型号走默认 linear：单行按主轴(偏水平)左→右。"""
    row = [(f"x{x}", x, 300) for x in (500, 100, 300, 200, 400)]
    assert get_sort_mode("UNKNOWN") == "linear"
    assert _order_names(row, get_sort_mode("UNKNOWN")) == ["x100", "x200", "x300", "x400", "x500"]


def test_linear_single_column_top_to_bottom():
    """竖直单列 linear：主轴偏竖直，按上→下。"""
    col = [(f"y{y}", 200, y) for y in (400, 100, 300, 200)]
    assert _order_names(col, "linear") == ["y100", "y200", "y300", "y400"]


def test_linear_rev():
    """linear:rev 整体反向。"""
    row = [(f"x{x}", x, 300) for x in (300, 100, 200)]
    assert _order_names(row, "linear:rev") == ["x300", "x200", "x100"]


@pytest.mark.parametrize("pts", [[], [_box(100, 100)]])
def test_degenerate_sizes(pts):
    """空 / 单元素不报错，原样返回。"""
    assert compute_order(pts, "columns:2:rowrev") == list(range(len(pts)))
