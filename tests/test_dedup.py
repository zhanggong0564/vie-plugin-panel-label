"""检测实例去重（rotated_box_overlap / dedup_overlapping_polygons）单元测试。

背景：YOLO 轴对齐 NMS（nmsThreshold=0.8）抑制不掉同一线标上的重复检测
（典型为全长框 + 半截框），导致 observed_count 多于标准数判 extra。
去重基于 mask 多边形最小外接旋转矩形的"交集/较小框面积"（IoS）：
同一线标的重复框 IoS 接近 1，相邻倾斜线标的旋转框几乎不相交。
"""
import numpy as np
import pytest

from vie_plugin_panel_label import utils as panel_utils
from vie_plugin_panel_label.utils import dedup_overlapping_polygons, rotated_box_overlap


def _rect(x1, y1, x2, y2):
    """轴对齐矩形的四点多边形（dedup 内部取 minAreaRect，旋转与否均可）。"""
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


class TestRotatedBoxOverlap:
    def test_contained_box_overlap_is_one(self):
        """半截框完全落在全长框内：IoS = 1"""
        full = _rect(0, 0, 40, 200)
        half = _rect(0, 0, 40, 100)
        assert rotated_box_overlap(full, half) == pytest.approx(1.0, abs=1e-3)

    def test_disjoint_boxes_overlap_is_zero(self):
        assert rotated_box_overlap(_rect(0, 0, 40, 200), _rect(60, 0, 100, 200)) == 0.0

    def test_slight_overlap_below_dup_level(self):
        """相邻线标轻微相交：IoS 远低于重复框水平"""
        overlap = rotated_box_overlap(_rect(0, 0, 40, 200), _rect(35, 0, 75, 200))
        assert 0.0 < overlap < 0.2

    def test_tilted_duplicate_high_overlap(self):
        """倾斜套管的重复框（同向旋转矩形，一长一短）IoS 仍接近 1"""
        base = np.array([[100, 100], [140, 140], [40, 240], [0, 200]], dtype=np.float32)
        short = np.array([[100, 100], [140, 140], [90, 190], [50, 150]], dtype=np.float32)
        assert rotated_box_overlap(base, short) > 0.9


class TestDedupOverlappingPolygons:
    def test_duplicate_keeps_higher_score(self):
        polys = [_rect(0, 0, 40, 200), _rect(0, 0, 40, 100)]
        keep = dedup_overlapping_polygons(polys, scores=[0.9, 0.8], class_ids=[0, 0], overlap_thresh=0.6)
        assert keep == [0]

    def test_duplicate_keeps_higher_score_regardless_of_order(self):
        polys = [_rect(0, 0, 40, 100), _rect(0, 0, 40, 200)]
        keep = dedup_overlapping_polygons(polys, scores=[0.8, 0.9], class_ids=[0, 0], overlap_thresh=0.6)
        assert keep == [1]

    def test_adjacent_boxes_both_kept(self):
        polys = [_rect(0, 0, 40, 200), _rect(35, 0, 75, 200), _rect(70, 0, 110, 200)]
        keep = dedup_overlapping_polygons(polys, scores=[0.9, 0.8, 0.7], class_ids=[0, 0, 0], overlap_thresh=0.6)
        assert keep == [0, 1, 2]

    def test_cross_class_overlap_not_deduped(self):
        """仅同类之间去重：line 与 QFU 重叠不互相抑制"""
        polys = [_rect(0, 0, 40, 200), _rect(0, 0, 40, 200)]
        keep = dedup_overlapping_polygons(polys, scores=[0.9, 0.8], class_ids=[0, 1], overlap_thresh=0.6)
        assert keep == [0, 1]

    def test_thresh_ge_one_disables(self):
        """阈值 >= 1 等效关闭去重"""
        polys = [_rect(0, 0, 40, 200), _rect(0, 0, 40, 200)]
        keep = dedup_overlapping_polygons(polys, scores=[0.9, 0.8], class_ids=[0, 0], overlap_thresh=1.0)
        assert keep == [0, 1]

    def test_empty_input(self):
        assert dedup_overlapping_polygons([], scores=[], class_ids=[], overlap_thresh=0.6) == []


class TestPointsToMaskContract:
    def test_return_maps_returns_rois_sorted_indices_and_transforms(self, monkeypatch):
        points = [np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)]
        sorted_idx = np.array([0], dtype=np.int64)
        monkeypatch.setattr(panel_utils, "sort_mask", lambda image, pts, row_alpha: (points, sorted_idx))
        monkeypatch.setattr(
            panel_utils,
            "mask2roi_local",
            lambda image, pts, return_maps=False: (
                (["roi"], ["transform"]) if return_maps else ["roi"]
            ),
        )

        mask_rois, actual_sorted_idx, transforms = panel_utils.Points_to_Mask(
            np.zeros((20, 20, 3), dtype=np.uint8),
            points,
            return_maps=True,
        )

        assert mask_rois == ["roi"]
        assert np.array_equal(actual_sorted_idx, sorted_idx)
        assert transforms == ["transform"]
