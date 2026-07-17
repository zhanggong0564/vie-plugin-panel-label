"""Panel-label local-coordinate ROI extraction tests."""

import cv2
import numpy as np

from vie_plugin_panel_label import utils as panel_utils


def _gradient_image(height=240, width=320):
    yy, xx = np.indices((height, width), dtype=np.uint16)
    return np.stack(
        (
            (xx % 256).astype(np.uint8),
            (yy % 256).astype(np.uint8),
            ((xx + yy) % 256).astype(np.uint8),
        ),
        axis=2,
    )


def _rotated_polygon():
    return cv2.boxPoints(
        ((165.0, 118.0), (150.0, 42.0), 17.0)
    ).astype(np.float32)


def test_points_to_local_mask_allocates_only_polygon_bounds():
    polygon = np.array(
        [[100, 200], [150, 200], [150, 240], [100, 240]],
        dtype=np.float32,
    )

    mask, offset = panel_utils.points_to_local_mask(
        (1000, 2000), polygon
    )

    assert offset == (100, 200)
    assert mask.shape == (41, 51)
    assert cv2.countNonZero(mask) == 41 * 51


def test_local_roi_matches_legacy_crop_and_transform():
    image = _gradient_image()
    polygons = [_rotated_polygon()]

    legacy_rois, legacy_transforms = panel_utils.mask2roi(
        image, polygons, return_maps=True
    )
    local_rois, local_transforms = panel_utils.mask2roi_local(
        image, polygons, return_maps=True
    )

    assert len(local_rois) == len(legacy_rois) == 1
    assert local_rois[0].shape == legacy_rois[0].shape
    np.testing.assert_array_equal(local_rois[0], legacy_rois[0])
    assert local_transforms[0].H == legacy_transforms[0].H
    assert local_transforms[0].W == legacy_transforms[0].W

    roi_points = np.array(
        [
            [0, 0],
            [
                local_transforms[0].W - 1,
                local_transforms[0].H - 1,
            ],
        ],
        dtype=np.float32,
    )
    legacy_points = panel_utils.map_roi_points_to_original(
        legacy_transforms[0], roi_points
    )
    local_points = panel_utils.map_roi_points_to_original(
        local_transforms[0], roi_points
    )
    np.testing.assert_allclose(local_points, legacy_points, atol=1.0)


def test_local_roi_matches_legacy_when_rotated_crop_exceeds_mask_bounds():
    image = _gradient_image(height=600, width=900)
    polygon = cv2.boxPoints(
        ((735.0, 420.0), (420.0, 55.0), 73.0)
    ).astype(np.float32)

    legacy_rois, legacy_transforms = panel_utils.mask2roi(
        image, [polygon], return_maps=True
    )
    local_rois, local_transforms = panel_utils.mask2roi_local(
        image, [polygon], return_maps=True
    )

    np.testing.assert_array_equal(local_rois[0], legacy_rois[0])
    np.testing.assert_array_equal(
        local_transforms[0].top, legacy_transforms[0].top
    )
    np.testing.assert_array_equal(
        local_transforms[0].bot, legacy_transforms[0].bot
    )


def test_local_roi_falls_back_per_polygon(monkeypatch):
    image = _gradient_image(40, 60)
    polygon = np.array(
        [[5, 5], [20, 5], [20, 25], [5, 25]], dtype=np.float32
    )
    fallback_transform = object()

    monkeypatch.setattr(
        panel_utils,
        "rotate_polygon_upright",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("degenerate")
        ),
    )
    monkeypatch.setattr(
        panel_utils,
        "mask2roi",
        lambda img, points, **kwargs: (
            ["legacy"],
            [fallback_transform],
        ),
    )

    rois, transforms = panel_utils.mask2roi_local(
        image, [polygon], return_maps=True
    )

    assert rois == ["legacy"]
    assert transforms == [fallback_transform]


def test_points_to_mask_uses_local_roi_path(monkeypatch):
    image = _gradient_image(40, 60)
    polygon = np.array(
        [[5, 5], [20, 5], [20, 25], [5, 25]], dtype=np.float32
    )
    sorted_indices = np.array([0], dtype=np.int64)
    transform = object()
    monkeypatch.setattr(
        panel_utils,
        "sort_mask",
        lambda *args, **kwargs: ([polygon], sorted_indices),
    )
    monkeypatch.setattr(
        panel_utils,
        "mask2roi_local",
        lambda *args, **kwargs: (["local"], [transform]),
    )

    rois, actual_indices, transforms = panel_utils.Points_to_Mask(
        image, [polygon], return_maps=True
    )

    assert rois == ["local"]
    assert np.array_equal(actual_indices, sorted_indices)
    assert transforms == [transform]


def test_points_to_mask_legacy_keeps_full_mask_path(monkeypatch):
    image = _gradient_image(40, 60)
    polygon = np.array(
        [[5, 5], [20, 5], [20, 25], [5, 25]], dtype=np.float32
    )
    sorted_indices = np.array([0], dtype=np.int64)
    monkeypatch.setattr(
        panel_utils,
        "sort_mask",
        lambda *args, **kwargs: ([polygon], sorted_indices),
    )
    monkeypatch.setattr(
        panel_utils,
        "mask2roi",
        lambda *args, **kwargs: ["legacy"],
    )

    rois, actual_indices = panel_utils.Points_to_Mask_legacy(
        image, [polygon]
    )

    assert rois == ["legacy"]
    assert np.array_equal(actual_indices, sorted_indices)
