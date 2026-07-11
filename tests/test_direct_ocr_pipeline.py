from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from vie_plugin_panel_label.panel_label_detect import OCRPipeline


def make_pipeline():
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.text_orient_score_thresh = 0.9
    pipeline.text_rec_score_thresh = 0.7
    pipeline.text_orient_model = MagicMock()
    pipeline.text_rec_model = MagicMock()
    return pipeline


def crop(value):
    return np.array([[[value], [value + 1]]], dtype=np.uint8)


def test_orient_crops_rotates_180_and_tracks_only_uncertain_indices():
    pipeline = make_pipeline()
    crops = [crop(1), crop(10), crop(20)]
    pipeline.text_orient_model.predict.return_value = [
        {"class_ids": [0], "scores": [0.99]},
        {"class_ids": [1], "scores": [0.95]},
        {"class_ids": [1], "scores": [0.50]},
    ]

    rotated, uncertain = pipeline._orient_crops(crops)

    assert np.array_equal(rotated[0], crops[0])
    assert np.array_equal(rotated[1], cv2.rotate(crops[1], cv2.ROTATE_180))
    assert np.array_equal(rotated[2], cv2.rotate(crops[2], cv2.ROTATE_180))
    assert uncertain == [2]


def test_recognize_with_fallback_uses_sparse_indices_and_writes_back_in_place():
    pipeline = make_pipeline()
    rotated = [crop(1), crop(10), crop(20)]
    initial = [
        {"rec_text": "A", "rec_score": 0.6},
        {"rec_text": "B", "rec_score": 0.7},
        {"rec_text": "C", "rec_score": 0.8},
    ]
    fallback = [
        {"rec_text": "A2", "rec_score": 0.9},
        {"rec_text": "C2", "rec_score": 0.6},
    ]
    pipeline.text_rec_model.predict.side_effect = [initial, fallback]

    final_crops, results = pipeline._recognize_with_fallback(rotated, [0, 2])

    fallback_input = pipeline.text_rec_model.predict.call_args_list[1].args[0]
    assert len(fallback_input) == 2
    assert np.array_equal(fallback_input[0], cv2.rotate(rotated[0], cv2.ROTATE_180))
    assert np.array_equal(fallback_input[1], cv2.rotate(rotated[2], cv2.ROTATE_180))
    assert np.array_equal(final_crops[0], cv2.rotate(rotated[0], cv2.ROTATE_180))
    assert np.array_equal(final_crops[1], rotated[1])
    assert np.array_equal(final_crops[2], rotated[2])
    assert results == [fallback[0], initial[1], initial[2]]


def test_orient_crops_rejects_result_count_mismatch():
    pipeline = make_pipeline()
    pipeline.text_orient_model.predict.return_value = [
        {"class_ids": [0], "scores": [0.99]},
    ]

    with np.testing.assert_raises_regex(ValueError, "orientation result count"):
        pipeline._orient_crops([crop(1), crop(2)])


def test_recognize_with_fallback_rejects_result_count_mismatch():
    pipeline = make_pipeline()
    pipeline.text_rec_model.predict.return_value = [
        {"rec_text": "A", "rec_score": 0.9},
    ]

    with np.testing.assert_raises_regex(ValueError, "recognition result count"):
        pipeline._recognize_with_fallback([crop(1), crop(2)], [])


def test_recognize_with_fallback_rejects_fallback_result_count_mismatch():
    pipeline = make_pipeline()
    pipeline.text_rec_model.predict.side_effect = [
        [
            {"rec_text": "A", "rec_score": 0.9},
            {"rec_text": "B", "rec_score": 0.8},
        ],
        [],
    ]

    with np.testing.assert_raises_regex(ValueError, "fallback recognition result count"):
        pipeline._recognize_with_fallback([crop(1), crop(2)], [1])


def test_extract_texts_normalizes_shapes_and_thresholds():
    pipeline = make_pipeline()
    results = [
        {"rec_text": "A", "rec_score": 0.9},
        {"rec_text": ["B"], "rec_score": 0.8},
        {"rec_text": [], "rec_score": 0.9},
        {"rec_text": "   ", "rec_score": 0.9},
        {"rec_text": "LOW", "rec_score": 0.69},
    ]

    assert pipeline._extract_texts(results) == ["A", "B", None, None, None]


def test_infer_preserves_sorted_line_mapping_across_all_output_fields():
    pipeline = make_pipeline()
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    polygons = np.array([
        [[0, 0], [4, 0], [4, 2], [0, 2]],
        [[10, 10], [14, 10], [14, 12], [10, 12]],
        [[20, 20], [24, 20], [24, 22], [20, 22]],
    ], dtype=object)
    pipeline.detect_model = MagicMock()
    pipeline.detect_model.infer.return_value = SimpleNamespace(
        class_ids=[0, 1, 0], scores=[0.91, 0.50, 0.82], mask_polygons=polygons
    )
    pipeline.dedup_overlap_thresh = 1.0
    roi_for_line_2 = crop(20)
    roi_for_line_0 = crop(1)
    pipeline.text_orient_model.predict.return_value = [
        {"class_ids": [0], "scores": [0.99]},
        {"class_ids": [0], "scores": [0.40]},
    ]
    pipeline.text_rec_model.predict.side_effect = [
        [
            {"rec_text": "LOW", "rec_score": 0.60},
            {"rec_text": "WRONG", "rec_score": 0.50},
        ],
        [{"rec_text": "LINE0", "rec_score": 0.95}],
    ]

    with patch(
        "vie_plugin_panel_label.panel_label_detect.Points_to_Mask",
        return_value=([roi_for_line_2, roi_for_line_0], [1, 0], [None, None]),
    ):
        result = pipeline.infer(image)

    assert result.index == [2, 0]
    assert result.class_id == [0, 0]
    assert result.confidence == [0.82, 0.91]
    assert result.texts == [None, "LINE0"]
    assert len(result.Points) == len(result.index) == len(result.class_id) == 2
    assert len(result.confidence) == len(result.texts) == len(result.text_crops) == 2
    assert np.array_equal(result.text_crops[0], roi_for_line_2)
    assert np.array_equal(result.text_crops[1], cv2.rotate(roi_for_line_0, cv2.ROTATE_180))
    assert result.Points == [
        np.int64(cv2.boxPoints(cv2.minAreaRect(np.array(polygons[2], dtype=np.float32)))).flatten().tolist(),
        np.int64(cv2.boxPoints(cv2.minAreaRect(np.array(polygons[0], dtype=np.float32)))).flatten().tolist(),
    ]
