from unittest.mock import MagicMock

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


def test_recognize_with_fallback_replaces_only_strictly_better_results():
    pipeline = make_pipeline()
    rotated = [crop(1), crop(10), crop(20)]
    initial = [
        {"rec_text": "A", "rec_score": 0.6},
        {"rec_text": "B", "rec_score": 0.7},
        {"rec_text": "C", "rec_score": 0.8},
    ]
    fallback = [
        {"rec_text": "A2", "rec_score": 0.9},
        {"rec_text": "B2", "rec_score": 0.7},
        {"rec_text": "C2", "rec_score": 0.6},
    ]
    pipeline.text_rec_model.predict.side_effect = [initial, fallback]

    final_crops, results = pipeline._recognize_with_fallback(rotated, [0, 1, 2])

    assert len(pipeline.text_rec_model.predict.call_args_list[1].args[0]) == 3
    assert np.array_equal(final_crops[0], cv2.rotate(rotated[0], cv2.ROTATE_180))
    assert np.array_equal(final_crops[1], rotated[1])
    assert np.array_equal(final_crops[2], rotated[2])
    assert results == [fallback[0], initial[1], initial[2]]


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
