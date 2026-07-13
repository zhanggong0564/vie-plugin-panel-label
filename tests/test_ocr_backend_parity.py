"""Strict Paddle/ONNX parity checks on local panel-label samples.

The samples and Paddle weights are deliberately discovered outside the worktree.
No image, crop, OCR text, or generated baseline is persisted by this module.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import random

import cv2
import numpy as np
import pytest

from services.base import OnnxRuntimeRunner
from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
from vie_plugin_panel_label.panel_label_detect import (
    OCRPipeline,
    PanelLabelDetect,
    Points_to_Mask,
)
from vie_plugin_panel_label.ocr_models import (
    PanelLabelOrientationClassifier,
    PanelLabelTextRecognizer,
)


SEED = 20260713
SOURCE_REPOSITORY = Path(
    "/home/zhanggong/workspace/VisInferEngine/mobile_vision"
)
SAMPLE_ROOT = SOURCE_REPOSITORY / "demo/data/panel_label/charging_pile"
WEIGHT_ROOT = SOURCE_REPOSITORY / "weights/panel_label/v2"
INPUT_RTOL = 1e-5
INPUT_ATOL = 1e-6
LOGITS_MAX_ABS_ERROR = 1e-2
SCORE_MAX_ABS_ERROR = 1e-3


def select_samples(root: Path, count: int = 20) -> list[Path]:
    """Select a stable sample while explicitly validating symlink roots."""
    if not root.exists():
        pytest.skip(f"panel-label sample root is unavailable: {root}")

    paths = []
    missing_targets = []
    for entry in sorted(root.iterdir()):
        if entry.is_symlink() and not entry.exists():
            missing_targets.append(f"{entry} -> {entry.readlink()}")
            continue
        if entry.is_dir():
            paths.extend(
                path for path in entry.rglob("*.jpg") if path.is_file()
            )
    paths.extend(path for path in root.glob("*.jpg") if path.is_file())
    paths = sorted(set(paths))
    if len(paths) < count:
        details = (
            f"; unavailable symlink targets: {', '.join(missing_targets)}"
            if missing_targets
            else ""
        )
        pytest.skip(
            f"need {count} panel-label images, found {len(paths)} under {root}"
            f"{details}"
        )
    return random.Random(SEED).sample(paths, count)


def _require_path(path: Path, description: str) -> Path:
    if not path.exists():
        pytest.skip(f"{description} is unavailable: {path}")
    return path


def _paddle_predictor(model_name: str, model_dir: Path):
    pytest.importorskip("paddle", reason="Paddle CPU reference runtime is unavailable")
    paddleocr = pytest.importorskip(
        "paddleocr", reason="PaddleOCR 3.4.0 reference is unavailable"
    )
    paddlex = pytest.importorskip(
        "paddlex", reason="PaddleX reference model API is unavailable"
    )
    import paddle

    if paddleocr.__version__ != "3.4.0":
        pytest.skip(
            f"PaddleOCR 3.4.0 is required, found {paddleocr.__version__}"
        )
    paddle.set_device("gpu:0")
    return paddlex.create_model(
        model_name,
        model_dir=str(model_dir),
        device="gpu:0",
    )._predictor


def _image_key(image: np.ndarray):
    """Return a process-local, non-reversible cache key without persisting data."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()
    return contiguous.shape, contiguous.dtype.str, digest


class PaddleOrientationReference:
    def __init__(self, model_dir: Path):
        self.predictor = _paddle_predictor(
            "PP-LCNet_x1_0_textline_ori", model_dir
        )
        self.raw_cache = {}
        self.result_cache = {}

    def raw(self, images):
        key = tuple(_image_key(image) for image in images)
        if key in self.raw_cache:
            return self.raw_cache[key]
        ops = self.predictor.preprocessors
        batch = ops["Read"](imgs=images)
        batch = ops["Resize"](imgs=batch)
        batch = ops["Normalize"](imgs=batch)
        batch = ops["ToCHW"](imgs=batch)
        tensor = ops["ToBatch"](imgs=batch)[0]
        logits = self.predictor.infer(x=[tensor])[0]
        self.raw_cache[key] = (tensor, logits)
        return tensor, logits

    def predict(self, images):
        if not images:
            return []
        keys = [_image_key(image) for image in images]
        if all(key in self.result_cache for key in keys):
            return [self.result_cache[key] for key in keys]
        _, logits = self.raw(images)
        class_ids, scores, _ = self.predictor.postprocessors["Topk"](
            [logits], topk=1
        )
        results = [
            {"class_ids": ids.tolist(), "scores": row_scores.tolist()}
            for ids, row_scores in zip(class_ids, scores)
        ]
        self.result_cache.update(zip(keys, results))
        return results


class PaddleRecognitionReference:
    def __init__(self, model_dir: Path):
        self.predictor = _paddle_predictor("PP-OCRv5_server_rec", model_dir)
        self.raw_cache = {}
        self.result_cache = {}

    def prime(self, images):
        """Infer equal-width crops together while retaining per-crop artifacts."""
        groups = {}
        ops = self.predictor.pre_tfs
        for image in images:
            key = _image_key(image)
            if (key,) in self.raw_cache:
                continue
            batch = ops["Read"](imgs=[image])
            batch = ops["ReisizeNorm"](imgs=batch)
            tensor = ops["ToBatch"](imgs=batch)[0]
            groups.setdefault(tensor.shape[1:], []).append((key, tensor))
        for items in groups.values():
            tensor = np.concatenate([item[1] for item in items], axis=0)
            logits = self.predictor.infer(x=[tensor])[0]
            for index, (key, single_tensor) in enumerate(items):
                self.raw_cache[(key,)] = (
                    single_tensor,
                    logits[index : index + 1],
                )

    def raw(self, images):
        key = tuple(_image_key(image) for image in images)
        if key in self.raw_cache:
            return self.raw_cache[key]
        ops = self.predictor.pre_tfs
        batch = ops["Read"](imgs=images)
        batch = ops["ReisizeNorm"](imgs=batch)
        tensor = ops["ToBatch"](imgs=batch)[0]
        logits = self.predictor.infer(x=[tensor])[0]
        self.raw_cache[key] = (tensor, logits)
        return tensor, logits

    def predict(self, images):
        if not images:
            return []
        # PaddleX 3.4.1's default TextRecPredictor batch sampler executes one
        # crop at a time. Preserve that real reference behavior here.
        results = []
        for image in images:
            key = _image_key(image)
            if key in self.result_cache:
                results.append(self.result_cache[key])
                continue
            _, logits = self.raw([image])
            ratio = image.shape[1] / float(image.shape[0])
            texts, scores = self.predictor.post_op(
                [logits],
                wh_ratio_list=[ratio],
                max_wh_ratio=max(320 / 48, ratio),
            )
            result = {"rec_text": texts[0], "rec_score": float(scores[0])}
            self.result_cache[key] = result
            results.append(result)
        return results


class CachedOnnxOrientation:
    """Cache verified ONNX outputs so orchestration does not repeat inference."""

    def __init__(self, model):
        self.model = model
        self.raw_cache = {}
        self.result_cache = {}

    def raw(self, images):
        key = tuple(_image_key(image) for image in images)
        if key not in self.raw_cache:
            tensor = self.model.preprocess(images)
            logits = self.model.runner.run({self.model.input_name: tensor})[0]
            self.raw_cache[key] = (tensor, logits)
        return self.raw_cache[key]

    def predict(self, images):
        if not images:
            return []
        keys = [_image_key(image) for image in images]
        if not all(key in self.result_cache for key in keys):
            _, logits = self.raw(images)
            class_ids = logits.argmax(axis=1)
            results = [
                {
                    "class_ids": [int(class_id)],
                    "scores": [float(logits[index, class_id])],
                }
                for index, class_id in enumerate(class_ids)
            ]
            self.result_cache.update(zip(keys, results))
        return [self.result_cache[key] for key in keys]


class CachedOnnxRecognition:
    """Use once-verified per-crop logits for decoding and OCR orchestration."""

    def __init__(self, model):
        self.model = model
        self.raw_cache = {}
        self.result_cache = {}

    def prime(self, images):
        """Infer equal-width crops in batches, then cache per-crop logits."""
        groups = {}
        for image in images:
            key = _image_key(image)
            if key in self.raw_cache:
                continue
            tensor = self.model.preprocess_batch([image])
            groups.setdefault(tensor.shape[1:], []).append((key, tensor))
        for items in groups.values():
            tensor = np.concatenate([item[1] for item in items], axis=0)
            logits = self.model.runner.run({self.model.input_name: tensor})[0]
            for index, (key, single_tensor) in enumerate(items):
                self.raw_cache[key] = (
                    single_tensor,
                    logits[index : index + 1],
                )

    def raw(self, images):
        assert len(images) == 1, "strict recognition model checks are per crop"
        key = _image_key(images[0])
        if key not in self.raw_cache:
            tensor = self.model.preprocess_batch(images)
            logits = self.model.runner.run({self.model.input_name: tensor})[0]
            self.raw_cache[key] = (tensor, logits)
        return self.raw_cache[key]

    def predict(self, images):
        results = []
        for image in images:
            key = _image_key(image)
            if key not in self.result_cache:
                _, logits = self.raw([image])
                self.result_cache[key] = self.model.decode(logits)[0]
            results.append(self.result_cache[key])
        return results


class FixedDetector:
    def __init__(self, result):
        self.result = result

    def infer(self, _image):
        return deepcopy(self.result)


def _pipeline(detector, orientation, recognition):
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detect_model = detector
    pipeline.text_orient_model = orientation
    pipeline.text_rec_model = recognition
    pipeline.text_rec_score_thresh = 0.7
    pipeline.text_orient_score_thresh = 0.7
    pipeline.dedup_overlap_thresh = 0.6
    return pipeline


def _comparison_error(actual, expected):
    if actual.shape != expected.shape:
        return float("inf"), True, False, None
    nonfinite = ~np.isfinite(actual) | ~np.isfinite(expected)
    if np.any(nonfinite):
        location = tuple(int(index) for index in np.argwhere(nonfinite)[0])
        return float("inf"), False, True, location
    delta = np.abs(actual - expected)
    if not delta.size:
        return 0.0, False, False, None
    flat_index = int(np.argmax(delta))
    location = tuple(int(index) for index in np.unravel_index(flat_index, delta.shape))
    return float(delta.flat[flat_index]), False, False, location


def _scalar_error(actual, expected):
    if not np.isfinite(actual) or not np.isfinite(expected):
        return float("inf"), True
    return abs(float(actual) - float(expected)), False


def _update_max_detail(
    details, layer, error, sample_number, tensor_index, shape, location
):
    current = details.get(layer)
    if current is None or error > current["error"]:
        details[layer] = {
            "error": error,
            "sample": sample_number,
            "tensor_index": tensor_index,
            "shape": tuple(shape),
            "location": location,
        }


def _allclose_exceeded(actual, expected, rtol, atol):
    if actual.shape != expected.shape:
        return True
    return not bool(np.allclose(actual, expected, rtol=rtol, atol=atol))


def test_gpu_parity_thresholds_are_pinned():
    """Prevent accidental drift from the explicitly approved GPU criteria."""
    assert INPUT_RTOL == 1e-5
    assert INPUT_ATOL == 1e-6
    assert LOGITS_MAX_ABS_ERROR == 1e-2
    assert SCORE_MAX_ABS_ERROR == 1e-3


def test_comparison_diagnostics_reject_nonfinite_and_locate_maximum():
    finite = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)
    shifted = finite.copy()
    shifted[1, 0] += 0.25
    error, shape_mismatch, nonfinite, location = _comparison_error(
        shifted, finite
    )
    assert error == pytest.approx(0.25)
    assert not shape_mismatch
    assert not nonfinite
    assert location == (1, 0)

    for value in (np.nan, np.inf, -np.inf):
        invalid = finite.copy()
        invalid[0, 1] = value
        error, shape_mismatch, nonfinite, location = _comparison_error(
            invalid, finite
        )
        assert error == float("inf")
        assert not shape_mismatch
        assert nonfinite
        assert location == (0, 1)
        scalar_error, scalar_nonfinite = _scalar_error(value, 0.0)
        assert scalar_error == float("inf")
        assert scalar_nonfinite

    details = {}
    _update_max_detail(details, "logits", 0.1, 2, 4, finite.shape, (1, 0))
    _update_max_detail(details, "logits", 0.05, 3, 1, finite.shape, (0, 0))
    assert details["logits"] == {
        "error": 0.1,
        "sample": 2,
        "tensor_index": 4,
        "shape": (2, 2),
        "location": (1, 0),
    }


def test_paddle_and_onnx_are_strictly_aligned_on_fixed_real_samples():
    samples = select_samples(SAMPLE_ROOT)
    detect_path = _require_path(WEIGHT_ROOT / "best.onnx", "detection ONNX model")
    orient_onnx = _require_path(
        WEIGHT_ROOT / "textline_ori_lcnet_v2.onnx", "orientation ONNX model"
    )
    orient_paddle = _require_path(
        WEIGHT_ROOT / "textline_ori_lcnet_v2", "orientation Paddle model"
    )
    rec_onnx = _require_path(
        WEIGHT_ROOT / "PP-OCRv5_server_rec_merged_v6_diff_lr.onnx",
        "recognition ONNX model",
    )
    rec_paddle = _require_path(
        WEIGHT_ROOT / "PP-OCRv5_server_rec_merged_v6_diff_lr",
        "recognition Paddle model",
    )

    paddle_orientation = PaddleOrientationReference(orient_paddle)
    paddle_recognition = PaddleRecognitionReference(rec_paddle)
    onnx_orientation = CachedOnnxOrientation(
        PanelLabelOrientationClassifier(
            str(orient_onnx),
            str(orient_paddle / "inference.yml"),
            runner=OnnxRuntimeRunner(
                str(orient_onnx),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                warmup=False,
            ),
        ),
    )
    onnx_recognition = CachedOnnxRecognition(
        PanelLabelTextRecognizer(
            str(rec_onnx),
            str(rec_paddle / "inference.yml"),
            runner=OnnxRuntimeRunner(
                str(rec_onnx),
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                warmup=False,
            ),
        ),
    )
    detector = PanelLabelDetect(str(detect_path))
    assert onnx_orientation.model.runner.providers[0] == "CUDAExecutionProvider"
    assert onnx_recognition.model.runner.providers[0] == "CUDAExecutionProvider"
    assert detector.runner.providers[0] == "CUDAExecutionProvider"

    crop_count = 0
    orientation_matches = 0
    text_matches = 0
    recognition_count = 0
    max_orientation_score_error = 0.0
    max_recognition_score_error = 0.0
    max_orientation_logits_error = 0.0
    max_recognition_logits_error = 0.0
    max_details = {}
    diagnostics = {
        "orientation_input_mismatch": 0,
        "orientation_input_nonfinite": 0,
        "orientation_logits_threshold_exceeded": 0,
        "orientation_logits_nonfinite": 0,
        "orientation_shape_mismatch": 0,
        "orientation_class_mismatch": 0,
        "orientation_result_count_mismatch": 0,
        "orientation_score_threshold_exceeded": 0,
        "orientation_score_nonfinite": 0,
        "recognition_input_mismatch": 0,
        "recognition_input_nonfinite": 0,
        "recognition_logits_threshold_exceeded": 0,
        "recognition_logits_nonfinite": 0,
        "recognition_shape_mismatch": 0,
        "recognition_text_mismatch": 0,
        "recognition_result_count_mismatch": 0,
        "recognition_score_threshold_exceeded": 0,
        "recognition_score_nonfinite": 0,
        "detection_count_mismatch": 0,
        "sorting_index_mismatch": 0,
        "orientation_arbitration_mismatch": 0,
        "field_mapping_mismatch": 0,
        "business_judgement_mismatch": 0,
    }

    for sample_number, sample in enumerate(samples, start=1):
        image = cv2.imread(str(sample))
        assert image is not None, f"failed to decode panel-label image: {sample}"
        detection = detector.infer(image)
        class_ids = np.asarray(detection.class_ids)
        polygons = np.asarray(detection.mask_polygons, dtype=object)
        line_polygons = polygons[class_ids == 0] if 0 in class_ids else []
        crops, sorted_indices, _ = Points_to_Mask(
            image, line_polygons, return_maps=True
        )

        if crops:
            crops = list(crops)
            crop_count += len(crops)
            paddle_input, paddle_logits = paddle_orientation.raw(crops)
            onnx_input, onnx_logits = onnx_orientation.raw(crops)
            diagnostics["orientation_input_mismatch"] += int(
                _allclose_exceeded(
                    onnx_input,
                    paddle_input,
                    rtol=INPUT_RTOL,
                    atol=INPUT_ATOL,
                )
            )
            input_error, _, input_nonfinite, input_location = _comparison_error(
                onnx_input, paddle_input
            )
            diagnostics["orientation_input_nonfinite"] += int(input_nonfinite)
            _update_max_detail(
                max_details,
                "orientation_input",
                input_error,
                sample_number,
                input_location[0] if input_location else None,
                onnx_input.shape,
                input_location,
            )
            error, shape_mismatch, nonfinite, location = _comparison_error(
                onnx_logits, paddle_logits
            )
            max_orientation_logits_error = max(
                max_orientation_logits_error, error
            )
            diagnostics["orientation_shape_mismatch"] += int(shape_mismatch)
            diagnostics["orientation_logits_nonfinite"] += int(nonfinite)
            diagnostics["orientation_logits_threshold_exceeded"] += int(
                not shape_mismatch and error > LOGITS_MAX_ABS_ERROR
            )
            _update_max_detail(
                max_details,
                "orientation_logits",
                error,
                sample_number,
                location[0] if location else None,
                onnx_logits.shape,
                location,
            )
            paddle_orient_results = paddle_orientation.predict(crops)
            onnx_orient_results = onnx_orientation.predict(crops)
            diagnostics["orientation_result_count_mismatch"] += int(
                len(onnx_orient_results) != len(paddle_orient_results)
            )
            for result_index, (paddle_result, onnx_result) in enumerate(
                zip(paddle_orient_results, onnx_orient_results)
            ):
                diagnostics["orientation_class_mismatch"] += int(
                    onnx_result["class_ids"] != paddle_result["class_ids"]
                )
                error, nonfinite = _scalar_error(
                    onnx_result["scores"][0], paddle_result["scores"][0]
                )
                diagnostics["orientation_score_nonfinite"] += int(nonfinite)
                diagnostics["orientation_score_threshold_exceeded"] += int(
                    error > SCORE_MAX_ABS_ERROR
                )
                max_orientation_score_error = max(
                    max_orientation_score_error, error
                )
                _update_max_detail(
                    max_details,
                    "orientation_score",
                    error,
                    sample_number,
                    result_index,
                    (),
                    None,
                )
                orientation_matches += int(
                    onnx_result["class_ids"] == paddle_result["class_ids"]
                )

            rotated, uncertain = _pipeline(
                FixedDetector(detection), paddle_orientation, paddle_recognition
            )._orient_crops(crops)
            recognition_inputs = list(rotated) + [
                cv2.rotate(rotated[index], cv2.ROTATE_180)
                for index in uncertain
            ]
            paddle_recognition.prime(recognition_inputs)
            onnx_recognition.prime(recognition_inputs)
            for recognition_index, recognition_input in enumerate(recognition_inputs):
                paddle_input, paddle_logits = paddle_recognition.raw(
                    [recognition_input]
                )
                onnx_input, onnx_logits = onnx_recognition.raw(
                    [recognition_input]
                )
                diagnostics["recognition_input_mismatch"] += int(
                    _allclose_exceeded(
                        onnx_input,
                        paddle_input,
                        rtol=INPUT_RTOL,
                        atol=INPUT_ATOL,
                    )
                )
                input_error, _, input_nonfinite, input_location = _comparison_error(
                    onnx_input, paddle_input
                )
                diagnostics["recognition_input_nonfinite"] += int(input_nonfinite)
                _update_max_detail(
                    max_details,
                    "recognition_input",
                    input_error,
                    sample_number,
                    recognition_index,
                    onnx_input.shape,
                    input_location,
                )
                error, shape_mismatch, nonfinite, location = _comparison_error(
                    onnx_logits, paddle_logits
                )
                max_recognition_logits_error = max(
                    max_recognition_logits_error, error
                )
                diagnostics["recognition_shape_mismatch"] += int(shape_mismatch)
                diagnostics["recognition_logits_nonfinite"] += int(nonfinite)
                diagnostics["recognition_logits_threshold_exceeded"] += int(
                    not shape_mismatch and error > LOGITS_MAX_ABS_ERROR
                )
                _update_max_detail(
                    max_details,
                    "recognition_logits",
                    error,
                    sample_number,
                    recognition_index,
                    onnx_logits.shape,
                    location,
                )
            paddle_rec_results = paddle_recognition.predict(recognition_inputs)
            onnx_rec_results = onnx_recognition.predict(recognition_inputs)
            diagnostics["recognition_result_count_mismatch"] += int(
                len(onnx_rec_results) != len(paddle_rec_results)
            )
            for result_index, (paddle_result, onnx_result) in enumerate(
                zip(paddle_rec_results, onnx_rec_results)
            ):
                recognition_count += 1
                diagnostics["recognition_text_mismatch"] += int(
                    onnx_result["rec_text"] != paddle_result["rec_text"]
                )
                error, nonfinite = _scalar_error(
                    onnx_result["rec_score"], paddle_result["rec_score"]
                )
                diagnostics["recognition_score_nonfinite"] += int(nonfinite)
                diagnostics["recognition_score_threshold_exceeded"] += int(
                    error > SCORE_MAX_ABS_ERROR
                )
                max_recognition_score_error = max(
                    max_recognition_score_error, error
                )
                _update_max_detail(
                    max_details,
                    "recognition_score",
                    error,
                    sample_number,
                    result_index,
                    (),
                    None,
                )
                text_matches += int(
                    onnx_result["rec_text"] == paddle_result["rec_text"]
                )

        paddle_pipeline = _pipeline(
            FixedDetector(detection), paddle_orientation, paddle_recognition
        )
        onnx_pipeline = _pipeline(
            FixedDetector(detection), onnx_orientation, onnx_recognition
        )
        paddle_result = paddle_pipeline.infer(image)
        onnx_result = onnx_pipeline.infer(image)
        expected_indices = [
            np.where(class_ids == 0)[0][sorted_indices[index]]
            for index in range(len(sorted_indices))
        ]
        diagnostics["detection_count_mismatch"] += int(
            len(onnx_result.index) != len(paddle_result.index)
        )
        diagnostics["sorting_index_mismatch"] += int(
            onnx_result.index != paddle_result.index
            or onnx_result.index != expected_indices
        )
        crops_match = (
            len(onnx_result.text_crops) == len(paddle_result.text_crops)
            and all(
                np.array_equal(onnx_crop, paddle_crop)
                for onnx_crop, paddle_crop in zip(
                    onnx_result.text_crops, paddle_result.text_crops
                )
            )
        )
        diagnostics["orientation_arbitration_mismatch"] += int(
            not crops_match
        )
        fields_match = (
            onnx_result.class_id == paddle_result.class_id
            and onnx_result.texts == paddle_result.texts
            and onnx_result.Points == paddle_result.Points
            and np.array_equal(
                onnx_result.confidence, paddle_result.confidence
            )
        )
        diagnostics["field_mapping_mismatch"] += int(not fields_match)

        judge = PanelLabelJudgeApi.__new__(PanelLabelJudgeApi)
        standard = paddle_result.texts
        diagnostics["business_judgement_mismatch"] += int(
            judge.analyze(onnx_result, standard)
            != judge.analyze(paddle_result, standard)
        )
        if sample_number % 5 == 0:
            print(
                f"PARITY PROGRESS: samples={sample_number}/{len(samples)}, "
                f"crops={crop_count}"
            )

    assert crop_count > 0, "fixed sample set produced no OCR crops"
    print(
        "PARITY SUMMARY: "
        f"samples={len(samples)}, crops={crop_count}, "
        f"orientation_exact={orientation_matches}/{crop_count}, "
        f"text_exact={text_matches}/{recognition_count}, "
        f"orientation_score_max_abs={max_orientation_score_error:.8g}, "
        f"recognition_score_max_abs={max_recognition_score_error:.8g}, "
        f"orientation_logits_max_abs={max_orientation_logits_error:.8g}, "
        f"recognition_logits_max_abs={max_recognition_logits_error:.8g}"
    )
    print(
        "PARITY MISMATCH COUNTS: "
        + ", ".join(f"{name}={count}" for name, count in diagnostics.items())
    )
    print(
        "PARITY MAX LOCATIONS: "
        + ", ".join(
            f"{layer}=(sample={detail['sample']}, "
            f"tensor_index={detail['tensor_index']}, shape={detail['shape']}, "
            f"location={detail['location']}, error={detail['error']:.8g})"
            for layer, detail in sorted(max_details.items())
        )
    )
    assert diagnostics["orientation_input_mismatch"] == 0
    assert diagnostics["orientation_input_nonfinite"] == 0
    assert diagnostics["orientation_shape_mismatch"] == 0
    assert diagnostics["orientation_logits_nonfinite"] == 0
    assert diagnostics["orientation_logits_threshold_exceeded"] == 0
    assert max_orientation_logits_error <= LOGITS_MAX_ABS_ERROR
    assert diagnostics["orientation_class_mismatch"] == 0
    assert diagnostics["orientation_result_count_mismatch"] == 0
    assert diagnostics["orientation_score_threshold_exceeded"] == 0
    assert diagnostics["orientation_score_nonfinite"] == 0
    assert max_orientation_score_error <= SCORE_MAX_ABS_ERROR
    assert diagnostics["recognition_input_mismatch"] == 0
    assert diagnostics["recognition_input_nonfinite"] == 0
    assert diagnostics["recognition_shape_mismatch"] == 0
    assert diagnostics["recognition_logits_nonfinite"] == 0
    assert diagnostics["recognition_logits_threshold_exceeded"] == 0
    assert max_recognition_logits_error <= LOGITS_MAX_ABS_ERROR
    assert diagnostics["recognition_text_mismatch"] == 0
    assert diagnostics["recognition_result_count_mismatch"] == 0
    assert diagnostics["recognition_score_threshold_exceeded"] == 0
    assert diagnostics["recognition_score_nonfinite"] == 0
    assert max_recognition_score_error <= SCORE_MAX_ABS_ERROR
    assert diagnostics["detection_count_mismatch"] == 0
    assert diagnostics["sorting_index_mismatch"] == 0
    assert diagnostics["orientation_arbitration_mismatch"] == 0
    assert diagnostics["field_mapping_mismatch"] == 0
    assert diagnostics["business_judgement_mismatch"] == 0
