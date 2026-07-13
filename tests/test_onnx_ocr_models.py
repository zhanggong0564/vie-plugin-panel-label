"""ONNX OCR adapters must reproduce PaddleOCR 3.4.0 preprocessing."""

from pathlib import Path

import cv2
import numpy as np
from packaging.requirements import Requirement
import pytest
import tomli
import yaml

from services.base import TensorInfo
from vie_plugin_panel_label.ocr_models import (
    PanelLabelOrientationClassifier,
    PanelLabelTextRecognizer,
)


CHARACTERS = (
    "(", ")", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Z", "a", "b", "c", "d", "e", "n", "r",
)


def test_plugin_declares_direct_yaml_dependency():
    pyproject = Path(__file__).parents[1] / "pyproject.toml"
    metadata = tomli.loads(pyproject.read_text(encoding="utf-8"))
    dependency_names = {
        Requirement(value).name.lower()
        for value in metadata["project"]["dependencies"]
    }

    assert "pyyaml" in dependency_names


class StubRunner:
    input_infos = (TensorInfo("x", (None, 3, 48, None), "tensor(float)"),)
    providers = ("CPUExecutionProvider",)

    def __init__(self, output_classes=49):
        self.output_infos = (
            TensorInfo("output", (None, None, output_classes), "tensor(float)"),
        )

    def run(self, inputs):
        return [np.zeros((len(inputs["x"]), 1, 49), dtype=np.float32)]


class RecordingRunner(StubRunner):
    def __init__(self):
        super().__init__()
        self.input_shapes = []

    def run(self, inputs):
        tensor = inputs["x"]
        self.input_shapes.append(tensor.shape)
        logits = np.zeros((len(tensor), 1, 49), dtype=np.float32)
        logits[:, :, 7 + len(self.input_shapes) - 1] = 1.0
        return [logits]


class FalseyRunner(StubRunner):
    def __bool__(self):
        return False


@pytest.fixture
def metadata_files(tmp_path: Path):
    orientation = tmp_path / "orientation.yml"
    orientation.write_text(
        yaml.safe_dump(
            {
                "PreProcess": {"transform_ops": [
                    {"ResizeImage": {"size": [160, 80]}},
                    {"NormalizeImage": {
                        "channel_num": 3,
                        "mean": [0.485, 0.456, 0.406],
                        "order": "",
                        "scale": 1 / 255,
                        "std": [0.229, 0.224, 0.225],
                    }},
                    {"ToCHWImage": None},
                ]},
                "PostProcess": {"Topk": {
                    "topk": 1,
                    "label_list": ["0_degree", "180_degree"],
                }},
            }
        ),
        encoding="utf-8",
    )
    recognition = tmp_path / "recognition.yml"
    recognition.write_text(
        yaml.safe_dump(
            {
                "PreProcess": {"transform_ops": [
                    {"DecodeImage": {"channel_first": False, "img_mode": "BGR"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                ]},
                "PostProcess": {
                    "name": "CTCLabelDecode",
                    "character_dict": list(CHARACTERS),
                },
            }
        ),
        encoding="utf-8",
    )
    return orientation, recognition


def test_orientation_preprocess_matches_reference(metadata_files):
    orientation, _ = metadata_files
    classifier = PanelLabelOrientationClassifier(
        "orientation.onnx", str(orientation), runner=StubRunner()
    )
    bgr = np.arange(31 * 79 * 3, dtype=np.uint8).reshape(31, 79, 3)

    resized = cv2.resize(bgr, (160, 80), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    expected = rgb * np.float32(1 / 255)
    expected = (expected - np.array([0.485, 0.456, 0.406], np.float32))
    expected /= np.array([0.229, 0.224, 0.225], np.float32)
    expected = expected.transpose(2, 0, 1)[None]

    np.testing.assert_allclose(
        classifier.preprocess([bgr]), expected, rtol=1e-5, atol=1e-6
    )


def test_orientation_preserves_explicit_falsey_runner(metadata_files):
    orientation, _ = metadata_files
    runner = FalseyRunner()

    classifier = PanelLabelOrientationClassifier(
        "missing-orientation.onnx", str(orientation), runner=runner
    )

    assert classifier.runner is runner


def test_recognition_metadata_loads_all_48_characters(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=StubRunner()
    )
    assert recognizer.characters == CHARACTERS
    assert len(recognizer.characters) == 48


def test_recognition_preserves_explicit_falsey_runner(metadata_files):
    _, recognition = metadata_files
    runner = FalseyRunner()

    recognizer = PanelLabelTextRecognizer(
        "missing-recognition.onnx", str(recognition), runner=runner
    )

    assert recognizer.runner is runner


def test_recognition_preprocess_matches_paddleocr_dynamic_width(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=StubRunner()
    )
    # 48 * (643 / 96) == 321.5: Paddle floors the canvas to 321,
    # while its per-image ceil width (322) is truncated to that canvas.
    bgr = np.arange(96 * 643 * 3, dtype=np.uint8).reshape(96, 643, 3)

    resized = cv2.resize(bgr, (321, 48), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32).transpose(2, 0, 1) / 255
    expected = ((expected - 0.5) / 0.5)[None]

    actual = recognizer.preprocess_batch([bgr])
    assert actual.shape == (1, 3, 48, 321)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_recognition_batch_uses_metadata_width_as_dynamic_minimum(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=StubRunner()
    )
    narrow = np.zeros((48, 64, 3), dtype=np.uint8)

    assert recognizer.preprocess_batch([narrow]).shape == (1, 3, 48, 320)
    bgr = np.arange(25 * 37 * 3, dtype=np.uint8).reshape(25, 37, 3)
    target_width = 72

    resized_width = min(target_width, int(np.ceil(48 * 37 / 25)))
    resized = cv2.resize(bgr, (resized_width, 48), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32).transpose(2, 0, 1) / 255
    expected = (expected - 0.5) / 0.5
    padded = np.zeros((3, 48, target_width), dtype=np.float32)
    padded[:, :, :resized_width] = expected

    np.testing.assert_allclose(
        recognizer.preprocess_image(bgr, target_width),
        padded,
        rtol=1e-5,
        atol=1e-6,
    )


def test_invalid_orientation_metadata_fails_during_initialization(
    metadata_files, tmp_path
):
    orientation, _ = metadata_files
    data = yaml.safe_load(orientation.read_text(encoding="utf-8"))
    data["PreProcess"]["transform_ops"][1]["NormalizeImage"]["std"] = [1, 1]
    invalid = tmp_path / "invalid.yml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="std"):
        PanelLabelOrientationClassifier(
            "orientation.onnx", str(invalid), runner=StubRunner()
        )


@pytest.mark.parametrize(
    "characters",
    [
        list(CHARACTERS[:-1]),
        [*CHARACTERS[:-1], "x"],
        None,
    ],
)
def test_recognition_rejects_character_dict_other_than_scene_alphabet(
    metadata_files, tmp_path, characters
):
    _, recognition = metadata_files
    data = yaml.safe_load(recognition.read_text(encoding="utf-8"))
    data["PostProcess"]["character_dict"] = characters
    invalid = tmp_path / "invalid_characters.yml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="exact 48-character"):
        PanelLabelTextRecognizer(
            "recognition.onnx", str(invalid), runner=StubRunner()
        )


def test_recognition_rejects_static_onnx_output_other_than_49_classes(
    metadata_files,
):
    _, recognition = metadata_files

    with pytest.raises(ValueError, match="49 classes"):
        PanelLabelTextRecognizer(
            "recognition.onnx", str(recognition), runner=StubRunner(50)
        )


@pytest.mark.parametrize(
    "decode_image",
    [
        {"channel_first": False, "img_mode": "RGB"},
        {"channel_first": True, "img_mode": "BGR"},
    ],
)
def test_recognition_rejects_non_bgr_decode_metadata(
    metadata_files, tmp_path, decode_image
):
    _, recognition = metadata_files
    data = yaml.safe_load(recognition.read_text(encoding="utf-8"))
    data["PreProcess"]["transform_ops"][0]["DecodeImage"] = decode_image
    invalid = tmp_path / "invalid_decode.yml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="DecodeImage requires"):
        PanelLabelTextRecognizer(
            "recognition.onnx", str(invalid), runner=StubRunner()
        )


def test_recognition_rejects_missing_decode_metadata(metadata_files, tmp_path):
    _, recognition = metadata_files
    data = yaml.safe_load(recognition.read_text(encoding="utf-8"))
    data["PreProcess"]["transform_ops"].pop(0)
    invalid = tmp_path / "missing_decode.yml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="DecodeImage"):
        PanelLabelTextRecognizer(
            "recognition.onnx", str(invalid), runner=StubRunner()
        )


def test_recognition_dynamic_canvas_is_capped_at_3200(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=StubRunner()
    )
    very_wide = np.zeros((48, 4000, 3), dtype=np.uint8)

    assert recognizer.preprocess_batch([very_wide]).shape == (1, 3, 48, 3200)


def test_recognition_static_input_shape_stretches_without_padding(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx",
        str(recognition),
        input_shape=[3, 32, 100],
        runner=StubRunner(),
    )
    bgr = np.arange(20 * 35 * 3, dtype=np.uint8).reshape(20, 35, 3)
    resized = cv2.resize(bgr, (100, 32), interpolation=cv2.INTER_LINEAR)
    expected = resized.astype(np.float32).transpose(2, 0, 1) / 255
    expected = ((expected - 0.5) / 0.5)[None]

    np.testing.assert_allclose(
        recognizer.preprocess_batch([bgr]), expected, rtol=1e-5, atol=1e-6
    )


def test_recognition_predict_preserves_ctc_result_contract(metadata_files):
    _, recognition = metadata_files
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=StubRunner()
    )

    assert recognizer.predict([np.zeros((48, 64, 3), dtype=np.uint8)]) == [
        {"rec_text": "", "rec_score": 0.0}
    ]


def test_recognition_predict_runs_each_crop_at_its_own_dynamic_width(
    metadata_files,
):
    _, recognition = metadata_files
    runner = RecordingRunner()
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=runner
    )
    narrow = np.zeros((48, 64, 3), dtype=np.uint8)
    wide = np.zeros((96, 643, 3), dtype=np.uint8)

    assert recognizer.predict([narrow, wide]) == [
        {"rec_text": "0", "rec_score": 1.0},
        {"rec_text": "1", "rec_score": 1.0},
    ]
    assert runner.input_shapes == [(1, 3, 48, 320), (1, 3, 48, 321)]


def test_recognition_predict_empty_input_does_not_call_runner(metadata_files):
    _, recognition = metadata_files
    runner = RecordingRunner()
    recognizer = PanelLabelTextRecognizer(
        "recognition.onnx", str(recognition), runner=runner
    )

    assert recognizer.predict([]) == []
    assert runner.input_shapes == []


def test_malformed_transform_operation_raises_clear_value_error(
    metadata_files, tmp_path
):
    orientation, _ = metadata_files
    data = yaml.safe_load(orientation.read_text(encoding="utf-8"))
    data["PreProcess"]["transform_ops"].insert(0, None)
    invalid = tmp_path / "invalid_transform.yml"
    invalid.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ValueError, match="transform_ops.*mapping"):
        PanelLabelOrientationClassifier(
            "orientation.onnx", str(invalid), runner=StubRunner()
        )
