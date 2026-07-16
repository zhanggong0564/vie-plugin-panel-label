"""Panel-label ONNX adapters compatible with PaddleOCR result contracts."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import yaml

from services.base import (
    BaseClassificationPipeline,
    BaseCtcRecognitionPipeline,
)
from services.inference import InferenceRunner


PANEL_LABEL_CHARACTERS = (
    "(", ")", "+", "-", ".", "/", "0", "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "A", "B", "C", "D", "E", "F", "G", "H",
    "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Z", "a", "b", "c", "d", "e", "n", "r",
)


def _load_metadata(metadata_path: str) -> dict:
    path = Path(metadata_path)
    if path.is_dir():
        path = path / "inference.yml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"invalid inference metadata: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("inference metadata must be a mapping")
    return data


def _transform(data: dict, name: str) -> dict:
    try:
        operations = data["PreProcess"]["transform_ops"]
    except (KeyError, TypeError) as exc:
        raise ValueError("metadata is missing PreProcess.transform_ops") from exc
    if not isinstance(operations, list):
        raise ValueError("PreProcess.transform_ops must be a list")
    if not all(
        isinstance(operation, dict) and len(operation) == 1
        for operation in operations
    ):
        raise ValueError("each PreProcess.transform_ops item must be a single-key mapping")
    matches = [operation[name] for operation in operations if name in operation]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(f"metadata must contain exactly one valid {name}")
    return matches[0]


def _three_floats(value, name: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain three numeric values") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be finite")
    return array


class PanelLabelOrientationClassifier(BaseClassificationPipeline):
    """PP-LCNet text-line orientation classifier backed by ONNX Runtime."""

    def __init__(
        self,
        metadata_path: str,
        *,
        runner: InferenceRunner,
    ) -> None:
        data = _load_metadata(metadata_path)
        resize = _transform(data, "ResizeImage")
        normalize = _transform(data, "NormalizeImage")
        size = resize.get("size")
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or not all(isinstance(value, int) and value > 0 for value in size)
        ):
            raise ValueError("ResizeImage.size must contain two positive integers")
        if normalize.get("channel_num") != 3 or normalize.get("order") != "":
            raise ValueError("NormalizeImage requires channel_num=3 and order='' ")
        try:
            scale = float(normalize["scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("NormalizeImage.scale must be numeric") from exc
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("NormalizeImage.scale must be positive and finite")
        mean = _three_floats(normalize.get("mean"), "mean")
        std = _three_floats(normalize.get("std"), "std")
        if np.any(std <= 0):
            raise ValueError("std values must be positive")
        try:
            topk = data["PostProcess"]["Topk"]
            labels = topk["label_list"]
        except (KeyError, TypeError) as exc:
            raise ValueError("metadata is missing Topk.label_list") from exc
        if topk.get("topk") != 1 or not isinstance(labels, list) or not labels:
            raise ValueError("Topk requires topk=1 and a non-empty label_list")

        self.input_width, self.input_height = size
        self.scale = np.float32(scale)
        self.mean = mean
        self.std = std
        super().__init__(runner, labels)

    def preprocess(self, images: Sequence[np.ndarray]) -> np.ndarray:
        tensors = []
        for image in images:
            resized = cv2.resize(
                image,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            normalized = (rgb * self.scale - self.mean) / self.std
            tensors.append(normalized.transpose(2, 0, 1))
        return np.ascontiguousarray(tensors, dtype=np.float32)


class PanelLabelTextRecognizer(BaseCtcRecognitionPipeline):
    """PP-OCRv5 dynamic-width CTC recognizer backed by ONNX Runtime."""

    def __init__(
        self,
        metadata_path: str,
        input_shape=None,
        *,
        runner: InferenceRunner,
    ) -> None:
        data = _load_metadata(metadata_path)
        decode = _transform(data, "DecodeImage")
        resize = _transform(data, "RecResizeImg")
        if (
            decode.get("img_mode") != "BGR"
            or decode.get("channel_first") is not False
        ):
            raise ValueError("DecodeImage requires img_mode='BGR' and channel_first=false")
        image_shape = resize.get("image_shape")
        if (
            not isinstance(image_shape, (list, tuple))
            or len(image_shape) < 3
            or not all(isinstance(value, int) and value > 0 for value in image_shape[:3])
            or image_shape[0] != 3
        ):
            raise ValueError("RecResizeImg.image_shape must start with [3, height, width]")
        try:
            postprocess = data["PostProcess"]
            characters = postprocess["character_dict"]
        except (KeyError, TypeError) as exc:
            raise ValueError("metadata is missing PostProcess.character_dict") from exc
        if postprocess.get("name") != "CTCLabelDecode":
            raise ValueError("PostProcess.name must be CTCLabelDecode")
        if (
            not isinstance(characters, list)
            or tuple(characters) != PANEL_LABEL_CHARACTERS
        ):
            raise ValueError("character_dict must match the exact 48-character alphabet")

        self.static_input_shape = None
        input_height = image_shape[1]
        self.metadata_width = image_shape[2]
        max_width = 3200
        if input_shape is not None:
            if (
                not isinstance(input_shape, (list, tuple))
                or len(input_shape) != 3
                or input_shape[0] != 3
                or not all(isinstance(value, int) and value > 0 for value in input_shape)
            ):
                raise ValueError("input_shape must be [3, height, width]")
            self.static_input_shape = tuple(input_shape)
            input_height = input_shape[1]
            max_width = input_shape[2]
        if runner.output_infos:
            output_shape = runner.output_infos[0].shape
            output_classes = output_shape[-1] if output_shape else None
            if (
                isinstance(output_classes, int)
                and output_classes > 0
                and output_classes != 49
            ):
                raise ValueError("recognition ONNX output must contain 49 classes")
        super().__init__(
            runner,
            characters,
            input_height=input_height,
            max_width=max_width,
        )

    def predict(self, images: Sequence[np.ndarray]) -> list[dict[str, str | float]]:
        """Recognize one ROI batch in a single backend invocation."""
        return super().predict(images)

    def _target_width(self, images: Sequence[np.ndarray]) -> int:
        if self.static_input_shape is None:
            super()._target_width(images)
            max_ratio = max(image.shape[1] / float(image.shape[0]) for image in images)
            return min(
                self.max_width,
                max(self.metadata_width, int(self.input_height * max_ratio)),
            )
        super()._target_width(images)
        return self.static_input_shape[2]

    def preprocess_image(self, image: np.ndarray, target_width: int) -> np.ndarray:
        if self.static_input_shape is not None:
            resized = cv2.resize(
                image,
                (target_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            output = resized.astype(np.float32).transpose(2, 0, 1) / 255
            return (output - 0.5) / 0.5

        ratio = image.shape[1] / float(image.shape[0])
        resized_width = min(target_width, int(np.ceil(self.input_height * ratio)))
        resized = cv2.resize(
            image,
            (resized_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255
        chw = (chw - 0.5) / 0.5
        output = np.zeros((3, self.input_height, target_width), np.float32)
        output[:, :, :resized_width] = chw
        return output
