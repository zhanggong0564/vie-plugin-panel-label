"""Compare panel-label RF-DETR ONNX CUDA and TensorRT stage outputs.

Run from the framework repository root, for example::

    conda run -n mobile_vision python \
      plugins/vie-plugin-panel-label/examples/compare_rfdetr_onnx_trt.py \
      --onnx weights/panel_label/v2/rfdetr-seg-nano.onnx \
      --engine /tmp/rfdetr-seg-nano.fp32.engine \
      --sample-root demo/data/panel_label/charging_pile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import cv2
import numpy as np


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
for search_path in (REPOSITORY_ROOT, PLUGIN_ROOT):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

from services.inference import (  # noqa: E402
    OnnxRuntimeOptions,
    OnnxRuntimeRunner,
    TensorRTRunner,
)
from vie_plugin_panel_label.panel_label_detect import PanelLabelDetect  # noqa: E402


DEFAULT_ONNX_PATH = Path(
    "./weights/panel_label/v2/rfdetr-seg-nano.onnx"
)
DEFAULT_ENGINE_PATH = DEFAULT_ONNX_PATH.with_suffix(".fp32.engine")
DEFAULT_SAMPLE_ROOT = Path("./demo/data/panel_label/charging_pile")
DEFAULT_SAMPLE_COUNT = 10
DEFAULT_SEED = 20260715
CLASS_COUNT = 2
EXPECTED_OUTPUT_NAMES = frozenset({"dets", "labels", "masks"})
RAW_RTOL = 1e-2
RAW_ATOL = 2e-2
BOX_IOU_MIN = 0.99
MASK_IOU_MIN = 0.98
SCORE_MAX_ABS_ERROR = 0.01


def discover_samples(root: Path) -> list[Path]:
    """Discover image files below ``root`` while following symlinked roots."""
    supported_suffixes = {".jpg", ".jpeg", ".png"}
    samples: set[Path] = set()
    visited_directories: set[Path] = set()
    for directory, child_directories, filenames in os.walk(
        root, followlinks=True
    ):
        resolved_directory = Path(directory).resolve()
        if resolved_directory in visited_directories:
            child_directories.clear()
            continue
        visited_directories.add(resolved_directory)
        for filename in filenames:
            path = (Path(directory) / filename).resolve()
            if path.is_file() and path.suffix.lower() in supported_suffixes:
                samples.add(path)
    return sorted(samples)


def select_samples(root: Path, count: int, seed: int) -> list[Path]:
    if count < 1:
        raise ValueError("sample count must be at least 1")
    if not root.is_dir():
        raise FileNotFoundError(f"sample directory is unavailable: {root}")
    samples = discover_samples(root)
    if len(samples) < count:
        raise ValueError(
            f"need {count} real samples, found {len(samples)} under {root}"
        )
    return random.Random(seed).sample(samples, count)


def tensor_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    return {
        "shape": [int(dimension) for dimension in array.shape],
        "dtype": str(array.dtype),
        "contiguous": bool(array.flags.c_contiguous),
        "finite": bool(np.isfinite(array).all()),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def raw_output_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
) -> dict[str, float | int]:
    delta = np.abs(
        np.asarray(actual, dtype=np.float32)
        - np.asarray(expected, dtype=np.float32)
    )
    mismatch_count = int(
        np.count_nonzero(
            ~np.isclose(actual, expected, rtol=RAW_RTOL, atol=RAW_ATOL)
        )
    )
    if delta.size == 0:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "p95_abs": 0.0,
            "mismatch_count": 0,
            "mismatch_ratio": 0.0,
        }
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "p95_abs": float(np.percentile(delta, 95)),
        "mismatch_count": mismatch_count,
        "mismatch_ratio": mismatch_count / actual.size,
    }


def outputs_by_name(runner, outputs) -> dict[str, np.ndarray]:
    if len(runner.output_infos) != len(outputs):
        raise ValueError("runner output metadata count does not match outputs")
    named = {
        info.name: output
        for info, output in zip(runner.output_infos, outputs)
    }
    if len(named) != len(outputs):
        raise ValueError("runner output names must be unique")
    return named


def decode_candidates(
    outputs: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, np.ndarray]:
    logits = np.clip(outputs["labels"][0, :, :CLASS_COUNT], -88.0, 88.0)
    foreground_scores = 1.0 / (1.0 + np.exp(-logits))
    scores = foreground_scores.max(axis=1)
    class_ids = foreground_scores.argmax(axis=1)
    keep = scores > threshold
    return {
        "scores": scores,
        "class_ids": class_ids,
        "query_indices": np.flatnonzero(keep),
    }


def box_iou(actual: np.ndarray, expected: np.ndarray) -> float:
    top_left = np.maximum(actual[:2], expected[:2])
    bottom_right = np.minimum(actual[2:], expected[2:])
    intersection_size = np.maximum(bottom_right - top_left, 0.0)
    intersection = float(np.prod(intersection_size))
    actual_area = float(np.prod(np.maximum(actual[2:] - actual[:2], 0.0)))
    expected_area = float(
        np.prod(np.maximum(expected[2:] - expected[:2], 0.0))
    )
    union = actual_area + expected_area - intersection
    return intersection / union if union > 0.0 else 1.0


def mask_iou(actual: np.ndarray, expected: np.ndarray) -> float:
    actual_binary = np.asarray(actual, dtype=bool)
    expected_binary = np.asarray(expected, dtype=bool)
    if actual_binary.shape != expected_binary.shape:
        raise ValueError(
            f"mask shape mismatch: {actual_binary.shape} != "
            f"{expected_binary.shape}"
        )
    intersection = np.logical_and(actual_binary, expected_binary).sum()
    union = np.logical_or(actual_binary, expected_binary).sum()
    return float(intersection / union) if union else 1.0


def compare_sample(
    sample_id: str,
    sample_path: Path,
    onnx_runner,
    trt_runner,
    onnx_model: PanelLabelDetect,
    trt_model: PanelLabelDetect,
) -> dict[str, Any]:
    failures: list[str] = []
    image = cv2.imread(str(sample_path))
    if image is None:
        raise ValueError(f"failed to decode {sample_id}")

    tensor, meta = onnx_model.preprocess(image)
    meta.ori_img = image.copy()
    input_metrics = tensor_summary(tensor)
    if tensor.dtype != np.float32:
        failures.append(f"input dtype is {tensor.dtype}, expected float32")
    if not input_metrics["contiguous"]:
        failures.append("input tensor is not contiguous")
    if not input_metrics["finite"]:
        failures.append("input tensor contains non-finite values")

    onnx_output_list = onnx_runner.run(
        {onnx_runner.input_infos[0].name: tensor}
    )
    trt_output_list = trt_runner.run(
        {trt_runner.input_infos[0].name: tensor}
    )
    onnx_outputs = outputs_by_name(onnx_runner, onnx_output_list)
    trt_outputs = outputs_by_name(trt_runner, trt_output_list)
    if set(onnx_outputs) != set(trt_outputs) or set(onnx_outputs) != (
        EXPECTED_OUTPUT_NAMES
    ):
        raise ValueError(
            f"unexpected output names: onnx={sorted(onnx_outputs)} "
            f"trt={sorted(trt_outputs)}"
        )

    raw_metrics: dict[str, Any] = {}
    for output_name in sorted(EXPECTED_OUTPUT_NAMES):
        onnx_output = onnx_outputs[output_name]
        trt_output = trt_outputs[output_name]
        if trt_output.shape != onnx_output.shape:
            failures.append(
                f"raw {output_name} shape mismatch: "
                f"{trt_output.shape} != {onnx_output.shape}"
            )
            continue
        if trt_output.dtype != onnx_output.dtype:
            failures.append(
                f"raw {output_name} dtype mismatch: "
                f"{trt_output.dtype} != {onnx_output.dtype}"
            )
        if not np.isfinite(trt_output).all():
            failures.append(f"raw {output_name} TRT output is non-finite")
        if not np.isfinite(onnx_output).all():
            failures.append(f"raw {output_name} ONNX output is non-finite")
        raw_metrics[output_name] = {
            "shape": [int(value) for value in trt_output.shape],
            "dtype": str(trt_output.dtype),
            **raw_output_metrics(trt_output, onnx_output),
        }

    onnx_candidates = decode_candidates(
        onnx_outputs, onnx_model.confThreshold
    )
    trt_candidates = decode_candidates(trt_outputs, trt_model.confThreshold)
    onnx_query_indices = onnx_candidates["query_indices"]
    trt_query_indices = trt_candidates["query_indices"]
    candidate_metrics: dict[str, Any] = {
        "onnx_query_indices": onnx_query_indices.tolist(),
        "trt_query_indices": trt_query_indices.tolist(),
        "query_indices_equal": bool(
            np.array_equal(onnx_query_indices, trt_query_indices)
        ),
        "onnx_count": len(onnx_query_indices),
        "trt_count": len(trt_query_indices),
    }
    if len(onnx_query_indices) != len(trt_query_indices):
        failures.append(
            "candidate count mismatch: "
            f"TRT={len(trt_query_indices)} ONNX={len(onnx_query_indices)}"
        )

    pair_count = min(len(onnx_query_indices), len(trt_query_indices))
    onnx_query_indices = onnx_query_indices[:pair_count]
    trt_query_indices = trt_query_indices[:pair_count]
    onnx_classes = onnx_candidates["class_ids"][onnx_query_indices]
    trt_classes = trt_candidates["class_ids"][trt_query_indices]
    candidate_metrics["onnx_class_ids"] = onnx_classes.tolist()
    candidate_metrics["trt_class_ids"] = trt_classes.tolist()
    if not np.array_equal(onnx_classes, trt_classes):
        failures.append(
            f"candidate classes differ: TRT={trt_classes.tolist()} "
            f"ONNX={onnx_classes.tolist()}"
        )

    selected_raw_metrics = {}
    for output_name in sorted(EXPECTED_OUTPUT_NAMES):
        if output_name == "labels":
            onnx_selected = onnx_outputs[output_name][
                0, onnx_query_indices, :CLASS_COUNT
            ]
            trt_selected = trt_outputs[output_name][
                0, trt_query_indices, :CLASS_COUNT
            ]
        else:
            onnx_selected = onnx_outputs[output_name][
                0, onnx_query_indices
            ]
            trt_selected = trt_outputs[output_name][0, trt_query_indices]
        selected_raw_metrics[output_name] = raw_output_metrics(
            trt_selected, onnx_selected
        )
    candidate_metrics["matched_raw"] = selected_raw_metrics

    score_errors = np.abs(
        trt_candidates["scores"][trt_query_indices]
        - onnx_candidates["scores"][onnx_query_indices]
    )
    candidate_metrics["score_max_abs"] = float(
        score_errors.max(initial=0.0)
    )

    source_h, source_w = image.shape[:2]
    onnx_boxes = onnx_model._restore_boxes(
        onnx_outputs["dets"][0, onnx_query_indices], source_w, source_h
    )
    trt_boxes = trt_model._restore_boxes(
        trt_outputs["dets"][0, trt_query_indices], source_w, source_h
    )
    candidate_box_ious = [
        box_iou(trt_box, onnx_box)
        for trt_box, onnx_box in zip(trt_boxes, onnx_boxes)
    ]
    candidate_mask_ious = [
        mask_iou(
            trt_outputs["masks"][0, trt_query_index] > 0.0,
            onnx_outputs["masks"][0, onnx_query_index] > 0.0,
        )
        for onnx_query_index, trt_query_index in zip(
            onnx_query_indices, trt_query_indices
        )
    ]
    candidate_metrics["box_iou_min"] = min(
        candidate_box_ious, default=1.0
    )
    candidate_metrics["mask_iou_min"] = min(
        candidate_mask_ious, default=1.0
    )
    if candidate_metrics["score_max_abs"] > SCORE_MAX_ABS_ERROR:
        failures.append(
            "candidate score error exceeds threshold: "
            f"{candidate_metrics['score_max_abs']:.6f}"
        )
    if candidate_metrics["box_iou_min"] < BOX_IOU_MIN:
        failures.append(
            "candidate box IoU below threshold: "
            f"{candidate_metrics['box_iou_min']:.6f}"
        )
    if candidate_metrics["mask_iou_min"] < MASK_IOU_MIN:
        failures.append(
            "candidate mask IoU below threshold: "
            f"{candidate_metrics['mask_iou_min']:.6f}"
        )

    onnx_result = onnx_model.post_process(onnx_output_list, meta)
    trt_result = trt_model.post_process(trt_output_list, meta)
    final_metrics: dict[str, Any] = {
        "onnx_count": len(onnx_result.boxes),
        "trt_count": len(trt_result.boxes),
        "onnx_class_ids": onnx_result.class_ids,
        "trt_class_ids": trt_result.class_ids,
    }
    if len(trt_result.boxes) != len(onnx_result.boxes):
        failures.append(
            "final count mismatch: "
            f"TRT={len(trt_result.boxes)} ONNX={len(onnx_result.boxes)}"
        )
    if trt_result.class_ids != onnx_result.class_ids:
        failures.append("final class IDs differ")

    final_box_ious = [
        box_iou(np.asarray(trt_box), np.asarray(onnx_box))
        for trt_box, onnx_box in zip(trt_result.boxes, onnx_result.boxes)
    ]
    final_mask_ious = [
        mask_iou(trt_mask, onnx_mask)
        for trt_mask, onnx_mask in zip(trt_result.masks, onnx_result.masks)
    ]
    final_score_errors = [
        abs(float(trt_score) - float(onnx_score))
        for trt_score, onnx_score in zip(
            trt_result.scores, onnx_result.scores
        )
    ]
    final_metrics.update(
        {
            "box_iou_min": min(final_box_ious, default=1.0),
            "mask_iou_min": min(final_mask_ious, default=1.0),
            "score_max_abs": max(final_score_errors, default=0.0),
        }
    )
    if final_metrics["box_iou_min"] < BOX_IOU_MIN:
        failures.append(
            f"final box IoU below threshold: {final_metrics['box_iou_min']:.6f}"
        )
    if final_metrics["mask_iou_min"] < MASK_IOU_MIN:
        failures.append(
            "final mask IoU below threshold: "
            f"{final_metrics['mask_iou_min']:.6f}"
        )
    if final_metrics["score_max_abs"] > SCORE_MAX_ABS_ERROR:
        failures.append(
            "final score error exceeds threshold: "
            f"{final_metrics['score_max_abs']:.6f}"
        )

    return {
        "sample": sample_id,
        "path": str(sample_path),
        "passed": not failures,
        "failures": failures,
        "input": input_metrics,
        "raw_outputs": raw_metrics,
        "candidates": candidate_metrics,
        "final": final_metrics,
    }


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_detections = sum(
        int(result["final"]["onnx_count"]) for result in results
    )
    return {
        "sample_count": len(results),
        "passed_count": sum(bool(result["passed"]) for result in results),
        "failed_count": sum(not result["passed"] for result in results),
        "total_detections": total_detections,
        "candidate_box_iou_min": min(
            result["candidates"]["box_iou_min"] for result in results
        ),
        "candidate_mask_iou_min": min(
            result["candidates"]["mask_iou_min"] for result in results
        ),
        "candidate_score_max_abs": max(
            result["candidates"]["score_max_abs"] for result in results
        ),
        "final_box_iou_min": min(
            result["final"]["box_iou_min"] for result in results
        ),
        "final_mask_iou_min": min(
            result["final"]["mask_iou_min"] for result in results
        ),
        "final_score_max_abs": max(
            result["final"]["score_max_abs"] for result in results
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare RF-DETR ONNX CUDA and TensorRT on real panel samples"
        )
    )
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE_PATH)
    parser.add_argument(
        "--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT
    )
    parser.add_argument(
        "--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if not args.onnx.is_file():
        raise FileNotFoundError(f"ONNX model is unavailable: {args.onnx}")
    if not args.engine.is_file():
        raise FileNotFoundError(
            f"TensorRT engine is unavailable: {args.engine}"
        )
    samples = select_samples(args.sample_root, args.sample_count, args.seed)

    onnx_runner = OnnxRuntimeRunner(
        str(args.onnx),
        OnnxRuntimeOptions(
            providers=("CUDAExecutionProvider",),
            warmup=False,
            require_cuda=True,
        ),
    )
    trt_runner = TensorRTRunner(str(args.engine))
    results: list[dict[str, Any]] = []
    try:
        onnx_model = PanelLabelDetect(runner=onnx_runner)
        trt_model = PanelLabelDetect(runner=trt_runner)
        for index, sample_path in enumerate(samples, start=1):
            sample_id = f"sample-{index:02d}"
            result = compare_sample(
                sample_id,
                sample_path,
                onnx_runner,
                trt_runner,
                onnx_model,
                trt_model,
            )
            results.append(result)
            final = result["final"]
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{sample_id} {status} detections={final['onnx_count']} "
                f"box_iou_min={final['box_iou_min']:.6f} "
                f"mask_iou_min={final['mask_iou_min']:.6f} "
                f"score_max_abs={final['score_max_abs']:.6f}",
                flush=True,
            )
            for failure in result["failures"]:
                print(f"  - {failure}", flush=True)
    finally:
        trt_runner.close()
        onnx_runner.close()

    summary = aggregate_results(results)
    if summary["total_detections"] == 0:
        summary["failed_count"] += 1
        summary["failure"] = "selected samples produced no detections"
    report = {
        "config": {
            "onnx": str(args.onnx),
            "engine": str(args.engine),
            "sample_root": str(args.sample_root),
            "sample_count": args.sample_count,
            "seed": args.seed,
            "thresholds": {
                "box_iou_min": BOX_IOU_MIN,
                "mask_iou_min": MASK_IOU_MIN,
                "score_max_abs": SCORE_MAX_ABS_ERROR,
            },
        },
        "summary": summary,
        "samples": results,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        print(f"detailed report: {args.output}")
    return 0 if summary["failed_count"] == 0 else 1


def main() -> int:
    try:
        return run(parse_args())
    except Exception as exc:
        print(f"comparison failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
