from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from services.rfdetr import RFDetrInfer
from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
from vie_plugin_panel_label.config import PanelLabelConfig
from vie_plugin_panel_label.panel_label_detect import OCRPipeline, PanelLabelDetect


def _fake_runner():
    return SimpleNamespace(
        input_infos=[SimpleNamespace(name="images", shape=(1, 3, 640, 640))],
        output_infos=[],
        providers=["TestProvider"],
        close=MagicMock(),
    )


def test_rfdetr_accepts_injected_runner():
    runner = _fake_runner()

    model = RFDetrInfer(nc=2, runner=runner)

    assert model.runner is runner


def test_panel_label_detect_forwards_injected_runner():
    runner = _fake_runner()

    model = PanelLabelDetect(runner=runner)

    assert model.runner is runner


def test_ocr_pipeline_injects_all_three_runners():
    detection_runner = _fake_runner()
    orientation_runner = _fake_runner()
    recognition_runner = _fake_runner()

    with (
        patch(
            "vie_plugin_panel_label.panel_label_detect.PanelLabelDetect"
        ) as detect_class,
        patch(
            "vie_plugin_panel_label.panel_label_detect.PanelLabelOrientationClassifier"
        ) as orient_class,
        patch(
            "vie_plugin_panel_label.panel_label_detect.PanelLabelTextRecognizer"
        ) as recognizer_class,
    ):
        OCRPipeline(
            "ori/inference.yml",
            "rec/inference.yml",
            detection_runner=detection_runner,
            orientation_runner=orientation_runner,
            recognition_runner=recognition_runner,
        )

    detect_class.assert_called_once_with(
        0.5, 0.5, task="seg", runner=detection_runner
    )
    orient_class.assert_called_once_with(
        "ori/inference.yml", runner=orientation_runner
    )
    recognizer_class.assert_called_once_with(
        "rec/inference.yml",
        input_shape=None,
        runner=recognition_runner,
    )


def test_panel_label_engine_path_defaults_from_detection_onnx(monkeypatch):
    monkeypatch.delenv("PANEL_LABEL_TRT_ENGINE_PATH", raising=False)

    cfg = PanelLabelConfig()

    assert cfg.PANEL_LABEL_TRT_ENGINE_PATH == (
        cfg.model_path.removesuffix(".onnx") + ".fp32.engine"
    )


def test_panel_label_engine_path_is_exposed_as_class_default():
    assert PanelLabelConfig.PANEL_LABEL_TRT_ENGINE_PATH == (
        PanelLabelConfig.model_path.removesuffix(".onnx") + ".fp32.engine"
    )


def test_panel_label_engine_path_allows_environment_override(monkeypatch):
    monkeypatch.setenv("PANEL_LABEL_TRT_ENGINE_PATH", "/models/panel.engine")

    assert PanelLabelConfig().PANEL_LABEL_TRT_ENGINE_PATH == "/models/panel.engine"


def test_judge_initialization_uses_scene_backend_for_detection_only():
    runners = [object(), object(), object()]
    settings = MagicMock()
    settings.inference_backend_for.return_value = "tensorrt"

    with (
        patch(
            "vie_plugin_panel_label.business_logic.create_inference_runner",
            side_effect=runners,
        ) as runner_factory,
        patch(
            "vie_plugin_panel_label.business_logic.OCRPipeline"
        ) as pipeline,
    ):
        PanelLabelJudgeApi(settings)

    settings.inference_backend_for.assert_called_once_with("panel_label")
    options = OnnxRuntimeOptions.from_settings(settings)
    assert runner_factory.call_args_list == [
        call(
            RunnerSpec(
                    backend="tensorrt",
                    scenario="panel_label",
                    onnx_path="./weights/panel_label/v2/rfdetr-seg-nano.onnx",
                    engine_path=(
                        "./weights/panel_label/v2/"
                        "rfdetr-seg-nano.fp32.engine"
                    ),
                ),
            options,
        ),
        call(
            RunnerSpec(
                backend="onnx",
                scenario="panel_label",
                onnx_path="./weights/panel_label/v2/textline_ori_lcnet_v2.onnx",
            ),
            options,
        ),
        call(
            RunnerSpec(
                backend="onnx",
                scenario="panel_label",
                onnx_path="./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx",
            ),
            OnnxRuntimeOptions.from_settings(
                settings, execution_mode="sequential"
            ),
        ),
    ]
    pipeline.assert_called_once_with(
        "weights/panel_label/v2/textline_ori_lcnet_v2/inference.yml",
        "weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml",
        0.6,
        0.8,
        0.7,
        0.9,
        None,
        dedup_overlap_thresh=0.6,
        detection_runner=runners[0],
        orientation_runner=runners[1],
        recognition_runner=runners[2],
    )


def test_ocr_pipeline_closes_all_models_even_when_one_close_fails():
    pipeline = OCRPipeline.__new__(OCRPipeline)
    pipeline.detect_model = MagicMock()
    pipeline.text_orient_model = MagicMock()
    pipeline.text_rec_model = MagicMock()
    pipeline.text_orient_model.close.side_effect = RuntimeError("close failed")

    with pytest.raises(RuntimeError, match="close failed"):
        pipeline.close()

    pipeline.detect_model.close.assert_called_once_with()
    pipeline.text_orient_model.close.assert_called_once_with()
    pipeline.text_rec_model.close.assert_called_once_with()


def test_panel_initialization_closes_runners_created_before_failure():
    first_runner = MagicMock()
    settings = MagicMock()
    settings.inference_backend_for.return_value = "tensorrt"

    with patch(
        "vie_plugin_panel_label.business_logic.create_inference_runner",
        side_effect=[first_runner, RuntimeError("orientation failed")],
    ):
        with pytest.raises(Exception, match="panel_label 模型加载失败"):
            PanelLabelJudgeApi(settings)

    first_runner.close.assert_called_once_with()
