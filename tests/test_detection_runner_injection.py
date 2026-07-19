from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from services.inference import OnnxRuntimeOptions, RunnerSpec
from services.rfdetr import RFDetrInfer
from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
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


def test_panel_label_detect_uses_polygon_only_masks_by_default():
    model = PanelLabelDetect(runner=_fake_runner())

    assert model.mask_output == "polygons_only"


def test_panel_label_detect_can_restore_full_masks():
    model = PanelLabelDetect(
        runner=_fake_runner(), cpu_fast_path=False
    )

    assert model.mask_output == "full"


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
        0.5,
        0.5,
        task="seg",
        runner=detection_runner,
        cpu_fast_path=True,
    )
    orient_class.assert_called_once_with(
        "ori/inference.yml", runner=orientation_runner
    )
    recognizer_class.assert_called_once_with(
        "rec/inference.yml",
        input_shape=None,
        runner=recognition_runner,
    )


def test_judge_initialization_creates_three_onnx_runners():
    runners = [object(), object(), object()]
    settings = MagicMock()

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

    options = OnnxRuntimeOptions.from_settings(settings)
    assert runner_factory.call_args_list == [
        call(
            RunnerSpec(
                scenario="panel_label",
                onnx_path="./weights/panel_label/v2/rfdetr-seg-nano.onnx",
            ),
            options,
        ),
        call(
            RunnerSpec(
                scenario="panel_label",
                onnx_path="./weights/panel_label/v2/textline_ori_lcnet_v2.onnx",
            ),
            options,
        ),
        call(
            RunnerSpec(
                scenario="panel_label",
                onnx_path="./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx",
            ),
            OnnxRuntimeOptions.from_settings(
                settings, execution_mode="sequential"
            ),
        ),
    ]
    pipeline.assert_called_once_with(
        "./weights/panel_label/v2/textline_ori_lcnet_v2/inference.yml",
        "./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml",
        0.6,
        0.8,
        0.7,
        0.9,
        None,
        dedup_overlap_thresh=0.6,
        cpu_fast_path=True,
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


@pytest.mark.parametrize("failure_index", [1, 2])
def test_panel_initialization_closes_all_runners_created_before_failure(
    failure_index,
):
    created_runners = [MagicMock() for _ in range(failure_index)]
    settings = MagicMock()
    side_effects = [*created_runners, RuntimeError("runner failed")]

    with patch(
        "vie_plugin_panel_label.business_logic.create_inference_runner",
        side_effect=side_effects,
    ):
        with pytest.raises(Exception, match="panel_label 模型加载失败"):
            PanelLabelJudgeApi(settings)

    for runner in created_runners:
        runner.close.assert_called_once_with()
