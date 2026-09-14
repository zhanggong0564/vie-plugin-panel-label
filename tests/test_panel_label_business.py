"""panel_label 业务逻辑单元测试"""
import numpy as np
import json
import pytest
from unittest.mock import patch
from schemas.exceptions import InvalidParamsError
from schemas.inference_context import InferenceContext

@pytest.fixture
def api_instance(monkeypatch):
    """绕过 OCRPipeline 加载，构造 PanelLabelJudgeApi 实例"""
    with (
        patch("vie_plugin_panel_label.business_logic.create_inference_runner"),
        patch("vie_plugin_panel_label.business_logic.OCRPipeline"),
    ):
        from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
        from config import settings
        yield PanelLabelJudgeApi(settings)


def _make_ctx(result, product_type, w=1000, h=1000, rule="all", extra=None):
    ctx = InferenceContext(image=np.zeros((h, w, 3), dtype=np.uint8), h=h, w=w,
                           product_type=product_type, rule=rule, extra=extra or {})
    ctx.raw_result = result
    return ctx


def test_model_initialization_uses_direct_ocr_contract():
    from config import settings
    from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi

    runners = [object(), object(), object()]
    with (
        patch(
            "vie_plugin_panel_label.business_logic.create_inference_runner",
            side_effect=runners,
        ),
        patch("vie_plugin_panel_label.business_logic.OCRPipeline") as pipeline,
    ):
        PanelLabelJudgeApi(settings)

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
        mask_threshold=0.7,
        detection_runner=runners[0],
        orientation_runner=runners[1],
        recognition_runner=runners[2],
    )


class TestRequestParamsValidation:
    def test_missing_line_order_and_guideline_raises(self, api_instance):
        """standard_result / guideline 由请求下发，缺失时报参数错误"""
        from vie_plugin_panel_label.models import PanellabelItem
        ctx = _make_ctx(PanellabelItem(), "TK2")  # extra 为空，未携带判定基准
        with pytest.raises(InvalidParamsError) as exc_info:
            api_instance.business_post_process(ctx)
        assert "line_order" in exc_info.value.error_msg
        assert exc_info.value.context.get("scenario") == "panel_label"

    def test_guideline_coordinates_required_in_schema(self):
        """默认契约下调用方必须传 guideline_coordinates。"""
        from pydantic import ValidationError
        from vie_plugin_panel_label.schemas import ModelParams

        with pytest.raises(ValidationError):
            ModelParams(product_type="TK2", line_order="TK2-2,TK2-1")

        mp = ModelParams(
            product_type="TK2",
            line_order="TK2-2,TK2-1",
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )
        assert mp.guideline_coordinates == (0.1, 0.2, 0.3, 0.4)

    @pytest.mark.parametrize(
        ("line_order", "expected"),
        [
            ("A,B", [["A", "B"]]),
            (" A, B ; B, A ", [["A", "B"], ["B", "A"]]),
            (["A", "B"], [["A", "B"]]),
            ([[" A ", "B"], ["B", " A "]], [["A", "B"], ["B", "A"]]),
            ("A,B, null ,C", [["A", "B", None, "C"]]),
            (["A", "B", None, "C"], [["A", "B", None, "C"]]),
            ("NULL,A;A,NuLl", [[None, "A"], ["A", None]]),
            ([[None, "A"], ["A", "null"]], [[None, "A"], ["A", None]]),
            ([None, None], [[None, None]]),
            ("null,null", [[None, None]]),
            ("A,null,C,null", [["A", None, "C", None]]),
            (["A", None, None, "C"], [["A", None, None, "C"]]),
            ("A,B,,C", [["A", "B", "C"]]),
            (["A", "", " ", None], [["A", None]]),
            ("A,nlll,C", [["A", "nlll", "C"]]),
        ],
    )
    def test_line_order_parsed_as_candidates(self, line_order, expected):
        from vie_plugin_panel_label.schemas import ModelParams

        mp = ModelParams(
            product_type="TK2",
            line_order=line_order,
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )

        assert mp.line_order == expected
        assert mp.model_dump(mode="json")["line_order"] == expected
        assert ModelParams.model_validate_json(mp.model_dump_json()).line_order == expected

    def test_line_order_schema_allows_null_items(self):
        from vie_plugin_panel_label.schemas import ModelParams

        schema = ModelParams.model_json_schema()
        variants = schema["properties"]["line_order"]["anyOf"]
        assert {variant["type"] for variant in variants} == {"string", "array"}
        nested_array = next(
            variant for variant in variants
            if variant.get("type") == "array"
            and variant["items"].get("type") == "array"
        )
        item_schema = nested_array["items"]["items"]
        assert {item["type"] for item in item_schema["anyOf"]} == {"string", "null"}

    @pytest.mark.parametrize(
        "line_order",
        [
            "",
            ";",
            "A,B;",
            ["", " "],
            [["A"], []],
            ["A", ["B"]],
            [[1]],
            [None, 1],
            [[None, False]],
            None,
        ],
    )
    def test_invalid_line_order_candidates_raise(self, line_order):
        from pydantic import ValidationError
        from vie_plugin_panel_label.schemas import ModelParams

        with pytest.raises(ValidationError):
            ModelParams(
                product_type="TK2",
                line_order=line_order,
                guideline_coordinates="0.1,0.2,0.3,0.4",
            )

    def test_guideline_8_values_parsed(self):
        """8 值四边形可解析为长度 8 元组"""
        from vie_plugin_panel_label.schemas import ModelParams
        mp = ModelParams(
            product_type="SCUJ2",
            line_order="A,B",
            guideline_coordinates="0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9",
        )
        assert mp.guideline_coordinates == (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9)

    def test_guideline_4_values_still_parsed(self):
        """回归：4 值矩形仍照常解析"""
        from vie_plugin_panel_label.schemas import ModelParams
        mp = ModelParams(
            product_type="TK2",
            line_order="TK2-2,TK2-1",
            guideline_coordinates="0.1,0.2,0.3,0.4",
        )
        assert mp.guideline_coordinates == (0.1, 0.2, 0.3, 0.4)

    def test_guideline_invalid_length_raises(self):
        """非 4/8 长度（如 6 值）报校验错误"""
        from pydantic import ValidationError
        from vie_plugin_panel_label.schemas import ModelParams
        with pytest.raises(ValidationError):
            ModelParams(
                product_type="SCUJ2",
                line_order="A,B",
                guideline_coordinates="0.1,0.2,0.3,0.4,0.5,0.6",
            )


class TestCompareKeyNormalization:
    def test_zero_o_confusion_not_mismatch(self, api_instance):
        """线标字体下 OCR 区分不了 O/0（TCU-DO1 常读成 TCU-D01），比对须归一"""
        key = api_instance._compare_key
        assert key("TCU-D01-2", "all") == key("TCU-DO1-2", "all")
        assert key("TCU-reader-GND", "all") == key("TCU-Reader-GND", "all")

    def test_front_back_rules_still_split_on_slash(self, api_instance):
        key = api_instance._compare_key
        assert key("QF2-1/PE1-J1", "front") == "qf2-1"
        assert key("FU34-2/KM1-O1", "back") == "km1-01"


@pytest.mark.parametrize("line_order", ["A,B,null,C", ["A", "B", None, "C"]])
@pytest.mark.parametrize("skipped_text", ["ANY", "", None])
@pytest.mark.parametrize("last_text", ["C", "WRONG"])
def test_null_request_preserves_sorted_response_contract(
    api_instance, line_order, skipped_text, last_text, log_records
):
    from services.base import InspectionVerdict
    from vie_plugin_panel_label.models import PanellabelItem
    from vie_plugin_panel_label.plugin import panel_label_router

    request = panel_label_router.request_schema({
        "product": "test",
        "type": "test",
        "modelParams": {
            "product_type": "TK2",
            "line_order": line_order,
            "guideline_coordinates": "0,0,1,1",
        },
    })
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)
    inputs = panel_label_router.get_inputs(request, image)
    texts = ["A", "B", skipped_text, last_text]
    confidences = [0.95, 0.85, 0.1, 0.75]
    # 原始检测按倒序返回，验证 null 应用于空间排序后的第三位。
    indices = [3, 2, 1, 0]
    raw = PanellabelItem(
        Points=[
            [100 + i * 100, 100, 150 + i * 100, 100,
             150 + i * 100, 150, 100 + i * 100, 150]
            for i in indices
        ],
        index=indices,
        class_id=[0] * 4,
        texts=[texts[i] for i in indices],
        confidence=[confidences[i] for i in indices],
    )
    ctx = _make_ctx(raw, inputs.product_type, rule=inputs.rule, extra=inputs.extra)

    api_instance.business_post_process(ctx)

    expected_status = last_text == "C"
    expected_verdict = InspectionVerdict.PASS if expected_status else InspectionVerdict.FAIL
    assert ctx.result.status is expected_status
    assert ctx.result.verdict is expected_verdict
    assert [item.status for item in ctx.result.detailList] == [True, True, True, expected_status]
    assert ctx.result.detailList[2].verdict is InspectionVerdict.PASS
    assert [item.name for item in ctx.result.detailList] == ["A", "B", skipped_text or "", last_text]
    assert [item.accuracy for item in ctx.result.detailList] == confidences
    assert ctx.result.detailList[2].coordinate == [300, 100, 350, 100, 350, 150, 300, 150]
    serialized = ctx.result.to_dict()
    assert serialized["status"] == ("true" if expected_status else "false")
    assert serialized["verdict"] == expected_verdict.value
    assert serialized["detailList"][2]["status"] == "true"
    assert serialized["detailList"][2]["verdict"] == InspectionVerdict.PASS.value
    judged = [record for record in log_records if record["extra"]["event"] == "panel_label.judged"]
    assert len(judged) == 1
    summary = json.loads(judged[0]["message"].split(" ", 1)[1])
    assert summary["skipped_positions"] == [3]
    assert summary["unrecognized_positions"] == []
    assert summary["detection_scores"] == confidences
    assert summary["recognition_scores"] == [None] * 4
    assert summary["recognition_threshold"] == 0.7
    assert [item["position"] for item in summary["mismatches"]] == ([] if expected_status else [4])
    assert judged[0]["level"].name == ("INFO" if expected_status else "WARNING")


def test_panel_parameter_summary_keeps_decision_fields_only():
    from vie_plugin_panel_label.plugin import panel_label_router
    from utils.request_logging import log_json

    request = panel_label_router.request_schema({
        "product": "test", "type": "test",
        "modelParams": {
            "product_type": "T1", "rule": "front",
            "line_order": ",".join(f"LABEL-{index}" for index in range(70)),
            "guideline_coordinates": "0,0,1,1",
            "example_images": [{"FileName": "ref", "FilePath": "private-url" * 500}],
        },
        "AICameraModel": [{"unneeded": "extra-payload" * 100}],
    })
    logged = log_json(panel_label_router.request_log_params(request))
    summary = json.loads(logged)
    assert "private-url" not in logged
    assert "extra-payload" not in logged
    assert summary["example_image_count"] == 1
    assert summary["line_order"][0]["total"] == 70
    assert summary["line_order"][0]["omitted"] == 20
    assert summary["guideline_coordinates"] == [0, 0, 1, 1]
    assert summary["rule"] == "front"


class TestDetailNameFallback:
    def test_unrecognized_ocr_text_does_not_emit_none_name(self, api_instance):
        from vie_plugin_panel_label.models import PanellabelItem

        raw = PanellabelItem(
            Points=[[0, 0, 10, 0, 10, 10, 0, 10]],
            index=[0],
            class_id=[0],
            texts=[None],
            confidence=[0.8],
        )
        ctx = _make_ctx(
            raw,
            "TK2",
            extra={"standard_result": ["TK2-1"], "guideline": (0.0, 0.0, 1.0, 1.0)},
        )

        api_instance.business_post_process(ctx)

        assert ctx.result.detailList[0].name == ""
        assert isinstance(ctx.result.detailList[0].name, str)
