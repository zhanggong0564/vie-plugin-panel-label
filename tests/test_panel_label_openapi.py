"""线标文档示例与真实解析契约。"""

from copy import deepcopy

import numpy as np
import pytest
from fastapi import FastAPI
from pydantic import ValidationError

from utils.openapi_docs import configure_openapi_docs
from vie_plugin_panel_label.plugin import panel_label_router


@pytest.mark.parametrize("line_order", [
    "TK2-2,null,TK2-1", ["TK2-2", None, "TK2-1"], [["TK2-2", None, "TK2-1"]],
])
@pytest.mark.parametrize("guideline", [
    "0.1,0.1,0.8,0.8", [0.1, 0.1, 0.8, 0.8],
    "0.1,0.1,0.9,0.1,0.9,0.9,0.1,0.9", [0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9],
])
def test_documented_input_formats_build_equivalent_inputs(line_order, guideline):
    payload = deepcopy(panel_label_router.request_document_example)
    payload["modelParams"].update(line_order=line_order, guideline_coordinates=guideline)
    request = panel_label_router.request_schema(payload)
    image = np.zeros((10, 10, 3), np.uint8)
    inputs = panel_label_router.get_inputs(request, image)
    assert inputs.image is image
    assert inputs.extra["standard_result"] == [["TK2-2", None, "TK2-1"]]
    expected = tuple(map(float, guideline.split(","))) if isinstance(guideline, str) else tuple(guideline)
    assert inputs.extra["guideline"] == expected
    assert inputs.rule == "all"


@pytest.mark.parametrize("field,value", [
    ("line_order", " ; "), ("line_order", []),
    ("guideline_coordinates", "0,0,1"), ("rule", "invalid"),
])
def test_documented_invalid_inputs_are_rejected(field, value):
    payload = deepcopy(panel_label_router.request_document_example)
    payload["modelParams"][field] = value
    with pytest.raises(ValidationError):
        panel_label_router.request_schema(payload)


def test_openapi_publishes_input_unions_and_plugin_example():
    app = FastAPI()
    app.include_router(panel_label_router.get_router())
    configure_openapi_docs(app)
    schema = app.openapi()
    schemas = schema["components"]["schemas"]
    operation = schema["paths"]["/panel_label_detect"]["post"]
    body_ref = operation["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"]
    field = schemas[body_ref.rsplit("/", 1)[-1]]["properties"]["json_data"]
    request = schemas[field["x-json-schema"]["$ref"].rsplit("/", 1)[-1]]
    params = schemas[request["properties"]["modelParams"]["$ref"].rsplit("/", 1)[-1]]
    for name in ("line_order", "guideline_coordinates"):
        assert {item["type"] for item in params["properties"][name]["anyOf"]} == {"string", "array"}
    assert "null" in operation["description"]
    assert "4 值" in operation["description"] and "8 值" in operation["description"]
    request = panel_label_router.request_schema(deepcopy(panel_label_router.request_document_example))
    assert request.modelParams.line_order == [["TK2-2", None, "TK2-1"], ["TK2-1", None, "TK2-2"]]
