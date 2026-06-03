"""run.py 识别错误样本落盘 (PPOCR rec) 测试。"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

RUN_PY = Path(__file__).resolve().parents[1] / "examples" / "run.py"


@pytest.fixture(scope="module")
def run_mod():
    spec = importlib.util.spec_from_file_location("panel_label_run_example", RUN_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_item(texts, crops):
    from vie_plugin_panel_label.models import PanellabelItem
    return PanellabelItem(texts=texts, text_crops=crops)


def _make_result(message, details):
    from schemas.data_base import MoMResult, DetectionItem
    return MoMResult(
        detailList=[DetectionItem(status=s, name=n) for s, n in details],
        status=all(s for s, _ in details),
        message=message,
    )


def test_save_rec_hard_samples_mismatch(tmp_path, run_mod):
    """mismatch: 仅 status=False 的行落盘，标签取 standard[i]。"""
    crop0 = np.full((10, 30, 3), 7, np.uint8)
    crop1 = np.full((10, 30, 3), 9, np.uint8)
    item = _make_item(["WRONG", "OK1"], [crop0, crop1])
    result = _make_result("mismatch", [(False, "WRONG"), (True, "OK1")])

    run_mod.PRODUCT_TYPE["__TESTTYPE__"] = ["STD0", "STD1"]
    try:
        saved = run_mod.save_rec_hard_samples(str(tmp_path), "imgA", item, result, "__TESTTYPE__")
    finally:
        run_mod.PRODUCT_TYPE.pop("__TESTTYPE__", None)

    assert saved == 1
    assert (tmp_path / "images" / "imgA_line0.png").exists()
    assert not (tmp_path / "images" / "imgA_line1.png").exists()
    label = (tmp_path / "label.txt").read_text(encoding="utf-8").strip()
    assert label == "images/imgA_line0.png\tSTD0"


def test_save_rec_hard_samples_skips_non_mismatch(tmp_path, run_mod):
    """missing/extra: 数量不齐无法可靠对齐，不落盘。"""
    item = _make_item(["A"], [np.zeros((10, 30, 3), np.uint8)])
    result = _make_result("missing", [(False, "A")])

    run_mod.PRODUCT_TYPE["__TESTTYPE__"] = ["STD0", "STD1"]
    try:
        saved = run_mod.save_rec_hard_samples(str(tmp_path), "imgB", item, result, "__TESTTYPE__")
    finally:
        run_mod.PRODUCT_TYPE.pop("__TESTTYPE__", None)

    assert saved == 0
    assert not (tmp_path / "label.txt").exists()


def test_save_rec_hard_samples_appends(tmp_path, run_mod):
    """跨图追加写入 label.txt。"""
    item = _make_item(["W"], [np.zeros((10, 30, 3), np.uint8)])
    result = _make_result("mismatch", [(False, "W")])
    run_mod.PRODUCT_TYPE["__TESTTYPE__"] = ["STD0"]
    try:
        run_mod.save_rec_hard_samples(str(tmp_path), "img1", item, result, "__TESTTYPE__")
        run_mod.save_rec_hard_samples(str(tmp_path), "img2", item, result, "__TESTTYPE__")
    finally:
        run_mod.PRODUCT_TYPE.pop("__TESTTYPE__", None)

    lines = (tmp_path / "label.txt").read_text(encoding="utf-8").strip().splitlines()
    assert lines == ["images/img1_line0.png\tSTD0", "images/img2_line0.png\tSTD0"]
