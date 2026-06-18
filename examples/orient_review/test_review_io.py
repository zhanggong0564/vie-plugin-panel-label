# orient_review/test_review_io.py
"""review_io 纯函数单测：旋转/备份/还原/候选与进度读写。不依赖模型与 GPU。

运行: cd orient_review && python -m pytest test_review_io.py -v
"""
import os

import cv2
import numpy as np
import pytest

import review_io as rio


@pytest.fixture
def env(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = np.zeros((20, 40, 3), np.uint8)
    img[0:5, :] = 255  # 顶部白条，上下不对称便于验证旋转
    p = str(img_dir / "t.jpg")
    cv2.imwrite(p, img)
    return {"path": p, "orig": cv2.imread(p).copy(), "bak": str(tmp_path / "bak")}


def test_rotate_then_restore_round_trip(env):
    rio.rotate_180(env["path"], backup_dir=env["bak"])
    rotated = cv2.imread(env["path"])
    assert np.array_equal(rotated, cv2.rotate(env["orig"], cv2.ROTATE_180))
    assert os.path.exists(os.path.join(env["bak"], "t.jpg"))
    rio.restore(env["path"], backup_dir=env["bak"])
    assert np.array_equal(cv2.imread(env["path"]), env["orig"])


def test_backup_once_keeps_true_original(env):
    # 翻转两次后备份仍应是最原始像素
    rio.rotate_180(env["path"], backup_dir=env["bak"])
    rio.rotate_180(env["path"], backup_dir=env["bak"])
    rio.restore(env["path"], backup_dir=env["bak"])
    assert np.array_equal(cv2.imread(env["path"]), env["orig"])


def test_restore_without_backup_raises(env):
    with pytest.raises(ValueError):
        rio.restore(env["path"], backup_dir=env["bak"])


def test_rotate_missing_file_raises(env):
    with pytest.raises(ValueError):
        rio.rotate_180(os.path.join(os.path.dirname(env["path"]), "nope.jpg"),
                       backup_dir=env["bak"])


def test_candidates_and_state_roundtrip(tmp_path):
    cj = str(tmp_path / "c.json")
    sj = str(tmp_path / "s.json")
    cands = [{"path": "/a/b.jpg", "score": 0.9}]
    rio.save_candidates(cands, cj)
    assert rio.load_candidates(cj) == cands
    assert rio.load_state(sj) == {}  # 不存在返回空
    rio.save_state({"/a/b.jpg": "rotated"}, sj)
    assert rio.load_state(sj) == {"/a/b.jpg": "rotated"}
