# orient_review/review_io.py
"""纯函数核心：候选/进度读写、原地旋转+备份、撤销还原。可单测。

与 annotation_review 的区别：那个改的是 cls 数据集的方向**标签**(txt)，
这里改的是 OCR rec 数据集的**图片像素**(原地旋转 180°)，识别标签(文本)不动。
"""
import json
import os
import shutil
import tempfile

import cv2
import numpy as np

# ===== 默认配置 =====
# 待复核的 OCR 识别数据集图片目录(仅作 predict_candidates 的默认值；
# review_tool 放行哪些图改由 candidates.json 决定，不再依赖此常量)
IMAGES_DIR = "/mnt/d/workspace/mobile_vision/data/annotated/line_marker/train/ocr/all/images"
# 备份目录：每张图就近落到 default_backup_dir(image_path)，无需全局常量
# predict_candidates.py 输出、工具读取的候选文件
CANDIDATES_JSON = os.path.join(os.path.dirname(__file__), "candidates.json")
# 复核进度文件
STATE_JSON = os.path.join(os.path.dirname(__file__), "review_state.json")
# 扫描状态文件(边扫边显示用：记录已扫/总数/是否扫完)
SCAN_JSON = os.path.join(os.path.dirname(__file__), "scan_status.json")


def _imread(path):
    """兼容中文路径读图。"""
    img = cv2.imread(path)
    if img is None:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def _imwrite_atomic(path, img):
    """原子写图(兼容中文路径)：先写临时文件再 os.replace。"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"图片编码失败: {path}")
    dir_name = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def default_backup_dir(image_path):
    """该图就近的备份目录：与其 images/ 目录同级的 orient_backup/。

    例: .../1+X/rec/images/x.png -> .../1+X/rec/orient_backup/x.png
    这样无论扫哪个目录，备份都落在对应数据集旁，不会串目录。
    """
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(image_path))), "orient_backup")


def backup_once(image_path, backup_dir=None):
    """覆盖前把原图快照备份一次；已备份则不动(保住最原始像素)。返回备份路径。"""
    backup_dir = backup_dir or default_backup_dir(image_path)
    os.makedirs(backup_dir, exist_ok=True)
    bak = os.path.join(backup_dir, os.path.basename(image_path))
    if not os.path.exists(bak):
        shutil.copy2(image_path, bak)
    return bak


def rotate_180(image_path, backup_dir=None):
    """把图片原地旋转 180° 覆盖，覆盖前 backup_once。失败抛异常。"""
    if not os.path.isfile(image_path):
        raise ValueError(f"图片不存在: {image_path}")
    img = _imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    backup_once(image_path, backup_dir)
    _imwrite_atomic(image_path, cv2.rotate(img, cv2.ROTATE_180))


def restore(image_path, backup_dir=None):
    """从备份还原原图(撤销翻转)。无备份则抛 ValueError。"""
    backup_dir = backup_dir or default_backup_dir(image_path)
    bak = os.path.join(backup_dir, os.path.basename(image_path))
    if not os.path.exists(bak):
        raise ValueError(f"无备份可还原: {image_path}")
    shutil.copy2(bak, image_path)


def load_candidates(candidates_json=None):
    candidates_json = candidates_json or CANDIDATES_JSON
    with open(candidates_json, "r", encoding="utf-8") as f:
        return json.load(f)


def save_candidates(candidates, candidates_json=None):
    _save_json_atomic(candidates_json or CANDIDATES_JSON, candidates)


def load_state(state_json=None):
    state_json = state_json or STATE_JSON
    if not os.path.exists(state_json):
        return {}
    with open(state_json, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state, state_json=None):
    _save_json_atomic(state_json or STATE_JSON, state)


def load_scan(scan_json=None):
    """读扫描状态；不存在返回 finished=True 的空状态(兼容无扫描进程的旧用法)。"""
    scan_json = scan_json or SCAN_JSON
    if not os.path.exists(scan_json):
        return {"done": 0, "total": 0, "found": 0, "finished": True}
    with open(scan_json, "r", encoding="utf-8") as f:
        return json.load(f)


def save_scan(scan, scan_json=None):
    _save_json_atomic(scan_json or SCAN_JSON, scan)


def _save_json_atomic(path, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
