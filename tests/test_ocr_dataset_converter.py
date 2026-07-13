"""tools/convert_ocr_dataset_to_ppocr.py 的单元测试。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.data.convert_ocr_dataset_to_ppocr import Sample, find_samples


def _write_labelme(json_path: Path, image_name: str, shapes: list[dict]) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(
            {
                "version": "3.3.9",
                "flags": {},
                "shapes": shapes,
                "imagePath": image_name,
                "imageData": None,
                "imageHeight": 100,
                "imageWidth": 400,
                "description": "",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _make_shape(description: str = "ABC", difficult: bool = False) -> dict:
    return {
        "label": "text",
        "score": 1.0,
        "points": [[10.0, 10.0], [90.0, 10.0], [90.0, 30.0], [10.0, 30.0]],
        "group_id": 0,
        "description": description,
        "difficult": difficult,
        "shape_type": "polygon",
        "flags": None,
        "attributes": {},
        "kie_linking": [],
    }


@pytest.fixture
def mini_dataset(tmp_path: Path) -> Path:
    """构造两侧两个工位、共 3 张样本的最小数据集。"""
    root = tmp_path / "dataset"
    img_bytes = b"\x89PNG\r\n\x1a\n"  # 仅作为占位，不会被解码（除非测试明确读图）

    # 交流侧/J46/IMG_1
    j46 = root / "ac" / "J46" / "crop_ocr"
    (j46 / "images").mkdir(parents=True)
    (j46 / "images" / "IMG_1.jpg").write_bytes(img_bytes)
    _write_labelme(j46 / "jsons" / "IMG_1.json", "IMG_1.jpg", [_make_shape("J46-A/PW-1")])

    # 交流侧/J46/IMG_2
    (j46 / "images" / "IMG_2.jpg").write_bytes(img_bytes)
    _write_labelme(j46 / "jsons" / "IMG_2.json", "IMG_2.jpg", [_make_shape("J46-B/PW-2")])

    # 直流侧/T1/IMG_3
    t1 = root / "dc" / "T1" / "crop_ocr"
    (t1 / "images").mkdir(parents=True)
    (t1 / "images" / "IMG_3.jpg").write_bytes(img_bytes)
    _write_labelme(t1 / "jsons" / "IMG_3.json", "IMG_3.jpg", [_make_shape("T1-X/KM-1")])

    return root


def test_find_samples_collects_all(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    assert len(samples) == 3
    stations = sorted(s.station_code for s in samples)
    assert stations == ["J46", "J46", "T1"]
    stems = sorted(s.original_stem for s in samples)
    assert stems == ["IMG_1", "IMG_2", "IMG_3"]


def test_sample_image_path_exists(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    for s in samples:
        assert isinstance(s, Sample)
        assert s.image_path.exists()
        assert s.json_path.exists()
        assert s.station_code in {"J46", "T1"}


from scripts.data.convert_ocr_dataset_to_ppocr import (
    build_det_annotation,
    det_filename,
    rec_filename,
    split_samples,
)


def test_split_deterministic_with_seed(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    # 3 个样本，val_ratio=0.34 → n_val=1
    train1, val1 = split_samples(samples, val_ratio=0.34, seed=42)
    train2, val2 = split_samples(samples, val_ratio=0.34, seed=42)
    assert [s.original_stem for s in train1] == [s.original_stem for s in train2]
    assert [s.original_stem for s in val1] == [s.original_stem for s in val2]
    assert len(train1) + len(val1) == 3
    assert len(val1) == 1


def test_split_zero_val(mini_dataset: Path) -> None:
    samples = find_samples(mini_dataset)
    train, val = split_samples(samples, val_ratio=0.0, seed=42)
    assert len(val) == 0
    assert len(train) == 3


def test_det_filename_no_chinese() -> None:
    name = det_filename("IMG_1", "J46", "jpg")
    assert name == "IMG_1_det_J46.jpg"
    assert all(ord(c) < 128 for c in name), "filename must be pure ASCII"


def test_rec_filename_format() -> None:
    assert rec_filename("IMG_2", "T1") == "IMG_2_rec_T1.png"


def test_expand_box_ratio_zero_is_noop() -> None:
    """ratio<=0 时原样返回。"""
    import numpy as np
    from scripts.data.convert_ocr_dataset_to_ppocr import _expand_box

    box = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.float32)
    assert np.array_equal(_expand_box(box, 0.0), box)


def test_expand_box_grows_by_short_side_ratio() -> None:
    """轴对齐框：宽高应各增长 2×ratio×短边，且仍居中。"""
    import numpy as np
    from scripts.data.convert_ocr_dataset_to_ppocr import _expand_box

    # 宽 100、高 30 的轴对齐框，短边=30，ratio=0.2 → 每边外扩 6px
    box = np.array([[10, 10], [110, 10], [110, 40], [10, 40]], dtype=np.float32)
    out = _expand_box(box, 0.2)

    d = 0.2 * 30  # = 6
    expected = np.array([[10 - d, 10 - d], [110 + d, 10 - d], [110 + d, 40 + d], [10 - d, 40 + d]], dtype=np.float32)
    assert np.allclose(out, expected, atol=1e-4)
    # 中心不变
    assert np.allclose(out.mean(axis=0), box.mean(axis=0), atol=1e-4)


def test_build_det_annotation_normal(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("HELLO")])
    ann = build_det_annotation(j)
    assert len(ann) == 1
    assert ann[0]["transcription"] == "HELLO"
    assert ann[0]["points"] == [[10, 10], [90, 10], [90, 30], [10, 30]]


def test_build_det_annotation_empty_description_becomes_hash(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("")])
    ann = build_det_annotation(j)
    assert ann[0]["transcription"] == "###"


def test_build_det_annotation_difficult_becomes_hash(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    _write_labelme(j, "a.jpg", [_make_shape("text", difficult=True)])
    ann = build_det_annotation(j)
    assert ann[0]["transcription"] == "###"


def test_build_det_annotation_skips_short_points(tmp_path: Path) -> None:
    j = tmp_path / "a.json"
    bad = _make_shape("x")
    bad["points"] = [[0.0, 0.0], [10.0, 10.0]]  # 仅 2 个点
    _write_labelme(j, "a.jpg", [bad])
    assert build_det_annotation(j) == []


from scripts.data.convert_ocr_dataset_to_ppocr import write_det_split


def test_write_det_split_creates_images_and_labels(mini_dataset: Path, tmp_path: Path) -> None:
    samples = find_samples(mini_dataset)
    train, val = split_samples(samples, val_ratio=0.34, seed=42)

    det_dir = tmp_path / "out" / "det"
    train_stats = write_det_split(train, det_dir, "train.txt")
    val_stats = write_det_split(val, det_dir, "val.txt")

    # 标签文件 + 图片
    assert (det_dir / "train.txt").exists()
    assert (det_dir / "val.txt").exists()
    images = list((det_dir / "images").iterdir())
    assert len(images) == 3
    for img in images:
        assert "_det_" in img.name
        assert img.suffix in {".jpg", ".jpeg", ".png", ".bmp"}
        assert all(ord(c) < 128 for c in img.name)

    # 标签行格式
    line = (det_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0]
    path_part, ann_part = line.split("\t")
    assert path_part.startswith("images/")
    ann = json.loads(ann_part)
    assert isinstance(ann, list) and "transcription" in ann[0]

    assert train_stats["kept"] + val_stats["kept"] == 3


def test_write_det_split_skips_empty_shape_image(tmp_path: Path) -> None:
    # 构造一个没有有效 shape 的样本
    root = tmp_path / "data"
    p = root / "ac" / "X1" / "crop_ocr"
    (p / "images").mkdir(parents=True)
    (p / "images" / "IMG_E.jpg").write_bytes(b"x")
    _write_labelme(p / "jsons" / "IMG_E.json", "IMG_E.jpg", shapes=[])
    samples = find_samples(root)

    det_dir = tmp_path / "out" / "det"
    stats = write_det_split(samples, det_dir, "train.txt")
    assert stats["kept"] == 0
    assert stats["empty_shape"] == 1
    # 既然全跳过，train.txt 仍应存在但为空
    assert (det_dir / "train.txt").exists()
    assert (det_dir / "train.txt").read_text() == ""


from scripts.data.convert_ocr_dataset_to_ppocr import write_dict


def test_write_dict_unique_sorted(tmp_path: Path) -> None:
    path = tmp_path / "dict.txt"
    n = write_dict(["ABC", "BCD", "A1"], path)
    chars = path.read_text(encoding="utf-8").splitlines()
    assert chars == sorted(set("ABCD1"))
    assert n == len(chars)


def test_write_dict_empty_produces_empty_file(tmp_path: Path) -> None:
    """空 transcriptions 不应产生孤零零的换行符。"""
    path = tmp_path / "dict.txt"
    n = write_dict([], path)
    assert n == 0
    assert path.read_text(encoding="utf-8") == ""


def test_split_alignment_det_train_val_disjoint(mini_dataset: Path, tmp_path: Path) -> None:
    """同一 sample 在 det 中只能落入 train 或 val 其中之一；写出的 stem 集合与 split 列表一致。"""
    samples = find_samples(mini_dataset)
    train_s, val_s = split_samples(samples, val_ratio=0.34, seed=42)

    det_dir = tmp_path / "det"
    write_det_split(train_s, det_dir, "train.txt")
    write_det_split(val_s, det_dir, "val.txt")

    def stems_from(label_path: Path) -> set[str]:
        return {
            line.split("\t")[0].removeprefix("images/").split("_det_")[0]
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line
        }

    det_train = stems_from(det_dir / "train.txt")
    det_val = stems_from(det_dir / "val.txt")
    assert not (det_train & det_val), "stem 跨 split"
    assert det_train == {s.original_stem for s in train_s}
    assert det_val == {s.original_stem for s in val_s}


# ---------------------------------------------------------------------------
# rec 像素一致性单测（需要真实 paddleocr 模型）
# ---------------------------------------------------------------------------

import cv2  # noqa: E402


@pytest.fixture(scope="module")
def rec_pipeline():
    """模块级 fixture：加载一次模型。"""
    pytest.importorskip("paddleocr")
    pytest.importorskip("paddlex")
    try:
        from vie_plugin_panel_label.config import PanelLabelConfig
        from scripts.data.convert_ocr_dataset_to_ppocr import build_rec_pipeline
    except Exception as e:
        pytest.skip(f"cannot import deps: {e}")

    cfg = PanelLabelConfig()
    orient_model_path = Path(cfg.orient_model_path)
    if not orient_model_path.is_dir() or not (
        orient_model_path / "inference.yml"
    ).is_file():
        pytest.skip(
            "PaddleOCR orientation model bundle is unavailable at "
            f"{orient_model_path}"
        )
    return build_rec_pipeline(cfg)


def _make_synthetic_text_image(text: str = "ABC123") -> bytes:
    """造一张 cv2.imdecode 能读的 JPG（白底黑字）字节。"""
    import numpy as np
    img = np.full((100, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


@pytest.fixture
def rec_mini_dataset(tmp_path: Path) -> Path:
    """rec 测试专用 mini 数据集：图片是真实可解码的 JPG。"""
    root = tmp_path / "rec_data"
    for station in ("J46", "T1"):
        d = root / "side" / station / "crop_ocr"
        (d / "images").mkdir(parents=True)
        for i in range(3):
            (d / "images" / f"IMG_{station}_{i}.jpg").write_bytes(_make_synthetic_text_image())
            _write_labelme(
                d / "jsons" / f"IMG_{station}_{i}.json",
                f"IMG_{station}_{i}.jpg",
                [_make_shape(f"{station}-{i}")],
            )
    return root


def test_rec_crop_pixel_equality(rec_pipeline, rec_mini_dataset: Path, tmp_path: Path) -> None:
    """核心：保存到磁盘的 PNG 读回后必须与 pipeline 内存里的 rotated_crop 像素相同。"""
    from scripts.data.convert_ocr_dataset_to_ppocr import process_rec_sample, write_rec_split, rec_filename
    import numpy as np

    samples = find_samples(rec_mini_dataset)
    assert len(samples) == 6

    rec_dir = tmp_path / "rec_out"
    stats, _ = write_rec_split(samples, rec_pipeline, rec_dir, "train.txt")
    # 至少有 1 张通过（pipeline 偶尔可能 det-zero，所以不要求全部）
    if stats["kept"] == 0:
        pytest.skip(f"no rec crops produced from synthetic images, stats={stats}")

    checked = 0
    for s in samples:
        in_mem = process_rec_sample(s, rec_pipeline)
        if isinstance(in_mem, dict):
            continue
        # 默认单比例（expand_ratios=(0.15,)）→ 单变体、文件名无标签
        rotated, _text, _ratio = in_mem[0]
        png_path = rec_dir / "images" / rec_filename(s.original_stem, s.station_code)
        if not png_path.exists():
            continue
        reloaded = cv2.imread(str(png_path))
        assert reloaded is not None
        assert reloaded.shape == rotated.shape, f"shape mismatch on {png_path}"
        assert np.array_equal(reloaded, rotated), f"pixel mismatch on {png_path}"
        checked += 1
    assert checked > 0, "no PNG was actually checked"


def test_rec_multi_ratio_augmentation(rec_pipeline, rec_mini_dataset: Path, tmp_path: Path) -> None:
    """多外扩比例：每条 strip 应按每个比例各产出一份带标签的 PNG。"""
    from dataclasses import replace
    from scripts.data.convert_ocr_dataset_to_ppocr import write_rec_split, rec_filename

    pipeline = replace(rec_pipeline, expand_ratios=(0.0, 0.15, 0.3))
    samples = find_samples(rec_mini_dataset)
    assert len(samples) == 6

    rec_dir = tmp_path / "rec_aug"
    stats, texts = write_rec_split(samples, pipeline, rec_dir, "train.txt")
    if stats["kept"] == 0:
        pytest.skip(f"no rec crops produced, stats={stats}")

    # 通过的 strip 每条出 3 份；产出数应为 3 的倍数
    assert stats["kept"] % 3 == 0
    assert len(texts) == stats["kept"]

    # 找一条产出齐全的 strip，验证三种标签文件名都在
    for s in samples:
        names = [rec_filename(s.original_stem, s.station_code, tag) for tag in ("e00", "e15", "e30")]
        if all((rec_dir / "images" / n).exists() for n in names):
            break
    else:
        pytest.fail("no strip produced all three e00/e15/e30 variants")


def test_rec_skip_empty_description(rec_pipeline, tmp_path: Path) -> None:
    """description=空 的样本 rec 必须跳过（即使图片可推理）。"""
    from scripts.data.convert_ocr_dataset_to_ppocr import process_rec_sample

    root = tmp_path / "rec_skip"
    p = root / "side" / "X1" / "crop_ocr"
    (p / "images").mkdir(parents=True)
    (p / "images" / "IMG_E.jpg").write_bytes(_make_synthetic_text_image())
    _write_labelme(p / "jsons" / "IMG_E.json", "IMG_E.jpg", [_make_shape("")])

    samples = find_samples(root)
    result = process_rec_sample(samples[0], rec_pipeline)
    assert isinstance(result, dict)
    assert result["skip_reason"] == "empty-desc"


# ---------------------------------------------------------------------------
# resplit-only 模式单测
# ---------------------------------------------------------------------------

from scripts.data.convert_ocr_dataset_to_ppocr import resplit_det_split, resplit_rec_split
from scripts.data.convert_ocr_dataset_to_ppocr import (
    _get_mini_boxes,
    _crop_by_quad,
    _expand_box,
)


# ---------------------------------------------------------------------------
# 像素级推理管线一致性测试：验证 convert tool 产出与 PaddleX 推理完全一致
# ---------------------------------------------------------------------------


def _make_labelme_json_for_polygon(
    json_dir: Path, image_name: str, polygon: list, description: str = "TEST"
) -> Path:
    """在指定目录下创建 LabelMe JSON，points 为传入的 polygon。"""
    json_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_dir / f"{Path(image_name).stem}.json"
    _write_labelme(
        json_path,
        image_name,
        [
            {
                "label": "text",
                "score": 1.0,
                "points": polygon,
                "group_id": 0,
                "description": description,
                "difficult": False,
                "shape_type": "polygon",
                "flags": None,
                "attributes": {},
                "kie_linking": [],
            }
        ],
    )
    return json_path


def _make_rec_dataset_for_polygon(
    root: Path, station: str, stem: str, image_bytes: bytes, polygon: list, description: str = "TEST"
) -> Path:
    """创建完整的 crop_ocr 目录结构（images + jsons），返回 root。"""
    d = root / "side" / station / "crop_ocr"
    (d / "images").mkdir(parents=True)
    img_path = d / "images" / f"{stem}.jpg"
    img_path.write_bytes(image_bytes)
    _make_labelme_json_for_polygon(d / "jsons", f"{stem}.jpg", polygon, description)
    return root


def test_rec_crop_matches_ppocr_api_with_mock_detection(rec_pipeline, tmp_path: Path) -> None:
    """像素级验证：convert tool rec crop 与 PaddleX CropByPolys 严格一致。

    路线 B：几何直接来自人工标注框，不做 DB unclip 外扩——人工框已贴合文字，
    对应推理里 text_det 输出 dt_polys 的语义。convert tool 流程为
    minAreaRect 规整 4 点 → _crop_by_quad + orient。

    测试验证：对同一标注框，convert tool 的 _crop_by_quad 与推理用的 PaddleX
    CropByPolys（喂入同一 minAreaRect box）逐像素一致，确保裁剪算子无偏差。

    Path A: polygon → minAreaRect box → CropByPolys(PaddleX API) → orient
    Path B: polygon → process_rec_sample(convert tool 管线)

    如果任一像素不同，说明 convert tool 的裁剪算子与推理存在偏差。
    """
    import numpy as np
    from paddlex.inference.pipelines.components import CropByPolys
    from scripts.data.convert_ocr_dataset_to_ppocr import process_rec_sample

    # ---- 1. 构造合成图片和多边形标注（模拟手标轮廓） ----
    polygon = [[30.0, 50.0], [370.0, 40.0], [380.0, 140.0], [20.0, 150.0]]

    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "TEST123", (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 4)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    image_bytes = buf.tobytes()

    # ---- 2. minAreaRect 规整 4 点（与 process_rec_sample 一致，不 unclip） ----
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    pts = np.array([[int(round(float(p[0]))), int(round(float(p[1])))] for p in polygon], dtype=np.int32)
    box, sside = _get_mini_boxes(pts.reshape(-1, 1, 2))
    assert sside >= 3
    box = np.array(box, dtype=np.float32)
    # 与 process_rec_sample 一致地外扩，再喂给推理裁剪算子做等价对比
    box = _expand_box(box, rec_pipeline.expand_ratios[0])

    # ---- 3. Path A (推理裁剪算子): box → CropByPolys(PaddleX) → orient ----
    crop_by_polys = CropByPolys(det_box_type="quad")
    crops_infer = list(crop_by_polys(image, [box]))
    assert len(crops_infer) >= 1
    crop_infer = crops_infer[0]

    orient_results = rec_pipeline.text_orient.predict([crop_infer])
    angle = int(orient_results[0]["class_ids"][0])
    crop_infer = cv2.rotate(crop_infer, cv2.ROTATE_180) if angle == 1 else crop_infer

    # ---- 4. Path B (转换工具): process_rec_sample 管线 ----
    root = _make_rec_dataset_for_polygon(tmp_path, "J46", "IMG_NOUNCLIP", image_bytes, polygon)
    samples = find_samples(root)
    assert len(samples) == 1

    result_b = process_rec_sample(samples[0], rec_pipeline)
    if isinstance(result_b, dict):
        pytest.skip(f"convert tool skipped: {result_b['skip_reason']}")
    crop_convert, text_convert, _ratio = result_b[0]

    # ---- 5. 像素级对比 ----
    assert crop_infer.shape == crop_convert.shape, (
        f"shape mismatch: paddlex_api={crop_infer.shape} vs convert_tool={crop_convert.shape}"
    )
    if not np.array_equal(crop_infer, crop_convert):
        diff = np.abs(crop_infer.astype(np.int32) - crop_convert.astype(np.int32))
        max_diff = int(diff.max())
        nonzero = int(np.count_nonzero(diff))
        diff_dir = tmp_path / "diff_debug"
        diff_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(diff_dir / "crop_infer_paddlex_api.png"), crop_infer)
        cv2.imwrite(str(diff_dir / "crop_convert_tool.png"), crop_convert)
        cv2.imwrite(str(diff_dir / "diff.png"), diff.astype(np.uint8))
        pytest.fail(
            f"pixel mismatch! max_diff={max_diff}, nonzero_pixels={nonzero} "
            f"(saved debug images to {diff_dir})"
        )

    assert text_convert == "TEST", f"text mismatch: {text_convert}"


def test_resplit_det_only_rewrites_labels(mini_dataset: Path, tmp_path: Path) -> None:
    """Resplit det 应只重写 txt，不重新拷贝；缺图的样本计 missing。"""
    samples = find_samples(mini_dataset)
    train, val = split_samples(samples, val_ratio=0.34, seed=42)
    det_dir = tmp_path / "out" / "det"

    # 第一次 split (val_ratio=0.34) 生成产物
    write_det_split(train, det_dir, "train.txt")
    write_det_split(val, det_dir, "val.txt")
    assert (det_dir / "train.txt").exists()

    # 改用新 val_ratio (0.0) 重新切分后只 resplit
    train2, val2 = split_samples(samples, val_ratio=0.0, seed=42)
    t_stats = resplit_det_split(train2, det_dir, "train.txt")
    v_stats = resplit_det_split(val2, det_dir, "val.txt")

    # 所有图都在 train (val_ratio=0)
    assert t_stats["kept"] == 3
    assert t_stats["missing"] == 0
    assert v_stats["kept"] == 0
    # val.txt 应该被清空（empty file 因为 val=[]）
    assert (det_dir / "val.txt").read_text(encoding="utf-8") == ""
    # train.txt 三行
    assert len((det_dir / "train.txt").read_text(encoding="utf-8").splitlines()) == 3


def test_resplit_det_counts_missing_when_image_absent(mini_dataset: Path, tmp_path: Path) -> None:
    """图不存在时不写入 txt，且计入 missing。"""
    samples = find_samples(mini_dataset)
    det_dir = tmp_path / "out" / "det"
    write_det_split(samples, det_dir, "train.txt")
    # 删一张图
    images = list((det_dir / "images").iterdir())
    images[0].unlink()

    stats = resplit_det_split(samples, det_dir, "train.txt")
    assert stats["kept"] == 2
    assert stats["missing"] == 1


def test_resplit_rec_reuses_existing_pngs(mini_dataset: Path, tmp_path: Path) -> None:
    """Resplit rec 应只引用已存在的 PNG；text 直接取自 LabelMe description。"""
    import cv2 as _cv2
    import numpy as _np

    rec_dir = tmp_path / "out" / "rec"
    images_out = rec_dir / "images"
    images_out.mkdir(parents=True)

    samples = find_samples(mini_dataset)
    # 手工放 1 个 PNG（模拟之前 rec 跑过的产物）；其它样本无图
    img = _np.full((30, 80, 3), 255, dtype=_np.uint8)
    target = samples[0]
    placed_name = rec_filename(target.original_stem, target.station_code)
    _cv2.imwrite(str(images_out / placed_name), img)

    stats, texts = resplit_rec_split(samples, rec_dir, "train.txt")
    assert stats["kept"] == 1
    assert stats["missing"] == len(samples) - 1
    line = (rec_dir / "train.txt").read_text(encoding="utf-8").splitlines()[0]
    path_part, text_part = line.split("\t")
    assert path_part == f"images/{placed_name}"
    # text 来自 LabelMe description
    assert text_part in {"J46-A/PW-1", "J46-B/PW-2", "T1-X/KM-1"}
    assert texts == [text_part]
