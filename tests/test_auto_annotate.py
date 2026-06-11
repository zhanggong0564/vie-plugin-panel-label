"""auto_annotate 单元测试"""
import json
import sys
import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# 将项目根目录加入 path（与其他测试保持一致）
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class TestBuildLabelmeJson:
    """_build_labelme_json 纯函数测试"""

    def test_basic_structure(self):
        """返回的 dict 必须包含 LabelMe 所有顶层字段"""
        from scripts.auto_annotate import _build_labelme_json

        result = _build_labelme_json(
            shapes=[],
            image_filename="img.jpg",
            image_height=100,
            image_width=200,
        )
        assert result["version"] == "3.3.9"
        assert result["flags"] == {}
        assert result["shapes"] == []
        assert result["imagePath"] == "img.jpg"
        assert result["imageData"] is None
        assert result["imageHeight"] == 100
        assert result["imageWidth"] == 200
        assert result["description"] == ""

    def test_shape_fields(self):
        """shapes 列表中每个元素必须包含所有 LabelMe shape 字段"""
        from scripts.auto_annotate import _build_labelme_json

        shapes = [
            {
                "label": "text",
                "score": 0.95,
                "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]],
                "description": "PE1-J5",
            }
        ]
        result = _build_labelme_json(
            shapes=shapes,
            image_filename="img.jpg",
            image_height=100,
            image_width=200,
        )
        shape = result["shapes"][0]
        assert shape["label"] == "text"
        assert shape["score"] == 0.95
        assert shape["points"] == [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]
        assert shape["group_id"] == 0
        assert shape["description"] == "PE1-J5"
        assert shape["difficult"] is False
        assert shape["shape_type"] == "polygon"
        assert shape["flags"] is None
        assert shape["attributes"] == {}
        assert shape["kie_linking"] == []

    def test_empty_description_when_ocr_failed(self):
        """OCR 未识别时 description 应为空字符串"""
        from scripts.auto_annotate import _build_labelme_json

        shapes = [
            {
                "label": "text",
                "score": 0.0,
                "points": [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]],
                "description": "",
            }
        ]
        result = _build_labelme_json(shapes=shapes, image_filename="x.jpg", image_height=50, image_width=100)
        assert result["shapes"][0]["description"] == ""


class TestAutoAnnotatorInit:
    """AutoAnnotator 初始化测试（mock PaddleOCR）"""

    def test_init_loads_three_models(self):
        """初始化时应加载 TextDetection、TextLineOrientationClassification、TextRecognition"""
        with (
            patch("scripts.auto_annotate.TextDetection") as mock_det,
            patch("scripts.auto_annotate.TextLineOrientationClassification") as mock_ori,
            patch("scripts.auto_annotate.TextRecognition") as mock_rec,
            patch("scripts.auto_annotate.CropByPolys") as mock_crop,
        ):
            from scripts.auto_annotate import AutoAnnotator
            ann = AutoAnnotator(
                orient_model_path="fake/orient",
                rec_model_path="fake/rec",
                score_thresh=0.8,
            )
            mock_det.assert_called_once()
            mock_ori.assert_called_once()
            mock_rec.assert_called_once()
            mock_crop.assert_called_once_with(det_box_type="quad")
            assert ann.score_thresh == 0.8


class TestAutoAnnotatorInferImage:
    """infer_image 测试：mock 全部 PaddleOCR 调用"""

    @pytest.fixture
    def annotator(self):
        """构造 AutoAnnotator，绕过真实模型加载"""
        with (
            patch("scripts.auto_annotate.TextDetection"),
            patch("scripts.auto_annotate.TextLineOrientationClassification"),
            patch("scripts.auto_annotate.TextRecognition"),
            patch("scripts.auto_annotate.CropByPolys"),
        ):
            from scripts.auto_annotate import AutoAnnotator
            ann = AutoAnnotator(orient_model_path="x", rec_model_path="y", score_thresh=0.7)
        return ann

    def test_no_text_detected_returns_empty_shapes(self, annotator):
        """文本检测无结果时，返回 shapes 为空列表的 LabelMe JSON"""
        # TextDetection 返回空 dt_polys
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": []}])

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert result["shapes"] == []
        assert result["imagePath"] == "img.jpg"
        assert result["imageHeight"] == 100
        assert result["imageWidth"] == 200

    def test_one_text_recognized(self, annotator):
        """检测到1个文本框且 OCR 分数达标时，shapes 有1条记录"""
        dt_poly = [[10, 20], [50, 20], [50, 40], [10, 40]]
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": [dt_poly]}])

        crop_img = np.zeros((20, 40, 3), dtype=np.uint8)
        annotator._crop = MagicMock(return_value=iter([crop_img]))

        annotator.text_ori.predict = MagicMock(return_value=[{"class_ids": [0]}])
        annotator.text_rec.predict = MagicMock(
            return_value=[{"rec_text": "PE1-J5", "rec_score": 0.95}]
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert len(result["shapes"]) == 1
        shape = result["shapes"][0]
        assert shape["description"] == "PE1-J5"
        assert shape["score"] == pytest.approx(0.95)
        assert shape["points"] == [[10.0, 20.0], [50.0, 20.0], [50.0, 40.0], [10.0, 40.0]]

    def test_multiple_polys_keeps_largest_area(self, annotator):
        """检测出多个框时，只保留面积最大的一个"""
        # small_poly: 10×10 = 100；large_poly: 100×50 = 5000
        small_poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        large_poly = [[0, 0], [100, 0], [100, 50], [0, 50]]
        annotator.text_det.predict = MagicMock(
            return_value=[{"dt_polys": [small_poly, large_poly]}]
        )

        crop_img = np.zeros((50, 100, 3), dtype=np.uint8)
        annotator._crop = MagicMock(return_value=iter([crop_img]))

        annotator.text_ori.predict = MagicMock(return_value=[{"class_ids": [0]}])
        annotator.text_rec.predict = MagicMock(
            return_value=[{"rec_text": "PE1-J5", "rec_score": 0.95}]
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        # 只应输出1条 shape，且对应面积最大的 large_poly
        assert len(result["shapes"]) == 1
        assert result["shapes"][0]["points"] == [
            [0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]
        ]

    def test_low_score_gives_empty_description(self, annotator):
        """OCR 识别分数低于阈值时，description 为空字符串，框仍保留"""
        dt_poly = [[10, 20], [50, 20], [50, 40], [10, 40]]
        annotator.text_det.predict = MagicMock(return_value=[{"dt_polys": [dt_poly]}])

        crop_img = np.zeros((20, 40, 3), dtype=np.uint8)
        annotator._crop = MagicMock(return_value=iter([crop_img]))

        annotator.text_ori.predict = MagicMock(return_value=[{"class_ids": [0]}])
        annotator.text_rec.predict = MagicMock(
            return_value=[{"rec_text": "BLURRY", "rec_score": 0.3}]  # 低于 0.7
        )

        image = np.zeros((100, 200, 3), dtype=np.uint8)
        result = annotator.infer_image(image, "img.jpg")

        assert len(result["shapes"]) == 1
        assert result["shapes"][0]["description"] == ""
        assert result["shapes"][0]["score"] == pytest.approx(0.3)


class TestAutoAnnotatorProcessDir:
    """process_dir 测试：使用 tmp_path + mock infer_image"""

    @pytest.fixture
    def annotator_no_models(self):
        """绕过模型加载，构造 AutoAnnotator"""
        with (
            patch("scripts.auto_annotate.TextDetection"),
            patch("scripts.auto_annotate.TextLineOrientationClassification"),
            patch("scripts.auto_annotate.TextRecognition"),
            patch("scripts.auto_annotate.CropByPolys"),
        ):
            from scripts.auto_annotate import AutoAnnotator
            return AutoAnnotator(orient_model_path="x", rec_model_path="y")

    def _make_fake_image(self, path: Path):
        """创建1×1像素的真实 JPEG 图片（cv2 可读）"""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        cv2.imwrite(str(path), img)

    def test_creates_jsons_dir(self, annotator_no_models, tmp_path):
        """处理后应在 images/ 的同层创建 jsons/ 目录"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        self._make_fake_image(images_dir / "img.jpg")

        fake_json = {"version": "3.3.9", "flags": {}, "shapes": [], "imagePath": "img.jpg",
                     "imageData": None, "imageHeight": 1, "imageWidth": 1, "description": ""}
        annotator_no_models.infer_image = MagicMock(return_value=fake_json)

        annotator_no_models.process_dir(images_dir)

        jsons_dir = tmp_path / "jsons"
        assert jsons_dir.is_dir()
        assert (jsons_dir / "img.json").exists()

    def test_skips_existing_json_by_default(self, annotator_no_models, tmp_path):
        """overwrite=False 时若 JSON 已存在应跳过（infer_image 不调用）"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        self._make_fake_image(images_dir / "img.jpg")

        jsons_dir = tmp_path / "jsons"
        jsons_dir.mkdir()
        (jsons_dir / "img.json").write_text("{}")

        annotator_no_models.infer_image = MagicMock()
        annotator_no_models.process_dir(images_dir, overwrite=False)

        annotator_no_models.infer_image.assert_not_called()

    def test_overwrites_when_flag_set(self, annotator_no_models, tmp_path):
        """overwrite=True 时应覆盖已存在的 JSON"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        self._make_fake_image(images_dir / "img.jpg")

        jsons_dir = tmp_path / "jsons"
        jsons_dir.mkdir()
        (jsons_dir / "img.json").write_text("{}")

        fake_json = {"version": "3.3.9", "flags": {}, "shapes": [], "imagePath": "img.jpg",
                     "imageData": None, "imageHeight": 1, "imageWidth": 1, "description": ""}
        annotator_no_models.infer_image = MagicMock(return_value=fake_json)

        annotator_no_models.process_dir(images_dir, overwrite=True)

        annotator_no_models.infer_image.assert_called_once()

    def test_skips_unreadable_image(self, annotator_no_models, tmp_path):
        """cv2 无法读取的文件应跳过，不抛异常"""
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        (images_dir / "broken.jpg").write_text("not an image")

        annotator_no_models.infer_image = MagicMock()
        # 应正常完成，不抛出异常
        annotator_no_models.process_dir(images_dir)
        annotator_no_models.infer_image.assert_not_called()
