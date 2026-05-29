'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 09:02:18
@FilePath     : panel_label_detect.py
@Description  : 面板标签检测
'''

from services.yolo import YoloOnnxInfer
from services.utils import *
import numpy as np
from schemas.data_base import DetectResult
from paddleocr import TextDetection, TextLineOrientationClassification, TextRecognition
from paddlex.inference.pipelines.components import CropByPolys
from .utils import Points_to_Mask
from typing import List
from dataclasses import dataclass, field

import time
from utils import vision_logger
from .product_type import PRODUCT_guideline


@dataclass
class PanellabelItem:
    Points: List[np.ndarray] = field(default_factory=list)
    index: List[int] = field(default_factory=list)
    class_id: List[int] = field(default_factory=list)
    texts: List[str] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)

    def save_img(self, image, save_path):
        for i, point in enumerate(self.Points):
            image = cv2.polylines(image, [point], True, (0, 255, 0), 2)
            image = cv2.putText(
                image, self.texts[i], (point[0][0], point[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2
            )
        cv2.imwrite(save_path, image)


class PanelLabelDetect(YoloOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="seg"):
        super().__init__(model_path, 2, confThreshold, nmsThreshold, task)
        self.id2name = {
            0: "line",
            1: "QFU",
        }


class OCRPipeline:
    def __init__(
        self,
        detect_model_path,
        orient_model_path,
        text_recognition_model_path,
        confThreshold=0.5,
        nmsThreshold=0.5,
        text_rec_score_thresh=0.7,
        text_rec_input_shape=None,
        text_det_limit_side_len=128,
        text_det_limit_type="min",
        text_det_thresh=0.3,
        text_det_box_thresh=0.4,
        text_det_unclip_ratio=2.0,
        text_det_input_shape=None,
    ):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")

        # Stage 1: Text Detection
        self.text_det_model = TextDetection(
            model_name="PP-OCRv5_server_det",
            limit_side_len=text_det_limit_side_len,
            limit_type=text_det_limit_type,
            thresh=text_det_thresh,
            box_thresh=text_det_box_thresh,
            unclip_ratio=text_det_unclip_ratio,
            input_shape=text_det_input_shape,
        )

        # Stage 2: Text Line Orientation
        self.text_orient_model = TextLineOrientationClassification(
            model_name="PP-LCNet_x1_0_textline_ori",
            model_dir=orient_model_path,
        )

        # Stage 3: Text Recognition
        self.text_rec_model = TextRecognition(
            model_name="PP-OCRv5_server_rec",
            model_dir=text_recognition_model_path,
            input_shape=text_rec_input_shape,
        )

        self.text_rec_score_thresh = text_rec_score_thresh
        self._crop_by_polys = CropByPolys(det_box_type="quad")

    def infer(self, image) -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        points_line = mask_polygons[class_ids == 0] if 0 in class_ids else []
        start = time.time()
        mask_rois, sorted_idxs = Points_to_Mask(image, points_line)
        end = time.time()
        vision_logger.debug(f"Points_to_Mask: {end - start:.4f}秒")
        start = time.time()

        # Stage 1: Text Detection on each mask_roi
        # 每张 roi 只有一个文本行，检测出多个则为误检测，只保留面积最大的
        all_dt_polys = []
        roi_to_idx = []
        for i, roi in enumerate(mask_rois):
            det_result = self.text_det_model.predict(roi)
            dt_polys = det_result[0]["dt_polys"]
            if len(dt_polys) == 0:
                continue
            areas = [cv2.contourArea(np.array(poly, dtype=np.float32).reshape(-1, 2)) for poly in dt_polys]
            best_poly = dt_polys[int(np.argmax(areas))]
            all_dt_polys.append([best_poly])
            roi_to_idx.append(i)
        det_end = time.time()
        vision_logger.debug(f"Text Detection: {det_end - start:.4f}秒")

        # Crop detected text regions
        all_crops = []
        crop_roi_map = []
        for i, dt_polys in enumerate(all_dt_polys):
            roi_idx = roi_to_idx[i]
            roi = mask_rois[roi_idx]
            crops = list(self._crop_by_polys(roi, dt_polys))
            if crops:
                all_crops.append(crops[0])
                crop_roi_map.append(roi_idx)

        # Stage 2: Text Line Orientation
        text_map: dict = {}
        if all_crops:
            orient_results = self.text_orient_model.predict(all_crops)
            angles = [int(r["class_ids"][0]) for r in orient_results]
            orient_end = time.time()
            vision_logger.debug(f"Text Orientation: {orient_end - det_end:.4f}秒")

            # Stage 3: Rotate + Text Recognition
            rotated_crops = [
                cv2.rotate(crop, cv2.ROTATE_180) if angle == 1 else crop for crop, angle in zip(all_crops, angles)
            ]
            rec_results = self.text_rec_model.predict(rotated_crops)
            rec_end = time.time()
            vision_logger.debug(f"Text Recognition: {rec_end - orient_end:.4f}秒")

            for crop_idx, rec_res in enumerate(rec_results):
                roi_idx = crop_roi_map[crop_idx]
                rec_text = rec_res["rec_text"]
                rec_score = rec_res["rec_score"]
                if isinstance(rec_text, list):
                    rec_text = rec_text[0] if rec_text else ""
                if rec_text and rec_text.strip() and rec_score >= self.text_rec_score_thresh:
                    text_map[roi_idx] = rec_text

        # 所有 YOLO 检测到的线标均进入结果，OCR 未识别的给 None
        all_rois = list(range(len(mask_rois)))
        texts = [text_map.get(i) for i in all_rois]

        end = time.time()
        vision_logger.debug(f"OCR 三阶段总耗时: {end - start:.4f}秒")
        line_indices = np.where(class_ids == 0)[0]
        ori_index = [line_indices[sorted_idxs[i]] for i in all_rois]
        positions = [
            np.int64(cv2.boxPoints(cv2.minAreaRect(np.array(mask_polygons[idx], dtype=np.float32)))).flatten().tolist()
            for idx in ori_index
        ]
        roi_classes_ids = class_ids[ori_index]
        confidences = [results.scores[idx] for idx in ori_index]
        panel_label_item = PanellabelItem(
            Points=positions,
            index=ori_index,
            class_id=roi_classes_ids.tolist(),
            texts=texts,
            confidence=confidences,
        )

        return panel_label_item


class OCRPipelineCrop:
    def __init__(self, detect_model_path, orient_model_path, confThreshold=0.5, nmsThreshold=0.5):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")

    def infer(self, image, sort_by="xy") -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        points_line = mask_polygons[class_ids == 0]
        start = time.time()
        mask_rois, sorted_idxs = Points_to_Mask(image, points_line, sort_by=sort_by)
        end = time.time()
        vision_logger.debug(f"Points_to_Mask: {end - start:.4f}秒")
        return mask_rois
