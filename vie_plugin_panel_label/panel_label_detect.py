'''
@Author       : gongzhang4
@Date         : 2026-02-26 09:20:56
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-06 09:02:18
@FilePath     : panel_label_detect.py
@Description  : 面板标签检测
'''

import time
from pathlib import Path

import cv2
import numpy as np

from services.rfdetr import RFDetrOnnxInfer
from utils import vision_logger

from .models import PanellabelItem
from .ocr_models import PanelLabelOrientationClassifier, PanelLabelTextRecognizer
from .utils import Points_to_Mask, dedup_overlapping_polygons


class PanelLabelDetect(RFDetrOnnxInfer):
    def __init__(self, model_path, confThreshold=0.5, nmsThreshold=0.5, task="seg"):
        super().__init__(model_path, 2, confThreshold, task)
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
        text_orient_score_thresh=0.7,
        text_rec_input_shape=None,
        dedup_overlap_thresh=0.6,
    ):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")
        # 同类实例旋转框 IoS 去重阈值（>=1 关闭），抑制同一线标的重复检测框
        self.dedup_overlap_thresh = dedup_overlap_thresh

        orient_metadata_path = Path(orient_model_path).with_suffix("") / "inference.yml"
        self.text_orient_model = PanelLabelOrientationClassifier(
            orient_model_path,
            str(orient_metadata_path),
        )

        recognition_metadata_path = (
            Path(text_recognition_model_path).with_suffix("") / "inference.yml"
        )
        self.text_rec_model = PanelLabelTextRecognizer(
            text_recognition_model_path,
            str(recognition_metadata_path),
            input_shape=text_rec_input_shape,
        )

        self.text_rec_score_thresh = text_rec_score_thresh
        self.text_orient_score_thresh = text_orient_score_thresh

    def _orient_crops(self, crops):
        orient_results = list(self.text_orient_model.predict(crops))
        if len(orient_results) != len(crops):
            raise ValueError(
                f"orientation result count {len(orient_results)} does not match "
                f"crop count {len(crops)}"
            )
        rotated = []
        uncertain = []
        for index, (crop_image, result) in enumerate(zip(crops, orient_results)):
            angle = int(result["class_ids"][0])
            score = float(result["scores"][0])
            rotated.append(
                cv2.rotate(crop_image, cv2.ROTATE_180)
                if angle == 1
                else crop_image
            )
            if score < self.text_orient_score_thresh:
                uncertain.append(index)
        return rotated, uncertain

    def _recognize_with_fallback(self, rotated_crops, uncertain_indices):
        final_crops = list(rotated_crops)
        results = list(self.text_rec_model.predict(final_crops))
        if len(results) != len(final_crops):
            raise ValueError(
                f"recognition result count {len(results)} does not match "
                f"crop count {len(final_crops)}"
            )
        if not uncertain_indices:
            return final_crops, results
        flipped = [
            cv2.rotate(final_crops[index], cv2.ROTATE_180)
            for index in uncertain_indices
        ]
        flipped_results = list(self.text_rec_model.predict(flipped))
        if len(flipped_results) != len(flipped):
            raise ValueError(
                f"fallback recognition result count {len(flipped_results)} "
                f"does not match crop count {len(flipped)}"
            )
        for position, index in enumerate(uncertain_indices):
            if float(flipped_results[position]["rec_score"]) > float(
                results[index]["rec_score"]
            ):
                final_crops[index] = flipped[position]
                results[index] = flipped_results[position]
        return final_crops, results

    def _extract_texts(self, rec_results):
        texts = []
        for result in rec_results:
            text = result["rec_text"]
            if isinstance(text, list):
                text = text[0] if text else ""
            score = float(result["rec_score"])
            texts.append(
                text
                if text and text.strip() and score >= self.text_rec_score_thresh
                else None
            )
        return texts

    def infer(self, image) -> PanellabelItem:
        results = self.detect_model.infer(image)
        class_ids = np.array(results.class_ids)
        scores = np.array(results.scores)
        mask_polygons = np.array(results.mask_polygons, dtype=object)
        # 二次去重：同一线标的重复检测框（全长框+半截框）轴对齐 NMS 抑制不掉，
        # 按 mask 旋转框 IoS 去重，避免 observed 数多于标准数误判 extra。
        if len(class_ids) > 1:
            keep = dedup_overlapping_polygons(mask_polygons, scores, class_ids, self.dedup_overlap_thresh)
            if len(keep) < len(class_ids):
                vision_logger.info(f"检测实例去重: {len(class_ids)} -> {len(keep)}")
                class_ids = class_ids[keep]
                scores = scores[keep]
                mask_polygons = mask_polygons[keep]
        points_line = mask_polygons[class_ids == 0] if 0 in class_ids else []
        start = time.time()
        mask_rois, sorted_idxs, _ = Points_to_Mask(image, points_line, return_maps=True)
        end = time.time()
        vision_logger.debug(f"Points_to_Mask: {end - start:.4f}秒")
        start = time.time()

        text_crops = []
        texts = []
        if mask_rois:
            rotated_crops, uncertain_indices = self._orient_crops(list(mask_rois))
            text_crops, rec_results = self._recognize_with_fallback(
                rotated_crops, uncertain_indices
            )
            texts = self._extract_texts(rec_results)

        end = time.time()
        vision_logger.debug(f"OCR 三阶段总耗时: {end - start:.4f}秒")
        line_indices = np.where(class_ids == 0)[0]
        roi_indices = range(len(mask_rois))
        ori_index = [line_indices[sorted_idxs[index]] for index in roi_indices]
        positions = [
            np.int64(cv2.boxPoints(cv2.minAreaRect(np.array(mask_polygons[idx], dtype=np.float32)))).flatten().tolist()
            for idx in ori_index
        ]
        roi_classes_ids = class_ids[ori_index]
        confidences = [scores[idx] for idx in ori_index]
        panel_label_item = PanellabelItem(
            Points=positions,
            index=ori_index,
            class_id=roi_classes_ids.tolist(),
            texts=texts,
            confidence=confidences,
            text_crops=text_crops,
        )

        return panel_label_item
