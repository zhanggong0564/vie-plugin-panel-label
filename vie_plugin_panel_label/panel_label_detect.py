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
from .utils import Points_to_Mask, dedup_overlapping_polygons, map_roi_points_to_original
from .models import PanellabelItem

import os
import yaml
import time
from utils import vision_logger


def _resolve_model_name(model_dir: str) -> str:
    """从导出推理目录的 inference.yml 解析 Global.model_name。

    paddleocr 高层封装不会自动解析，需显式传入与目录架构匹配的 model_name，
    否则会触发 'Model name mismatch' 断言。
    """
    cfg_path = os.path.join(model_dir, "inference.yml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["Global"]["model_name"]


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
        text_det_model_path=None,
        text_det_limit_side_len=128,
        text_det_limit_type="min",
        text_det_thresh=0.3,
        text_det_box_thresh=0.4,
        text_det_unclip_ratio=2.0,
        text_det_input_shape=None,
        dedup_overlap_thresh=0.6,
    ):
        self.detect_model = PanelLabelDetect(detect_model_path, confThreshold, nmsThreshold, task="seg")
        # 同类实例旋转框 IoS 去重阈值（>=1 关闭），抑制同一线标的重复检测框
        self.dedup_overlap_thresh = dedup_overlap_thresh

        # ===== 直送对比分支（feat/direct-ocr）=====
        # 跳过 DBNet 文本检测：mask2roi 展平后的 roi 本身即单行文本条，直接送 cls+rec。
        # 保留 __init__ 签名不变（det 相关参数不再使用），business_logic/run.py 无需改动。
        # 与 main（det 版）的差异仅此一处，切分支即对比。
        self.text_det_model = None

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
        mask_rois, sorted_idxs, roi_transforms = Points_to_Mask(image, points_line, return_maps=True)
        end = time.time()
        vision_logger.debug(f"Points_to_Mask: {end - start:.4f}秒")
        start = time.time()

        # Stage 1: 直送对比分支 —— 跳过文本检测，展平 roi 整条直接当待识别小图。
        # det 版在此用 DBNet 裁紧文字区；本分支不裁，text_det_points 全空（仅影响可视化蓝框）。
        all_crops = list(mask_rois)
        crop_roi_map = list(range(len(mask_rois)))
        text_det_map: dict = {}  # 直送无文本检测框，留空
        det_end = time.time()
        vision_logger.debug(f"Text Detection(direct, skipped): {det_end - start:.4f}秒")

        # Stage 2: Text Line Orientation
        text_map: dict = {}
        crop_map: dict = {}  # roi_idx -> rotated_crop（识别模型输入小图，供数据回流落盘）
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
                crop_map[roi_idx] = rotated_crops[crop_idx]
                rec_text = rec_res["rec_text"]
                rec_score = rec_res["rec_score"]
                if isinstance(rec_text, list):
                    rec_text = rec_text[0] if rec_text else ""
                if rec_text and rec_text.strip() and rec_score >= self.text_rec_score_thresh:
                    text_map[roi_idx] = rec_text

        # 所有 YOLO 检测到的线标均进入结果，OCR 未识别的给 None
        all_rois = list(range(len(mask_rois)))
        texts = [text_map.get(i) for i in all_rois]
        text_det_points = [text_det_map.get(i) for i in all_rois]
        text_crops = [crop_map.get(i) for i in all_rois]

        end = time.time()
        vision_logger.debug(f"OCR 三阶段总耗时: {end - start:.4f}秒")
        line_indices = np.where(class_ids == 0)[0]
        ori_index = [line_indices[sorted_idxs[i]] for i in all_rois]
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
            text_det_points=text_det_points,
            text_crops=text_crops,
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
