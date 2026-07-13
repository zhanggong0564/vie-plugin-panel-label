'''
@Author       : gongzhang4
@Date         : 2026-03-02 08:01:03
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-22 09:50:46
@FilePath     : panel_label_config.py
@Description  :
'''

import os


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔环境变量：0/false/no/off（不分大小写）为关，未设置取默认值。"""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "false", "no", "off")


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量，未设置或解析失败时取默认值。"""
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


class PanelLabelConfig:
    model_path = "./weights/panel_label/v2/best.onnx"
    orient_model_path = "./weights/panel_label/v2/textline_ori_lcnet_v2.onnx"
    text_recognition_model_path = "./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
    confThreshold = 0.6
    nmsThreshold = 0.8
    # TextLineOrientation 文本行方向分类
    # 仅当 top-1 置信度 >= 该阈值才直接采信模型方向；低于该阈值则正反两个朝向
    # 都送识别，取 rec_score 高者（用识别置信度仲裁方向，比赌分类器稳）。
    # 设 0.5 即全信分类器(关闭仲裁)，设 1.0 即全部样本都双向识别。
    text_orient_score_thresh = 0.9
    # TextRecognition
    text_rec_score_thresh = 0.7
    text_rec_input_shape = None

    def __init__(self):
        # guideline 引导框 ROI 过滤开关：默认开启，默认 API 契约要求请求携带
        # guideline_coordinates；特殊兼容部署可设 PANEL_LABEL_GUIDELINE_FILTER=false 关闭。
        self.enable_guideline_filter = _env_flag("PANEL_LABEL_GUIDELINE_FILTER", True)
        # 检测实例去重阈值（旋转框交集/较小框面积）：同一线标的重复框（全长框+半截框）
        # 轴对齐 NMS 抑制不掉，超过该重叠度只保留高置信度者；设 >=1 关闭去重。
        self.dedup_overlap_thresh = _env_float("PANEL_LABEL_DEDUP_OVERLAP", 0.6)
        # guideline ROI 过滤阈值：检测框落在引导区域内的面积占比达到该值才保留。
        self.guideline_overlap_thresh = _env_float("PANEL_LABEL_GUIDELINE_OVERLAP", 0.9)
