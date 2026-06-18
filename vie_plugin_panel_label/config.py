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
    text_det_model_path = "./weights/panel_label/text_det_plane_ppocrv5m_v1"
    orient_model_path = "./weights/panel_label/v2/textline_ori_lcnet_v2"
    text_recognition_model_path = "./weights/panel_label/v2/PP-OCRv5_server_rec_merged_v6_diff_lr"
    confThreshold = 0.7
    nmsThreshold = 0.8
    # TextDetection

    text_det_limit_side_len = 1248
    text_det_limit_type = "max"
    text_det_thresh = 0.3
    text_det_box_thresh = 0.3
    text_det_unclip_ratio = 2
    text_det_input_shape = [3, 160, 1248]
    # TextRecognition
    text_rec_score_thresh = 0.7
    text_rec_input_shape = None

    def __init__(self):
        # guideline 引导框 ROI 过滤开关：默认关闭，部署时设环境变量
        # PANEL_LABEL_GUIDELINE_FILTER=true 即可开启，无需改代码。
        self.enable_guideline_filter = _env_flag("PANEL_LABEL_GUIDELINE_FILTER", True)
        # 检测实例去重阈值（旋转框交集/较小框面积）：同一线标的重复框（全长框+半截框）
        # 轴对齐 NMS 抑制不掉，超过该重叠度只保留高置信度者；设 >=1 关闭去重。
        self.dedup_overlap_thresh = _env_float("PANEL_LABEL_DEDUP_OVERLAP", 0.6)
