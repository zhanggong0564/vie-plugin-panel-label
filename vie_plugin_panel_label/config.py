'''
@Author       : gongzhang4
@Date         : 2026-03-02 08:01:03
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-22 09:50:46
@FilePath     : panel_label_config.py
@Description  :
'''


class PanelLabelConfig:
    model_path = "./weights/panel_label/label_det_yolo_v3.onnx"
    text_det_model_path = "./weights/panel_label/text_det_plane_ppocrv5m_v1"
    orient_model_path = "./weights/panel_label/textline_ori_lcnet_v4"
    text_recognition_model_path = "./weights/panel_label/text_rec_ppocrv5s_v4"
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
    text_rec_input_shape = [3, 48, 320]
