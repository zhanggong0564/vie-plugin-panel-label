'''
@Author       : gongzhang4
@Date         : 2026-03-02 08:01:03
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-04-22 09:50:46
@FilePath     : panel_label_config.py
@Description  :
'''


class PanelLabelConfig:
    model_path = "./weights/panel_label/best_v3.onnx"
    orient_model_path = "./weights/panel_label/PP-LCNet_x1_0_textline_ori_v3"
    text_recognition_model_path = "./weights/panel_label/PP-OCRv5_server_rec_plane_infer_v3"
    confThreshold = 0.7
    nmsThreshold = 0.8
    # TextDetection
    text_det_limit_side_len = 480
    text_det_limit_type = "max"
    text_det_thresh = 0.3
    text_det_box_thresh = 0.3
    text_det_unclip_ratio = 2.0
    text_det_input_shape = [3, 128, 640]
    # TextRecognition
    text_rec_score_thresh = 0.7
    text_rec_input_shape = [3, 48, 320]
