"""panel_label 插件配置默认值测试（原 test/test_config.py 中的对应断言迁移而来）。"""
from vie_plugin_panel_label.config import PanelLabelConfig


def test_panel_label_config_defaults():
    cfg = PanelLabelConfig()
    assert cfg.model_path == "./weights/panel_label/best_v3.onnx"
    assert cfg.confThreshold == 0.7
    assert cfg.text_det_model_path == "./weights/panel_label/PP-OCRv5_mobile_det_panel_v1"
