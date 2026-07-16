"""panel_label 插件配置默认值测试（原 test/test_config.py 中的对应断言迁移而来）。"""

from vie_plugin_panel_label.config import PanelLabelConfig
from vie_plugin_panel_label.panel_label_detect import PanelLabelDetect
from services.rfdetr import RFDetrInfer


def test_panel_label_config_defaults():
    cfg = PanelLabelConfig()
    assert cfg.model_path == "./weights/panel_label/v2/rfdetr-seg-nano.onnx"
    assert cfg.confThreshold == 0.6


def test_panel_label_detect_uses_rfdetr_base():
    assert issubclass(PanelLabelDetect, RFDetrInfer)


def test_config_points_to_onnx_models():
    cfg = PanelLabelConfig()
    assert cfg.orient_model_path.endswith("textline_ori_lcnet_v2.onnx")
    assert cfg.text_recognition_model_path.endswith(
        "PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
    )


def test_direct_ocr_config_has_no_text_detection_fields():
    cfg = PanelLabelConfig()
    prefix = "text" + "_det_"
    stale = {
        prefix + "model_path",
        prefix + "limit_side_len",
        prefix + "limit_type",
        prefix + "thresh",
        prefix + "box_thresh",
        prefix + "unclip_ratio",
        prefix + "input_shape",
    }
    assert stale.isdisjoint(vars(type(cfg)))


class TestGuidelineFilterSwitch:
    def test_default_enabled(self, monkeypatch):
        """当前部署默认开启 guideline 过滤，可用环境变量显式关闭。"""
        monkeypatch.delenv("PANEL_LABEL_GUIDELINE_FILTER", raising=False)
        assert PanelLabelConfig().enable_guideline_filter is True

    def test_env_disable(self, monkeypatch):
        for off in ("false", "0", "no", "off", "False", "OFF"):
            monkeypatch.setenv("PANEL_LABEL_GUIDELINE_FILTER", off)
            assert PanelLabelConfig().enable_guideline_filter is False

    def test_env_enable(self, monkeypatch):
        for on in ("true", "1", "yes", "on"):
            monkeypatch.setenv("PANEL_LABEL_GUIDELINE_FILTER", on)
            assert PanelLabelConfig().enable_guideline_filter is True


class TestDedupOverlapThresh:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("PANEL_LABEL_DEDUP_OVERLAP", raising=False)
        assert PanelLabelConfig().dedup_overlap_thresh == 0.6

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PANEL_LABEL_DEDUP_OVERLAP", "0.75")
        assert PanelLabelConfig().dedup_overlap_thresh == 0.75

    def test_env_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("PANEL_LABEL_DEDUP_OVERLAP", "abc")
        assert PanelLabelConfig().dedup_overlap_thresh == 0.6


class TestGuidelineOverlapThresh:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("PANEL_LABEL_GUIDELINE_OVERLAP", raising=False)
        assert PanelLabelConfig().guideline_overlap_thresh == 0.9

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("PANEL_LABEL_GUIDELINE_OVERLAP", "0.95")
        assert PanelLabelConfig().guideline_overlap_thresh == 0.95

    def test_env_invalid_falls_back(self, monkeypatch):
        monkeypatch.setenv("PANEL_LABEL_GUIDELINE_OVERLAP", "abc")
        assert PanelLabelConfig().guideline_overlap_thresh == 0.9
