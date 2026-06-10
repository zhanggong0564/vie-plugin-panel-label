"""panel_label 插件配置默认值测试（原 test/test_config.py 中的对应断言迁移而来）。"""
from vie_plugin_panel_label.config import PanelLabelConfig


def test_panel_label_config_defaults():
    cfg = PanelLabelConfig()
    assert cfg.model_path == "./weights/panel_label/label_det_yolo_v3.onnx"
    assert cfg.confThreshold == 0.7
    assert cfg.text_det_model_path == "./weights/panel_label/text_det_plane_ppocrv5m_v1"


class TestGuidelineFilterSwitch:
    def test_default_disabled(self, monkeypatch):
        """当前部署默认关闭 guideline 过滤，开启需显式设 PANEL_LABEL_GUIDELINE_FILTER=true"""
        monkeypatch.delenv("PANEL_LABEL_GUIDELINE_FILTER", raising=False)
        assert PanelLabelConfig().enable_guideline_filter is False

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
