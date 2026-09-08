"""Typed runtime configuration for the panel-label scene."""

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from services.base import SceneSettings


class PanelLabelConfig(SceneSettings):
    model_config = SettingsConfigDict(
        env_prefix="PANEL_LABEL_",
        env_file=".env",
        extra="ignore",
    )

    model_path: str = (
        "./weights/panel_label/v2/rfdetr-seg-nano_v1.3.onnx"
    )
    orient_model_path: str = (
        "./weights/panel_label/v2/textline_ori_lcnet_v2.onnx"
    )
    orient_metadata_path: str = (
        "./weights/panel_label/v2/textline_ori_lcnet_v2/inference.yml"
    )
    text_recognition_model_path: str = (
        "./weights/panel_label/v2/"
        "PP-OCRv5_server_rec_merged_v6_diff_lr.onnx"
    )
    text_recognition_metadata_path: str = (
        "./weights/panel_label/v2/"
        "PP-OCRv5_server_rec_merged_v6_diff_lr/inference.yml"
    )
    conf_threshold: float = Field(default=0.6, ge=0, le=1)
    nms_threshold: float = Field(default=0.8, ge=0, le=1)
    mask_threshold: float = Field(default=0.7, gt=0, lt=1)
    text_orient_score_thresh: float = Field(default=0.9, ge=0, le=1)
    text_rec_score_thresh: float = Field(default=0.7, ge=0, le=1)
    text_rec_input_shape: tuple[int, int, int] | None = None
    cpu_fast_path: bool = True
    enable_guideline_filter: bool = Field(
        default=True,
        validation_alias="PANEL_LABEL_GUIDELINE_FILTER",
    )
    dedup_overlap_thresh: float = Field(
        default=0.6,
        ge=0,
        validation_alias="PANEL_LABEL_DEDUP_OVERLAP",
    )
    guideline_overlap_thresh: float = Field(
        default=0.9,
        ge=0,
        le=1,
        validation_alias="PANEL_LABEL_GUIDELINE_OVERLAP",
    )

    @property
    def confThreshold(self) -> float:
        return self.conf_threshold

    @property
    def nmsThreshold(self) -> float:
        return self.nms_threshold
