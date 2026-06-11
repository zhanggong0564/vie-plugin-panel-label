'''
@Author       : gongzhang4
@Date         : 2026-03-27 11:22:29
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:20:03
@FilePath     : panel_routers.py
@Description  : 线标检测接口
'''

import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from routers.base_router import BaseRouter, BackflowTarget
from schemas.data_base import InputParamsBusiness
from .schemas import PanelLabelRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("panel_label")

# 线标文件名形如 "AI-中压线标检验TK2-1-1764780181920.jpg" / "1+X线标检验PE1-A-1779526099406.jpg":
#   - 可选前缀 "AI-"（上游 AI 处理标记），先剥掉再解析
#   - 末段 -<digits> 是 timestamp，去掉扩展名后用贪婪匹配切出
#   - 型号尾部的 -<digits> 是单面多拍的图片序号（如 TK2-1 的 -1），按型号聚合需去掉
_FILENAME_TS_RE = re.compile(r"^(.+)-(\d+)$")
_AI_PREFIX_RE = re.compile(r"^AI-", re.IGNORECASE)
_MODEL_INDEX_SUFFIX_RE = re.compile(r"-\d+$")


class PanelLabelRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return PanelLabelRequest(**json_dict)

    def resolve_backflow_target(self, original_filename, fallback_product_type=None):
        """线标专属：从文件名拆出中文场景名与型号，按型号聚合、按时间戳命名。

        解析成功用 (场景, 型号, timestamp) 落盘；不符合规则时回退框架默认
        （场景=detector_type，型号=product_type / _unknown_model）。
        """
        scene, model, timestamp = self._parse_filename(original_filename)
        if model and timestamp:
            return BackflowTarget(
                scene_dir=self._sanitize_dir_name(scene) if scene else self.detector_type,
                model_dir=model,
                save_stem=timestamp,
            )
        return super().resolve_backflow_target(original_filename, fallback_product_type)

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """从文件名解析 (场景, 型号, timestamp)。

        形如 'AI-中压线标检验TK2-1-1764780181920.jpg' / '1+X线标检验PE1-A-...':
        去扩展名 → 剥掉可选 'AI-' 前缀 → 末尾 -<digits> 切出 timestamp →
        前半段最后一个中文字符之前是场景名，之后是型号 → 型号尾部 -<digits>
        图片序号去掉（TK2-1 → TK2）。不符合规则时返回 (None, None, None)。
        """
        stem = _AI_PREFIX_RE.sub("", Path(filename).stem, count=1)
        m = _FILENAME_TS_RE.match(stem)
        if not m:
            return None, None, None
        body, timestamp = m.group(1), m.group(2)
        last_cjk_idx = -1
        for i, ch in enumerate(body):
            if "一" <= ch <= "鿿":
                last_cjk_idx = i
        if last_cjk_idx < 0:
            return None, None, None
        scene = body[: last_cjk_idx + 1]
        raw_model = body[last_cjk_idx + 1 :]
        model = _MODEL_INDEX_SUFFIX_RE.sub("", raw_model, count=1) or raw_model
        return scene, model, timestamp

    def get_inputs(self, request_params: PanelLabelRequest, image: np.ndarray):
        mp = request_params.modelParams
        # 标准顺序与引导框随请求下发，经 schema 校验/解析后透传给业务层。
        extra = {
            "standard_result": mp.line_order,
            "guideline": mp.guideline_coordinates,
        }
        input = InputParamsBusiness(
            image=image, product_type=mp.product_type, rule=mp.rule, extra=extra
        )
        return input


panel_label_router = PanelLabelRouter(
    router_name="panel_router",
    api_path="/panel_label_detect",
    summary="线标检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="panel_label",
    tag="线标OCR检测",
)
