'''
@Author       : gongzhang4
@Date         : 2026-03-27 11:22:29
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:20:03
@FilePath     : panel_routers.py
@Description  : 线标检测接口
'''

import os
from typing import Any, Optional

import numpy as np

from routers.base_router import BaseRouter, BackflowTarget, UNKNOWN_MODEL_DIR
from schemas.data_base import InputParamsBusiness
from .schemas import PanelLabelRequest
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("panel_label")


class PanelLabelRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return PanelLabelRequest(**json_dict)

    def resolve_backflow_target(self, original_filename, fallback_product_type=None):
        """线标专属落盘命名：顶层场景取文件名首段，型号取 API product_type，文件名保留原名。

        路径形如 ``data/{文件名按-分割首段}/{日期}/{型号}/{ok|ng}/images|records/{原始文件名}``：
          - 场景目录 = 原始文件名按 '-' 分割的第一段（如 'AI-集中式-…' → 'AI'），不再解析中文场景名
          - 型号目录 = 请求 product_type，空则取 AICameraModel.AIParameterValue（见 _extract_product_type），
            仍为空回退 _unknown_model；不再从文件名解析型号
          - 落盘文件名 = 原始文件名（去扩展名，扩展名由框架据原文件名补回），不再改写为时间戳
        """
        safe_filename = self._safe_client_filename(original_filename)
        stem = os.path.splitext(safe_filename)[0] or safe_filename
        # 在去扩展名的 stem 上切首段，避免无 '-' 文件名把扩展名带进场景目录
        scene = stem.split("-", 1)[0] or self.detector_type
        model_dir = (
            self._sanitize_dir_name(fallback_product_type)
            if fallback_product_type
            else UNKNOWN_MODEL_DIR
        )
        return BackflowTarget(scene_dir=scene, model_dir=model_dir, save_stem=stem)

    @staticmethod
    def _extract_product_type(request_params: Any) -> Optional[str]:
        """型号兜底：优先 modelParams.product_type，为空时取 AICameraModel 的 AIParameterValue。

        AICameraModel 不在 PanelLabelRequest 的强类型字段里，靠 ``extra='allow'`` 透传，
        这里按属性/字典两种形态宽松读取，取列表中第一个非空 AIParameterValue。
        """
        mp = getattr(request_params, "modelParams", None)
        product_type = getattr(mp, "product_type", None) if mp is not None else None
        if product_type and str(product_type).strip():
            return product_type
        ai_models = getattr(request_params, "AICameraModel", None)
        if isinstance(ai_models, (list, tuple)):
            for item in ai_models:
                value = (
                    item.get("AIParameterValue")
                    if isinstance(item, dict)
                    else getattr(item, "AIParameterValue", None)
                )
                if value and str(value).strip():
                    return value
        return None

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
