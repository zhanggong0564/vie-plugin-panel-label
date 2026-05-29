'''
@Author       : gongzhang4
@Date         : 2026-03-27 11:22:29
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:20:03
@FilePath     : panel_routers.py
@Description  : 线标检测接口
'''

from routers.base_router import BaseRouter
from .schemas import PanelLabelRequest
import numpy as np
from schemas.data_base import InputParamsBusiness
from . import business_logic  # noqa: F401  导入即触发 @detection_factory.register("panel_label")


class PanelLabelRouter(BaseRouter):
    def __init__(self, router_name, api_path, summary, description, detector_type, tag=None):
        super().__init__(router_name, api_path, summary, description, detector_type, tag=tag)

    def request_schema(self, json_dict):
        return PanelLabelRequest(**json_dict)

    def get_inputs(self, request_params: PanelLabelRequest, image: np.ndarray):
        product_type = request_params.modelParams.product_type
        rule = request_params.modelParams.rule
        input = InputParamsBusiness(image=image, product_type=product_type, rule=rule)
        return input


panel_label_router = PanelLabelRouter(
    router_name="panel_router",
    api_path="/panel_label_detect",
    summary="线标检测接口",
    description="根据输入的图像和产品类型，返回检测结果",
    detector_type="panel_label",
    tag="线标OCR检测",
)
