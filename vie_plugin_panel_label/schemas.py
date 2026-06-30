'''
@Author       : gongzhang4
@Date         : 2026-03-27 12:16:00
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:16:02
@FilePath     : panel_label_schemas.py
@Description  :
'''

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Optional, Tuple, Literal  # 新增Optional
from schemas.common import GuideLineItem, ExampleImageItem


class ModelParams(BaseModel):
    """modelParams整体模型（guide_line/example_images设为可选）"""

    guide_line: Optional[List[GuideLineItem]] = Field(default_factory=list, description="参考线图片列表")
    example_images: Optional[List[ExampleImageItem]] = Field(default_factory=list, description="示例图片列表")
    product_type: str = Field(..., description="产品型号(例如:QF2)")
    rule: Literal["front", "back", "all"] = Field(default="all", description="字符比较规则：front=斜杠前，back=斜杠后，all=全检")
    # 标准线标顺序与引导框由业务随请求下发，不再从本地词典读取。
    # line_order：逗号分隔的标准 OCR 顺序，如 "TK2-2,TK2-1"。
    line_order: List[str] = Field(..., description="标准线标顺序，逗号分隔，如 'TK2-2,TK2-1'")
    # guideline_coordinates：归一化引导框 x,y,w,h，如 "0.154,0.4075,0.692,0.336"。
    # 服务端关闭 guideline 过滤（PANEL_LABEL_GUIDELINE_FILTER=false）时可不传；开启时缺失由业务层报参数错误。
    guideline_coordinates: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="引导框归一化坐标 x,y,w,h，如 '0.154,0.4075,0.692,0.336'；关闭 guideline 过滤时可省略"
    )

    @field_validator("line_order", mode="before")
    @classmethod
    def _split_line_order(cls, v):
        """把逗号分隔字符串拆成去空白、去空项的列表；已是列表则原样放行。"""
        if isinstance(v, str):
            return [t.strip() for t in v.split(",") if t.strip()]
        return v

    @field_validator("guideline_coordinates", mode="before")
    @classmethod
    def _split_guideline(cls, v):
        """把逗号分隔字符串拆成浮点序列，交由 Tuple[float×4] 校验数量与类型。"""
        if isinstance(v, str):
            return [p.strip() for p in v.split(",") if p.strip() != ""]
        return v


class PanelLabelRequest(BaseModel):
    """请求中json_data对应的结构化模型"""

    # 允许透传未声明字段（如数据回流型号兜底用到的 AICameraModel 列表），
    # 不为其新增强类型字段，由 Router._extract_product_type 宽松读取。
    model_config = ConfigDict(extra="allow")

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
