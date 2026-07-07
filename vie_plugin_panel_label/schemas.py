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
    # guideline_coordinates：归一化引导区域。4 值=轴对齐矩形 x,y,w,h；
    # 8 值=四边形 x1,y1,x2,y2,x3,y3,x4,y4（顺时针四角）。
    guideline_coordinates: Tuple[float, ...] = Field(
        ...,
        description="必填引导区域归一化坐标：4 值=矩形 x,y,w,h；8 值=四边形顺时针四角",
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
        """把逗号分隔字符串拆成浮点序列；仅允许 4 值(矩形)或 8 值(四边形)。"""
        if v is None:
            return v
        if isinstance(v, str):
            v = [p.strip() for p in v.split(",") if p.strip() != ""]
        if len(v) not in (4, 8):
            raise ValueError("guideline_coordinates 长度必须为 4(矩形 x,y,w,h) 或 8(四边形顺时针四角)")
        return v


class PanelLabelRequest(BaseModel):
    """请求中json_data对应的结构化模型"""

    # 允许透传未声明字段（如数据回流型号兜底用到的 AICameraModel 列表），
    # 不为其新增强类型字段，由 Router._extract_product_type 宽松读取。
    model_config = ConfigDict(extra="allow")

    product: str = Field(..., description="产品类型")
    type: str = Field(..., description="物料号")
    modelParams: ModelParams = Field(..., description="模型参数")
