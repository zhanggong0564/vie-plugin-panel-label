'''
@Author       : gongzhang4
@Date         : 2026-03-27 12:16:00
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-03-27 12:16:02
@FilePath     : panel_label_schemas.py
@Description  :
'''

from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import List, Dict, Optional, Tuple, Literal, Union
from schemas.common import VisualReferenceParams


class ModelParams(VisualReferenceParams):
    """modelParams整体模型（guide_line/example_images设为可选）"""

    product_type: str = Field(..., description="产品型号(例如:QF2)")
    rule: Literal["front", "back", "all"] = Field(default="all", description="字符比较规则：front=斜杠前，back=斜杠后，all=全检")
    # 标准线标顺序与引导框由业务随请求下发，不再从本地词典读取。
    # line_order：分号分隔候选顺序，候选内用逗号分隔，如 "TK2-2,TK2-1;TK2-1,TK2-2"。
    line_order: List[List[Optional[str]]] = Field(
        ...,
        description=(
            "候选线标顺序；分号分隔候选，逗号分隔线标，如 'TK2-2,TK2-1;TK2-1,TK2-2'；"
            "任意位置可用 null 占位并跳过文字比对，仍计入数量，"
            "支持字符串 null（大小写不敏感）或列表中的 JSON null"
        ),
    )
    # guideline_coordinates：归一化引导区域。4 值=轴对齐矩形 x,y,w,h；
    # 8 值=四边形 x1,y1,x2,y2,x3,y3,x4,y4（顺时针四角）。
    guideline_coordinates: Tuple[float, ...] = Field(
        ...,
        description="必填引导区域归一化坐标：4 值=矩形 x,y,w,h；8 值=四边形顺时针四角",
    )

    @field_validator(
        "line_order", mode="before",
        json_schema_input_type=Union[str, List[Optional[str]], List[List[Optional[str]]]],
    )
    @classmethod
    def _split_line_order(cls, v):
        """将字符串、一维列表或二维列表统一为非空候选顺序列表。"""
        if isinstance(v, str):
            raw_candidates = [candidate.split(",") for candidate in v.split(";")]
        elif isinstance(v, (list, tuple)):
            if not v:
                raise ValueError("line_order 至少需要一个候选顺序")
            if all(item is None or isinstance(item, str) for item in v):
                raw_candidates = [v]
            elif all(isinstance(item, (list, tuple)) for item in v):
                raw_candidates = v
            else:
                raise ValueError("line_order 必须是字符串、一维字符串列表或二维字符串列表")
        else:
            return v

        candidates = []
        for candidate in raw_candidates:
            if any(item is not None and not isinstance(item, str) for item in candidate):
                raise ValueError("line_order 中的线标必须是字符串或 null")
            normalized = [
                None if item is None or item.strip().lower() == "null" else item.strip()
                for item in candidate
                if item is None or item.strip()
            ]
            if not normalized:
                raise ValueError("line_order 候选顺序不能为空")
            candidates.append(normalized)
        return candidates

    @field_validator(
        "guideline_coordinates", mode="before",
        json_schema_input_type=Union[str, Tuple[float, ...]],
    )
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
