"""PanelLabelJudgeApi 面板标签业务逻辑单元测试"""
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from schemas.inference_context import InferenceContext
from vie_plugin_panel_label.business_logic import PanelLabelJudgeApi
from vie_plugin_panel_label.models import PanelInfo, ErrorType, PanellabelItem
from schemas.data_base import DetectionItem


@pytest.fixture
def standard_result():
    """标准 OCR 顺序：生产由请求 line_order 下发，测试直接以列表传入。"""
    return ["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"]


@pytest.fixture
def guideline():
    """引导框归一化坐标 x,y,w,h：生产由请求 guideline_coordinates 下发。"""
    return (0.1, 0.1, 0.8, 0.8)


@pytest.fixture
def judge():
    with patch.object(
        PanelLabelJudgeApi, "_initialize_model", return_value=None
    ):
        judge = PanelLabelJudgeApi(MagicMock())
        judge.detector = MagicMock()
        judge.w = 1000
        judge.h = 800
        return judge


class TestPanelLabelAnalyze:
    def test_fix_slash_misrecognition_paired_brackets(self, judge):
        """成对括号不做修改"""
        assert judge._fix_slash_misrecognition("QF2-1(53)") == "QF2-1(53)"
        assert judge._fix_slash_misrecognition("((PE1))") == "((PE1))"
        assert judge._fix_slash_misrecognition("QF(PE)3") == "QF(PE)3"

    def test_fix_slash_misrecognition_single_left_bracket(self, judge):
        """单个左括号修正为 / """
        assert judge._fix_slash_misrecognition("QF2-1(PE1") == "QF2-1/PE1"
        assert judge._fix_slash_misrecognition("(PE1-J1") == "/PE1-J1"

    def test_fix_slash_misrecognition_single_right_bracket(self, judge):
        """单个右括号修正为 / """
        assert judge._fix_slash_misrecognition("QF2-1)PE1") == "QF2-1/PE1"
        assert judge._fix_slash_misrecognition("PE1-J1)") == "PE1-J1/"

    def test_fix_slash_misrecognition_all_left_brackets(self, judge):
        """全部左括号（不成对）全部修正为 / """
        assert judge._fix_slash_misrecognition("((PE1-J1") == "//PE1-J1"
        assert judge._fix_slash_misrecognition("QF(PE(3") == "QF/PE/3"

    def test_fix_slash_misrecognition_all_right_brackets(self, judge):
        """全部右括号（不成对）全部修正为 / """
        assert judge._fix_slash_misrecognition("PE1-J1))") == "PE1-J1//"
        assert judge._fix_slash_misrecognition("QF)PE)3") == "QF/PE/3"

    def test_fix_slash_misrecognition_no_brackets(self, judge):
        """无括号文本原样返回"""
        assert judge._fix_slash_misrecognition("QF2-1/PE1") == "QF2-1/PE1"
        assert judge._fix_slash_misrecognition("ABCD") == "ABCD"

    def test_fix_slash_misrecognition_in_analyze(self, judge, standard_result):
        """OCR 误识别括号为斜杠后，analyze 仍能正确匹配"""
        # 标准：LINE1/xxx，OCR 识别为 LINE1(xxx（左括号误判）
        observed = PanellabelItem(
            Points=[
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
                [10, 200, 300, 200, 300, 400, 10, 400],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["LINE1(xxx", "LINE2/yyy", "LINE3/zzz"],
            confidence=[0.95, 0.88, 0.72],
        )
        result = judge.analyze(observed, standard_result)
        assert result.result is True
        # 确认修正后的 observed_result 中括号已被替换
        assert result.observed_result[0] == "LINE1/xxx"

    def test_perfect_match(self, judge, standard_result):
        observed = PanellabelItem(
            Points=[
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
                [10, 200, 300, 200, 300, 400, 10, 400],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"],
            confidence=[0.95, 0.88, 0.72],
        )
        result = judge.analyze(observed, standard_result)
        assert result.result is True
        assert result.message == ErrorType.OK.value

    def test_count_mismatch(self, judge, standard_result):
        observed = PanellabelItem(
            Points=[[10, 20, 30, 40, 50, 60, 70, 80]],
            index=[0],
            class_id=[0],
            texts=["LINE1/aaa"],  # 只有1个，标准有3个
            confidence=[0.95],
        )
        result = judge.analyze(observed, standard_result)
        assert result.result is False
        assert result.message == ErrorType.MISSING.value

    def test_content_mismatch(self, judge, standard_result):
        observed = PanellabelItem(
            Points=[
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
                [10, 200, 300, 200, 300, 400, 10, 400],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["LINE1/xxx", "WRONG/yyy", "LINE3/zzz"],
            confidence=[0.9, 0.8, 0.7],
        )
        result = judge.analyze(observed, standard_result)
        assert result.result is False
        assert result.message == ErrorType.MISMATCH.value
        assert 1 in result.error_indexs

    def test_case_insensitive_match(self, judge, standard_result):
        observed = PanellabelItem(
            Points=[
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
                [10, 200, 300, 200, 300, 400, 10, 400],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["line1/xxx", "Line2/yyy", "LINE3/zzz"],  # 大小写不同
            confidence=[0.9, 0.8, 0.7],
        )
        result = judge.analyze(observed, standard_result)
        assert result.result is True
        assert result.message == ErrorType.OK.value

    def test_only_compare_before_slash(self, judge, standard_result):
        observed = PanellabelItem(
            Points=[
                [10, 20, 30, 40, 50, 60, 70, 80],
                [100, 200, 300, 400, 500, 600, 700, 800],
                [10, 200, 300, 200, 300, 400, 10, 400],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=[
                "LINE1/different_suffix",
                "LINE2/other",
                "LINE3/whatever",
            ],
            confidence=[0.9, 0.8, 0.7],
        )
        result = judge.analyze(observed, standard_result, rule="front")
        assert result.result is True


class TestGuidelineFilter:
    def test_points_inside_rect_kept(self, judge, guideline):
        judge.w = 1000
        judge.h = 800
        results = PanellabelItem(
            Points=[
                [200, 200, 300, 200, 300, 300, 200, 300],  # 在 (100,80,800,640) 内
            ],
            index=[0],
            class_id=[0],
            texts=["KEEP"],
            confidence=[0.9],
        )
        filtered = judge.guideline_filter(results, guideline, judge.w, judge.h)
        assert len(filtered.Points) == 1
        assert filtered.texts == ["KEEP"]

    def test_points_outside_rect_excluded(self, judge, guideline):
        judge.w = 1000
        judge.h = 800
        results = PanellabelItem(
            Points=[
                [950, 750, 960, 750, 960, 760, 950, 760],  # 在 roi 外
            ],
            index=[0],
            class_id=[0],
            texts=["DISCARD"],
            confidence=[0.9],
        )
        filtered = judge.guideline_filter(results, guideline, judge.w, judge.h)
        assert len(filtered.Points) == 0

    def test_polygon_points_inside_kept(self, judge):
        """8 值四边形：角点全在内 → 保留"""
        judge.w = 1000
        judge.h = 800
        # 归一化四角（顺时针）：左上(0.1,0.1) 右上(0.9,0.1) 右下(0.9,0.9) 左下(0.1,0.9)
        poly = (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9)
        results = PanellabelItem(
            Points=[[200, 200, 300, 200, 300, 300, 200, 300]],  # 全在四边形内
            index=[0],
            class_id=[0],
            texts=["KEEP"],
            confidence=[0.9],
        )
        filtered = judge.guideline_filter(results, poly, judge.w, judge.h)
        assert len(filtered.Points) == 1
        assert filtered.texts == ["KEEP"]

    def test_polygon_points_outside_excluded(self, judge):
        """8 值四边形：有角点在外 → 剔除"""
        judge.w = 1000
        judge.h = 800
        poly = (0.1, 0.1, 0.9, 0.1, 0.9, 0.9, 0.1, 0.9)
        results = PanellabelItem(
            Points=[[950, 750, 990, 750, 990, 790, 950, 790]],  # 在四边形外
            index=[0],
            class_id=[0],
            texts=["DISCARD"],
            confidence=[0.9],
        )
        filtered = judge.guideline_filter(results, poly, judge.w, judge.h)
        assert len(filtered.Points) == 0

    def test_polygon_partial_outside_excluded(self, judge):
        """8 值四边形：部分角点越界即整框剔除（与矩形 all-inside 语义一致）"""
        judge.w = 1000
        judge.h = 800
        # 窄四边形，右边界约 x=300
        poly = (0.1, 0.1, 0.3, 0.1, 0.3, 0.9, 0.1, 0.9)
        results = PanellabelItem(
            Points=[[200, 200, 400, 200, 400, 300, 200, 300]],  # 右两角 x=400 越界
            index=[0],
            class_id=[0],
            texts=["DISCARD"],
            confidence=[0.9],
        )
        filtered = judge.guideline_filter(results, poly, judge.w, judge.h)
        assert len(filtered.Points) == 0


class TestGuidelineFilterSwitch:
    def test_judge_default_matches_config(self, judge):
        """实例开关与配置默认一致（当前默认关闭）"""
        from vie_plugin_panel_label.config import PanelLabelConfig
        assert judge.enable_guideline_filter is PanelLabelConfig().enable_guideline_filter

    def test_disabled_skips_filter_and_guideline_optional(self, judge):
        """开关关闭时跳过 ROI 过滤：ROI 外的框照常参与判定，guideline 缺失也不报错"""
        judge.enable_guideline_filter = False
        results = PanellabelItem(
            Points=[[950, 750, 990, 750, 990, 790, 950, 790]],  # 在默认 ROI 外
            index=[0],
            class_id=[0],
            texts=["LINE1/xxx"],
            confidence=[0.9],
        )
        ctx = _make_ctx(judge, results, ["LINE1/xxx"], None)
        judge.business_post_process(ctx)
        assert ctx.result.status is True
        assert len(ctx.result.detailList) == 1

    def test_enabled_missing_guideline_raises(self, judge, standard_result):
        """开关开启时 guideline 仍为必要参数，缺失报参数错误"""
        from schemas.exceptions import InvalidParamsError

        judge.enable_guideline_filter = True
        ctx = _make_ctx(judge, PanellabelItem(), standard_result, None)
        with pytest.raises(InvalidParamsError) as exc_info:
            judge.business_post_process(ctx)
        assert "guideline_coordinates" in exc_info.value.error_msg


def _make_ctx(judge, results, standard_result, guideline):
    """构造带 standard_result / guideline 透传袋的推理上下文。"""
    ctx = InferenceContext(
        image=np.zeros((judge.h, judge.w, 3), dtype=np.uint8),
        h=judge.h, w=judge.w, product_type="TYPE1",
        extra={"standard_result": standard_result, "guideline": guideline},
    )
    ctx.raw_result = results
    return ctx


class TestBusinessLogicPostProcess:
    def test_success_scenario(self, judge, standard_result, guideline):
        # ROI: (0.1,0.1,0.8,0.8) * (w=1000,h=800) = rect(100,80,800,640)
        # 坐标均位于 ROI 内部
        results = PanellabelItem(
            Points=[
                [200, 200, 300, 200, 300, 300, 200, 300],
                [400, 200, 500, 200, 500, 300, 400, 300],
                [600, 200, 700, 200, 700, 300, 600, 300],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["LINE1/xxx", "LINE2/yyy", "LINE3/zzz"],
            confidence=[0.95, 0.88, 0.72],
        )
        judge.enable_guideline_filter = True
        ctx = _make_ctx(judge, results, standard_result, guideline)
        judge.business_post_process(ctx)
        mom = ctx.result
        assert mom.status is True
        assert mom.message == ErrorType.OK.value
        assert len(mom.detailList) == 3
        for item in mom.detailList:
            assert item.status is True

    def test_mismatch_items_have_false_status(self, judge, standard_result, guideline):
        # ROI: (0.1,0.1,0.8,0.8) * (w=1000,h=800) = rect(100,80,800,640)
        # 坐标均位于 ROI 内部
        results = PanellabelItem(
            Points=[
                [200, 200, 300, 200, 300, 300, 200, 300],
                [400, 200, 500, 200, 500, 300, 400, 300],
                [600, 200, 700, 200, 700, 300, 600, 300],
            ],
            index=[0, 1, 2],
            class_id=[0, 0, 0],
            texts=["LINE1/xxx", "WRONG_DATA", "LINE3/zzz"],
            confidence=[0.95, 0.88, 0.72],
        )
        judge.enable_guideline_filter = True
        ctx = _make_ctx(judge, results, standard_result, guideline)
        judge.business_post_process(ctx)
        mom = ctx.result
        assert mom.status is False
        # 第1个和第3个正确，第2个错误
        assert mom.detailList[0].status is True
        assert mom.detailList[1].status is False
        assert mom.detailList[2].status is True


class TestPanelInfo:
    def test_default_values(self):
        info = PanelInfo()
        assert info.result is False
        assert info.message == ErrorType.UNKNOWN.value
        assert info.standard_result == []
        assert info.observed_result == []
        assert info.error_indexs == []

    def test_custom_values(self):
        info = PanelInfo(
            result=True,
            product_type="T1",
            standard_result=["A", "B"],
            observed_result=["a", "b"],
            observed_result_points=[[1, 2, 3, 4, 5, 6, 7, 8]],
            message=ErrorType.OK.value,
        )
        assert info.result is True
        assert info.product_type == "T1"
        assert info.standard_result == ["A", "B"]
