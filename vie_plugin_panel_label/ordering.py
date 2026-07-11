'''
@Description : 线标排序引擎（按型号声明的 sort_mode 决定，几何鲁棒、零运行期阈值调参）。

设计要点：
  - 排序只依赖检测框的几何中心，与 OCR 识别出的文字无关——这样校验的才是
    "线有没有接对位置"，两根物理接反但都识别正确的线不会被漏判。
  - linear：中心点做 PCA 求排列主轴，沿主轴投影排序。方向约定：主轴偏竖直
    则朝下为正（上→下），偏水平则朝右为正（左→右）；单条线/斜排/单行单列通吃，零阈值。
  - columns:N：按横向显著间隙把中心聚成 N 列（左→右），列内沿竖直排（上→下）。
  - rows:N：columns 的转置——按纵向显著间隙把中心聚成 N 行（上→下），行内沿水平排（左→右）。
    EXIF 旋转会把"视觉上的左右两列"变成模型坐标系里的上下两行，此时该用 rows。
  - 方向修饰后缀（可叠加）：
      :rev    整体反向
      :colrev columns 列序反向（右列先）；rows 行内反向（行内右→左）
      :rowrev columns 列内反向（列内下→上）；rows 行序反向（下行先）
    例：QF2 在模型坐标系是上下两行、上行先、行内左→右 → "rows:2"。

模式串语法： "linear" | "linear:rev"
            | "columns:N" | "columns:N:rowrev" | "columns:N:colrev" | "columns:N:colrev:rowrev"
            | "rows:N"    | "rows:N:rowrev"     | "rows:N:colrev"     | "rows:N:colrev:rowrev"
'''

from typing import List, Optional, Tuple

import numpy as np


def _centers(points) -> np.ndarray:
    """把多边形列表（每个为扁平 [x1,y1,x2,y2,...]）转成 (N,2) 中心点矩阵。"""
    cs = []
    for p in points:
        arr = np.asarray(p, dtype=np.float64).reshape(-1, 2)
        cs.append(arr.mean(axis=0))
    return np.asarray(cs, dtype=np.float64) if cs else np.zeros((0, 2), dtype=np.float64)


def _principal_axis(centers: np.ndarray) -> np.ndarray:
    """中心点的 PCA 最大特征向量（排列主轴），单位向量。"""
    c = centers - centers.mean(axis=0)
    cov = np.cov(c.T)
    if not np.all(np.isfinite(cov)):
        return np.array([1.0, 0.0])
    w, v = np.linalg.eigh(cov)
    u = v[:, int(np.argmax(w))]
    n = np.linalg.norm(u)
    return u / n if n > 0 else np.array([1.0, 0.0])


def _orient_down_right(u: np.ndarray) -> np.ndarray:
    """约定主轴方向：偏竖直→朝下(+y)为正，偏水平→朝右(+x)为正。"""
    if abs(u[1]) >= abs(u[0]):  # 偏竖直
        return u if u[1] >= 0 else -u
    return u if u[0] >= 0 else -u  # 偏水平


def _linear_order(centers: np.ndarray) -> List[int]:
    if len(centers) <= 1:
        return list(range(len(centers)))
    u = _orient_down_right(_principal_axis(centers))
    s = (centers - centers.mean(axis=0)) @ u
    return list(np.argsort(s, kind="stable"))


def _columns_order(centers: np.ndarray, ncol: int, colrev: bool = False, rowrev: bool = False) -> List[int]:
    n = len(centers)
    if n <= 1 or ncol <= 1:
        # 退化为单列：按竖直排（rowrev 时下→上）
        col = list(np.argsort(centers[:, 1], kind="stable")) if n else []
        return col[::-1] if rowrev else col
    # 按 x 升序，找 ncol-1 个最大横向间隙作为列分界
    order_x = list(np.argsort(centers[:, 0], kind="stable"))
    xs = centers[order_x, 0]
    gaps = np.diff(xs)
    cut_set = set()
    if len(gaps) >= ncol - 1:
        cut_set = {int(i) for i in np.argsort(gaps)[-(ncol - 1):]}
    # 切分成列（已按 x 升序，故列天然左→右）
    columns: List[List[int]] = []
    cur = [order_x[0]]
    for i in range(1, len(order_x)):
        if (i - 1) in cut_set:
            columns.append(cur)
            cur = []
        cur.append(order_x[i])
    columns.append(cur)
    if colrev:  # 列序反向：右列先
        columns = columns[::-1]
    # 列内按竖直（y 升序，上→下；rowrev 时下→上）
    result: List[int] = []
    for col in columns:
        col_sorted = sorted(col, key=lambda idx: centers[idx, 1])
        if rowrev:
            col_sorted = col_sorted[::-1]
        result.extend(col_sorted)
    return result


def _rows_order(centers: np.ndarray, nrow: int, rowrev: bool = False, colrev: bool = False) -> List[int]:
    """columns 的转置：按竖直显著间隙把中心聚成 nrow 行（上→下），行内沿水平排（左→右）。

    rowrev=行序反向（下行先），colrev=行内反向（右→左）。x/y 与 _columns_order 完全互换。
    """
    n = len(centers)
    if n <= 1 or nrow <= 1:
        # 退化为单行：按水平排（colrev 时右→左）
        row = list(np.argsort(centers[:, 0], kind="stable")) if n else []
        return row[::-1] if colrev else row
    # 按 y 升序，找 nrow-1 个最大竖直间隙作为行分界
    order_y = list(np.argsort(centers[:, 1], kind="stable"))
    ys = centers[order_y, 1]
    gaps = np.diff(ys)
    cut_set = set()
    if len(gaps) >= nrow - 1:
        cut_set = {int(i) for i in np.argsort(gaps)[-(nrow - 1):]}
    # 切分成行（已按 y 升序，故行天然上→下）
    rows: List[List[int]] = []
    cur = [order_y[0]]
    for i in range(1, len(order_y)):
        if (i - 1) in cut_set:
            rows.append(cur)
            cur = []
        cur.append(order_y[i])
    rows.append(cur)
    if rowrev:  # 行序反向：下行先
        rows = rows[::-1]
    # 行内按水平（x 升序，左→右；colrev 时右→左）
    result: List[int] = []
    for row in rows:
        row_sorted = sorted(row, key=lambda idx: centers[idx, 0])
        if colrev:
            row_sorted = row_sorted[::-1]
        result.extend(row_sorted)
    return result


def _parse_mode(sort_mode: str) -> Tuple[str, int, set]:
    """解析模式串 → (mode, count, tokens)。tokens 含 rev/colrev/rowrev。非法值回退 linear。

    count 对 columns 即列数 ncol、对 rows 即行数 nrow。
    """
    parts = (sort_mode or "linear").strip().split(":")
    mode = parts[0].lower() or "linear"
    tokens = {p.lower() for p in parts[1:]}
    count = 1
    if mode in ("columns", "rows"):
        for p in parts[1:]:
            if p.isdigit():
                count = int(p)
                break
    if mode not in ("linear", "columns", "rows"):
        mode = "linear"
    return mode, count, tokens


def compute_order(points, sort_mode: str = "linear") -> List[int]:
    """按 sort_mode 计算 points 的排序索引（permutation）。"""
    centers = _centers(points)
    if len(centers) <= 1:
        return list(range(len(centers)))
    mode, count, tokens = _parse_mode(sort_mode)
    if mode == "columns":
        perm = _columns_order(
            centers, count, colrev=("colrev" in tokens), rowrev=("rowrev" in tokens)
        )
    elif mode == "rows":
        perm = _rows_order(
            centers, count, rowrev=("rowrev" in tokens), colrev=("colrev" in tokens)
        )
    else:
        perm = _linear_order(centers)
    if "rev" in tokens:  # 整体反向（linear 反向；columns 等价 colrev+rowrev）
        perm = perm[::-1]
    return perm


def _reindex(seq, perm):
    """按 perm 重排；长度不匹配（如空列表占位）则原样返回，避免越界。"""
    if seq is None or len(seq) != len(perm):
        return seq
    return [seq[i] for i in perm]


def order_panel_item(item, sort_mode: str = "linear"):
    """按 sort_mode 重排 PanellabelItem 的所有平行字段，返回新实例。"""
    perm = compute_order(item.Points, sort_mode)
    if perm == list(range(len(perm))):
        return item
    cls = type(item)
    return cls(
        Points=_reindex(item.Points, perm),
        index=_reindex(item.index, perm),
        class_id=_reindex(item.class_id, perm),
        texts=_reindex(item.texts, perm),
        confidence=_reindex(item.confidence, perm),
        text_crops=_reindex(item.text_crops, perm),
    )
