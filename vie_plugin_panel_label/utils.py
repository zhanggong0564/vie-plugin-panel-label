'''
@Author       : gongzhang4
@Date         : 2026-02-28 01:22:13
@LastEditors  : 张弓 zhanggong1@sungrowpower.com
@LastEditTime : 2026-05-14 01:54:13
@FilePath     : utils.py
@Description  :
'''

import math

import cv2
import numpy as np
from typing import List, Tuple


def _rect_long_side_angle_deg(rect):
    (_, _), (w, h), a = rect
    if h > w:
        a = a + 90.0
    return a % 180.0  # [0,180)


def _angle_to_vertical_distance(a_deg: float) -> float:
    """
    到竖直(90°)的最小角距离，范围 [0,90]
    """
    d = abs(a_deg - 90.0)
    return min(d, 180.0 - d)


def sort_mask(
    ori_img: np.ndarray,
    points: np.ndarray,
    row_alpha: float = 0.6,  # 分行阈值系数
) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    简化排序：
    1) 先用整体布局宽高比 W/H 判断是否偏单列
    2) 单列：按 y 排序
    3) 否则：按 y 分行，行内按 x
    """
    if points is None or len(points) == 0:
        return [], np.array([], dtype=np.int64)

    items = []  # (idx, cx, cy, w,h, angle)
    vertical_count = 0
    for i, p in enumerate(points):
        arr = np.array(p, dtype=np.float32).reshape(-1, 2)
        rect = cv2.minAreaRect(arr)
        angle = _rect_long_side_angle_deg(rect)
        angle = _angle_to_vertical_distance(angle)  # 到竖直(90°)的最小角距离,越小越竖
        if angle < 45.0:
            vertical_count += 1
        (cx, cy), (w, h), _ = rect
        items.append((i, cx, cy, w, h, angle))

    ratio = vertical_count / len(items)
    # --- 2) 单列：上到下 ---
    if ratio < 0.5:
        items_sorted = sorted(items, key=lambda t: (t[2], t[1]))  # cy, cx
        sorted_idx = np.array([t[0] for t in items_sorted], dtype=np.int64)
        return [points[i] for i in sorted_idx], sorted_idx

    # --- 3) 多行：先按 y 分行，再行内按 x ---
    items.sort(key=lambda t: t[2])  # 按 cy
    heights = np.array([max(t[3], t[4]) for t in items], dtype=np.float32)
    row_thr = max(5.0, float(np.median(heights) * row_alpha))

    rows = []
    row_cys = []

    for it in items:
        cy = it[2]
        if not rows:
            rows.append([it])
            row_cys.append(cy)
            continue

        # 放入最近行（在阈值内）
        ds = [abs(cy - rc) for rc in row_cys]
        k = int(np.argmin(ds))
        if ds[k] <= row_thr:
            rows[k].append(it)
            row_cys[k] = float(np.mean([x[2] for x in rows[k]]))
        else:
            rows.append([it])
            row_cys.append(cy)

    # 行顺序上->下；行内左->右
    order = np.argsort(row_cys)
    out = []
    for r in order:
        row = sorted(rows[int(r)], key=lambda t: (t[1], t[2]))  # cx, cy
        out.extend([t[0] for t in row])

    sorted_idx = np.array(out, dtype=np.int64)
    return [points[i] for i in sorted_idx], sorted_idx


def points_to_mask(shape_hw, points):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def rotate_upright(img, mask):

    ys, xs = np.where(mask > 0)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    h, w = mask.shape[:2]
    x_min, y_min = np.min(box, axis=0)
    x_max, y_max = np.max(box, axis=0)
    x_min = np.clip(x_min, 0, w)
    y_min = np.clip(y_min, 0, h)
    x_max = np.clip(x_max, 0, w)
    y_max = np.clip(y_max, 0, h)
    roi = img[y_min:y_max, x_min:x_max]
    mask_roi = mask[y_min:y_max, x_min:x_max]

    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    cx -= x_min
    cy -= y_min
    diag = int(np.sqrt(rw**2 + rh**2))  # 对角线长度
    new_w, new_h = diag, diag

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    M[0, 2] += (new_w - roi.shape[1]) / 2
    M[1, 2] += (new_h - roi.shape[0]) / 2
    # h, w = roi.shape[:2]
    img_r = cv2.warpAffine(roi, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    mask_r = cv2.warpAffine(
        mask_roi, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return img_r, mask_r


def smooth_1d(y, k):
    if k is None or k <= 1:
        return y
    k = int(k)
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(y.reshape(-1, 1), (k, 1), 0).ravel()


def contour_top_bottom(mask):
    ##
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    mask = cv2.erode(mask, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        raise ValueError("mask 里没找到轮廓")
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :]  # (N,2)

    xs = cnt[:, 0]
    ys = cnt[:, 1]
    x_min, x_max = int(xs.min()), int(xs.max())

    width = x_max - x_min + 1
    xi = xs.astype(np.int32) - x_min
    ys_f = ys.astype(np.float32)
    minY_arr = np.full(width, np.inf, dtype=np.float32)
    maxY_arr = np.full(width, -np.inf, dtype=np.float32)
    np.minimum.at(minY_arr, xi, ys_f)
    np.maximum.at(maxY_arr, xi, ys_f)
    minY_arr[minY_arr == np.inf] = np.nan
    maxY_arr[maxY_arr == -np.inf] = np.nan

    x_all = np.arange(x_min, x_max + 1, dtype=np.float32)
    top = np.stack([x_all, minY_arr - 10], axis=1)
    bot = np.stack([x_all, maxY_arr + 10], axis=1)

    def fill_nan(y):
        n = len(y)
        idx = np.where(~np.isnan(y))[0]
        if len(idx) < 2:
            # 极端情况：退化了
            y[np.isnan(y)] = 0
            return y
        return np.interp(np.arange(n), idx, y[idx]).astype(np.float32)

    top[:, 1] = fill_nan(top[:, 1])
    bot[:, 1] = fill_nan(bot[:, 1])
    return top, bot


def mask2roi(img: np.ndarray, points: np.array, smooth=21, sample_step=1, border_mode="replicate") -> List[np.ndarray]:
    rois = []
    for point in points:
        mask = points_to_mask(img.shape[:2], point)
        img_r, mask_r = rotate_upright(img, mask)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        # mask_r = cv2.erode(mask_r, kernel, iterations=2)
        # roi = cv2.bitwise_and(img_r, img_r, mask=mask_r)
        ###求矩形框，crop得到roi
        top, bot = contour_top_bottom(mask_r)

        # 平滑边界，减少展平后的“波纹”
        top[:, 1] = smooth_1d(top[:, 1], smooth)
        bot[:, 1] = smooth_1d(bot[:, 1], smooth)

        # 下采样加速
        top = top[::sample_step]
        bot = bot[::sample_step]

        W = len(top)
        H = int(np.max(bot[:, 1] - top[:, 1]) * 0.8)
        # 固定H
        # H = 256

        # 构建 remap
        map_x = np.empty((H, W), dtype=np.float32)
        map_y = np.empty((H, W), dtype=np.float32)

        a = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, np.newaxis]  # (H, 1)
        map_x = top[:, 0] + a * (bot[:, 0] - top[:, 0])  # (H, W)
        map_y = top[:, 1] + a * (bot[:, 1] - top[:, 1])

        bm = cv2.BORDER_REPLICATE if border_mode == "replicate" else cv2.BORDER_REFLECT_101
        flat = cv2.remap(img_r, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=bm)
        rois.append(flat)
        # cv2.imwrite("roi.jpg", flat)

    return rois


def Points_to_Mask(image_src, points, sort_by="y"):
    points_line, sorted_idx = sort_mask(image_src, points, 0.8)
    mask_rois = mask2roi(image_src, points_line)
    return mask_rois, sorted_idx


def rect_contains(rect, pt, include_border=True):
    x, y, w, h = rect
    px, py = pt
    if include_border:
        return (x <= px <= x + w) and (y <= py <= y + h)
    else:
        return (x < px < x + w) and (y < py < y + h)
