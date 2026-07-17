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
from dataclasses import dataclass
from typing import List, Optional, Tuple
from utils import vision_logger


_LOCAL_ROI_FALLBACK_WARNED = False


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


def points_to_local_mask(shape_hw, points, padding=0):
    """Rasterize one polygon only inside its clipped bounding rectangle."""
    image_h, image_w = shape_hw
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("polygon must contain at least three points")
    x, y, width, height = cv2.boundingRect(pts)
    x1 = max(0, min(image_w, x - padding))
    y1 = max(0, min(image_h, y - padding))
    x2 = max(0, min(image_w, x + width + padding))
    y2 = max(0, min(image_h, y + height + padding))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("polygon does not overlap the image")
    local_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    local_points = pts - np.array([x1, y1], dtype=np.int32)
    cv2.fillPoly(local_mask, [local_points], 255)
    if cv2.countNonZero(local_mask) == 0:
        raise ValueError("polygon produced an empty local mask")
    return local_mask, (x1, y1)


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
    # M: crop 坐标 -> img_r 坐标；offset: crop 坐标 -> 原图坐标。供 ROI 点反映射回原图用。
    return img_r, mask_r, M, (int(x_min), int(y_min))


def rotate_polygon_upright(img, points):
    """Rotate one polygon upright without allocating an image-sized mask."""
    local_mask, (base_x, base_y) = points_to_local_mask(
        img.shape[:2], points, padding=2
    )
    ys, xs = np.where(local_mask > 0)
    if len(xs) == 0:
        raise ValueError("polygon produced an empty local mask")
    global_pixels = np.stack(
        [xs + base_x, ys + base_y], axis=1
    ).astype(np.float32)
    rect = cv2.minAreaRect(global_pixels)
    global_box = cv2.boxPoints(rect).astype(np.int32)
    local_h, local_w = local_mask.shape
    image_x1, image_y1 = np.min(global_box, axis=0)
    image_x2, image_y2 = np.max(global_box, axis=0)
    image_x1 = int(np.clip(image_x1, 0, img.shape[1]))
    image_y1 = int(np.clip(image_y1, 0, img.shape[0]))
    image_x2 = int(np.clip(image_x2, 0, img.shape[1]))
    image_y2 = int(np.clip(image_y2, 0, img.shape[0]))
    overlap_x1 = max(image_x1, base_x)
    overlap_y1 = max(image_y1, base_y)
    overlap_x2 = min(image_x2, base_x + local_w)
    overlap_y2 = min(image_y2, base_y + local_h)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        raise ValueError("polygon has an invalid rotated bounding box")

    roi = img[image_y1:image_y2, image_x1:image_x2]
    mask_roi = np.zeros(roi.shape[:2], dtype=np.uint8)
    source_x1 = overlap_x1 - base_x
    source_y1 = overlap_y1 - base_y
    source_x2 = overlap_x2 - base_x
    source_y2 = overlap_y2 - base_y
    target_x1 = overlap_x1 - image_x1
    target_y1 = overlap_y1 - image_y1
    target_x2 = overlap_x2 - image_x1
    target_y2 = overlap_y2 - image_y1
    mask_roi[target_y1:target_y2, target_x1:target_x2] = local_mask[
        source_y1:source_y2, source_x1:source_x2
    ]

    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
    cx -= image_x1
    cy -= image_y1
    diagonal = int(np.sqrt(rw**2 + rh**2))
    if diagonal <= 0:
        raise ValueError("polygon has an invalid rotated size")
    matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    matrix[0, 2] += (diagonal - roi.shape[1]) / 2
    matrix[1, 2] += (diagonal - roi.shape[0]) / 2
    image_rotated = cv2.warpAffine(
        roi,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    mask_rotated = cv2.warpAffine(
        mask_roi,
        matrix,
        (diagonal, diagonal),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return image_rotated, mask_rotated, matrix, (image_x1, image_y1)


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


@dataclass
class RoiTransform:
    """展平 ROI 坐标 -> 原图坐标的反映射元数据（逐条线一份）。

    展平(fx,fy) --top/bot 线性插值--> img_r --Minv 逆仿射--> crop --+offset--> 原图
    """
    top: np.ndarray   # (W, 2) img_r 坐标系的上边界采样点
    bot: np.ndarray   # (W, 2) img_r 坐标系的下边界采样点
    H: int
    W: int
    Minv: np.ndarray  # (2,3) img_r -> crop 的逆仿射
    offset: Tuple[int, int]  # crop -> 原图 的 (x_min, y_min) 偏移


def map_roi_points_to_original(tf: RoiTransform, pts) -> np.ndarray:
    """把展平 ROI 坐标系下的点 (N,2) 反映射回原图像素坐标 (N,2)。"""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    cols = np.clip(np.round(pts[:, 0]).astype(np.int64), 0, tf.W - 1)
    a = (pts[:, 1] / max(tf.H - 1, 1)).reshape(-1, 1)
    top = tf.top[cols]
    bot = tf.bot[cols]
    imgr = top + a * (bot - top)  # (N,2) img_r 坐标
    ones = np.ones((imgr.shape[0], 1), dtype=np.float32)
    crop = np.hstack([imgr.astype(np.float32), ones]) @ tf.Minv.T  # (N,2) crop 坐标
    crop[:, 0] += tf.offset[0]
    crop[:, 1] += tf.offset[1]
    return crop


def _flatten_rotated_roi(
    img_r,
    mask_r,
    matrix,
    offset,
    smooth,
    sample_step,
    border_mode,
    return_map,
):
    top, bot = contour_top_bottom(mask_r)
    top[:, 1] = smooth_1d(top[:, 1], smooth)
    bot[:, 1] = smooth_1d(bot[:, 1], smooth)
    top = top[::sample_step]
    bot = bot[::sample_step]

    width = len(top)
    height = int(np.max(bot[:, 1] - top[:, 1]) * 0.8)
    if width <= 0 or height <= 0:
        raise ValueError("flattened ROI has an invalid size")

    alpha = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, np.newaxis]
    map_x = top[:, 0] + alpha * (bot[:, 0] - top[:, 0])
    map_y = top[:, 1] + alpha * (bot[:, 1] - top[:, 1])
    border = (
        cv2.BORDER_REPLICATE
        if border_mode == "replicate"
        else cv2.BORDER_REFLECT_101
    )
    flattened = cv2.remap(
        img_r,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=border,
    )
    if not return_map:
        return flattened, None
    transform = RoiTransform(
        top=np.asarray(top, dtype=np.float32),
        bot=np.asarray(bot, dtype=np.float32),
        H=height,
        W=width,
        Minv=cv2.invertAffineTransform(matrix),
        offset=offset,
    )
    return flattened, transform


def mask2roi(
    img: np.ndarray, points: np.array, smooth=21, sample_step=1, border_mode="replicate", return_maps=False
):
    rois = []
    transforms: List[Optional[RoiTransform]] = []
    for point in points:
        mask = points_to_mask(img.shape[:2], point)
        img_r, mask_r, matrix, offset = rotate_upright(img, mask)
        flattened, transform = _flatten_rotated_roi(
            img_r,
            mask_r,
            matrix,
            offset,
            smooth,
            sample_step,
            border_mode,
            return_maps,
        )
        rois.append(flattened)
        if return_maps:
            transforms.append(transform)

    if return_maps:
        return rois, transforms
    return rois


def _warn_local_roi_fallback_once(error):
    global _LOCAL_ROI_FALLBACK_WARNED
    if _LOCAL_ROI_FALLBACK_WARNED:
        return
    _LOCAL_ROI_FALLBACK_WARNED = True
    vision_logger.warning(
        "局部 ROI 展平失败，当前及后续异常 ROI 回退旧路径: {}", error
    )


def mask2roi_local(
    img: np.ndarray, points: np.array, smooth=21, sample_step=1, border_mode="replicate", return_maps=False
):
    rois = []
    transforms: List[Optional[RoiTransform]] = []
    for point in points:
        try:
            img_r, mask_r, matrix, offset = rotate_polygon_upright(
                img, point
            )
            flattened, transform = _flatten_rotated_roi(
                img_r,
                mask_r,
                matrix,
                offset,
                smooth,
                sample_step,
                border_mode,
                return_maps,
            )
        except (ValueError, cv2.error) as exc:
            _warn_local_roi_fallback_once(exc)
            legacy = mask2roi(
                img,
                [point],
                smooth=smooth,
                sample_step=sample_step,
                border_mode=border_mode,
                return_maps=return_maps,
            )
            if return_maps:
                legacy_rois, legacy_transforms = legacy
                flattened = legacy_rois[0]
                transform = legacy_transforms[0]
            else:
                flattened = legacy[0]
                transform = None
        rois.append(flattened)
        if return_maps:
            transforms.append(transform)

    if return_maps:
        return rois, transforms
    return rois


def Points_to_Mask(image_src, points, sort_by="y", return_maps=False):
    points_line, sorted_idx = sort_mask(image_src, points, 0.8)
    if return_maps:
        mask_rois, transforms = mask2roi_local(
            image_src, points_line, return_maps=True
        )
        return mask_rois, sorted_idx, transforms
    mask_rois = mask2roi_local(image_src, points_line)
    return mask_rois, sorted_idx


def Points_to_Mask_legacy(image_src, points, sort_by="y", return_maps=False):
    points_line, sorted_idx = sort_mask(image_src, points, 0.8)
    if return_maps:
        mask_rois, transforms = mask2roi(
            image_src, points_line, return_maps=True
        )
        return mask_rois, sorted_idx, transforms
    mask_rois = mask2roi(image_src, points_line)
    return mask_rois, sorted_idx


def rotated_box_overlap(poly1, poly2) -> float:
    """两多边形最小外接旋转矩形的交集面积 / 较小矩形面积（IoS），范围 [0,1]。

    用 IoS 而非 IoU：同一线标的重复检测常是"全长框 + 半截框"，半截框几乎
    完全落在全长框内，IoS 接近 1 而 IoU 只有长度占比；相邻倾斜线标的
    旋转矩形几乎不相交，IoS 接近 0，区分度好。
    """
    r1 = cv2.boxPoints(cv2.minAreaRect(np.asarray(poly1, dtype=np.float32).reshape(-1, 2)))
    r2 = cv2.boxPoints(cv2.minAreaRect(np.asarray(poly2, dtype=np.float32).reshape(-1, 2)))
    inter, _ = cv2.intersectConvexConvex(r1, r2)
    if inter <= 0:
        return 0.0
    smaller = min(cv2.contourArea(r1), cv2.contourArea(r2))
    return float(inter / smaller) if smaller > 0 else 0.0


def dedup_overlapping_polygons(polygons, scores, class_ids, overlap_thresh: float):
    """同类实例间按旋转框 IoS 去重，保留高置信度者；返回升序的保留索引。

    YOLO 轴对齐 NMS（宽松阈值）抑制不掉同一线标上的重复框，在此基于
    mask 多边形做二次去重。overlap_thresh >= 1 时等效关闭。
    """
    order = sorted(range(len(polygons)), key=lambda i: scores[i], reverse=True)
    keep = []
    for i in order:
        is_dup = any(
            class_ids[i] == class_ids[j] and rotated_box_overlap(polygons[i], polygons[j]) > overlap_thresh
            for j in keep
        )
        if not is_dup:
            keep.append(i)
    return sorted(keep)


def rect_contains(rect, pt, include_border=True):
    x, y, w, h = rect
    px, py = pt
    if include_border:
        return (x <= px <= x + w) and (y <= py <= y + h)
    else:
        return (x < px < x + w) and (y < py < y + h)


def polygon_contains(poly_pts, pt, include_border=True):
    """点 pt 是否落在四边形 poly_pts 内（顺时针四角，像素坐标）。

    poly_pts 接受扁平 [x1,y1,...,x4,y4] 或 [(x,y),...]；用 cv2.pointPolygonTest
    判含：返回 +1 内、0 边界、-1 外。include_border 控制边界点是否计入，
    与 rect_contains 的 include_border 语义对称。
    """
    poly = np.asarray(poly_pts, dtype=np.float32).reshape(-1, 2)
    dist = cv2.pointPolygonTest(poly, (float(pt[0]), float(pt[1])), False)
    if include_border:
        return dist >= 0
    return dist > 0


def polygon_overlap_ratio(subject_poly, roi_poly) -> float:
    """subject_poly 落在 roi_poly 内的面积占比，范围 [0, 1]。"""
    subject = np.asarray(subject_poly, dtype=np.float32).reshape(-1, 2)
    roi = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
    subject_area = cv2.contourArea(subject)
    if subject_area <= 0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(subject, roi)
    if inter_area <= 0:
        return 0.0
    return min(1.0, float(inter_area) / float(subject_area))
