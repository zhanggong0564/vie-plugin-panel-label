"""Polygon containment, overlap, and detection de-duplication helpers."""

import cv2
import numpy as np


def rotated_box_overlap(poly1, poly2) -> float:
    """Return intersection-over-smaller-area for two rotated boxes."""
    rectangle1 = cv2.minAreaRect(
        np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    )
    rectangle2 = cv2.minAreaRect(
        np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    )
    box1 = cv2.boxPoints(rectangle1)
    box2 = cv2.boxPoints(rectangle2)
    intersection, _ = cv2.intersectConvexConvex(box1, box2)
    if intersection <= 0:
        return 0.0
    smaller_area = min(cv2.contourArea(box1), cv2.contourArea(box2))
    return float(intersection / smaller_area) if smaller_area > 0 else 0.0


def polygon_overlap(poly1, poly2) -> float:
    """Return intersection-over-smaller-area for the actual polygons."""
    polygon1 = np.asarray(poly1, dtype=np.float32).reshape(-1, 2)
    polygon2 = np.asarray(poly2, dtype=np.float32).reshape(-1, 2)
    intersection, _ = cv2.intersectConvexConvex(polygon1, polygon2)
    if intersection <= 0:
        return 0.0
    smaller_area = min(
        cv2.contourArea(polygon1),
        cv2.contourArea(polygon2),
    )
    return float(intersection / smaller_area) if smaller_area > 0 else 0.0


def dedup_overlapping_polygons(
    polygons,
    scores,
    class_ids,
    overlap_thresh: float,
):
    """Keep the highest-scoring polygon from each overlapping class group."""
    order = sorted(
        range(len(polygons)),
        key=lambda index: scores[index],
        reverse=True,
    )
    keep = []
    for index in order:
        is_duplicate = any(
            class_ids[index] == class_ids[kept_index]
            and rotated_box_overlap(
                polygons[index],
                polygons[kept_index],
            )
            > overlap_thresh
            and polygon_overlap(
                polygons[index],
                polygons[kept_index],
            )
            > overlap_thresh
            for kept_index in keep
        )
        if not is_duplicate:
            keep.append(index)
    return sorted(keep)


def rect_contains(rect, point, include_border=True):
    x, y, width, height = rect
    point_x, point_y = point
    if include_border:
        return (
            x <= point_x <= x + width
            and y <= point_y <= y + height
        )
    return x < point_x < x + width and y < point_y < y + height


def polygon_contains(poly_pts, point, include_border=True):
    """Return whether a point is inside a polygon."""
    polygon = np.asarray(poly_pts, dtype=np.float32).reshape(-1, 2)
    distance = cv2.pointPolygonTest(
        polygon,
        (float(point[0]), float(point[1])),
        False,
    )
    return distance >= 0 if include_border else distance > 0


def polygon_overlap_ratio(subject_poly, roi_poly) -> float:
    """Return the fraction of the subject polygon inside the ROI."""
    subject = np.asarray(subject_poly, dtype=np.float32).reshape(-1, 2)
    roi = np.asarray(roi_poly, dtype=np.float32).reshape(-1, 2)
    subject_area = cv2.contourArea(subject)
    if subject_area <= 0:
        return 0.0
    intersection_area, _ = cv2.intersectConvexConvex(subject, roi)
    if intersection_area <= 0:
        return 0.0
    return min(1.0, float(intersection_area) / float(subject_area))
