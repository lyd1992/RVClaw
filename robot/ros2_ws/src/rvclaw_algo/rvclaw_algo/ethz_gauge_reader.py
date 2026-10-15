"""Pointer-gauge reading with the ETH Zurich ``analog_gauge_reader`` models.

Provenance
==========
The three networks are the published weights of

    https://github.com/ethz-asl/analog_gauge_reader          (MIT)
    "Under Pressure: Learning-Based Analog Gauge Reading In The Wild"

They ship here as ONNX because the board cannot run the upstream Python stack:
the K3 is riscv64 with Python 3.14, while upstream pins ``torch==2.0.0``,
``mmcv==2.0.0`` (an x86_64 cp38 wheel only) and ``mmocr==1.0.0``.
``tools/export_ethz_onnx_20260918.py`` and ``tools/export_ethz_keypoint_onnx.py``
build the graphs; ``output/ethz-gauge-onnx/MODELS.md`` records their sha256 and
the eager-vs-ONNX agreement.

Pipeline (upstream ``pipeline.process_image``)
==============================================
    gauge-face box -> square crop -> 448x448
    key-point heat maps (3 channels: start / notch / end)
    ellipse through the notch points
    polar-angle zero point from start + end
    OCR the dial numbers
    needle mask -> line
    angle -> value least squares over (angle, number), RANSAC
    reading = fit(needle angle)

Deviations from upstream -- all deliberate, all listed here
===========================================================
1.  ``scipy.odr`` is absent on the board.  The needle line is the same
    orthogonal-distance fit, taken from the principal axis of the mask pixel
    cloud (total least squares) instead of an ODR iteration.
2.  ``sklearn.cluster.MeanShift`` is absent.  ``_mean_shift_modes`` reimplements
    the two phases upstream relies on -- mode seeking from every point, then
    densest-first claiming of everything within one bandwidth -- using a grid
    index, so each iteration touches only neighbouring cells.
3.  ``sklearn.cluster.KMeans(n_clusters=1)`` is absent.  One cluster is exactly
    the centroid, so ``_centroid`` is used.
4.  ``sklearn.linear_model.RANSACRegressor`` is absent.  ``_ransac_line`` keeps
    the same MAD residual threshold and inlier refit; a dial carries so few
    numbers that every sample pair can be enumerated, which is deterministic.
5.  The needle-line/ellipse intersection is evaluated on an explicit
    ``(point, direction)`` pair rather than as ``y = m * x + c``.  Upstream's
    form degenerates for a needle close to vertical: ``x`` becomes
    ill-conditioned and ``m`` explodes, so the two analytic roots collapse.  The
    intersection is the same one, just parameterised so every needle direction
    stays well conditioned.
6.  ``MMOCRInferencer(det='DB_r18', rec='ABINet')`` is replaced by this project's
    in-process PP-OCRv5 (``rvclaw_algo.ppocr_onnx``), which yields the same
    (polygon, text, score) triples.  Upstream always reads the rectified, rotated
    view; here the plain crop is tried first and the rectified view is the
    fallback, because upstream's rotation is global while dial digits are each
    oriented individually -- rectifying can turn readable digits upside down.
7.  The detected box is clipped to the image before slicing.  Upstream slices
    with the raw box, and a negative index silently yields an empty crop
    whenever the gauge touches the border.
8.  OCR polygons are mapped back through the *inverted* transform matrix.
    Upstream re-derives the rotation by hand with the opposite sign convention
    to ``cv2.warpAffine``, which displaces every polygon by twice the rotation
    angle (invisible only when that angle is zero).
9.  ``UNIT_LIST`` gains the SI pressure units (Pa / kPa / hPa / inHg / kgf/cm2).
    Upstream's list knows ``bar`` and ``MPa`` but not ``kPa``, so a dial marked
    kPa would come back with no unit at all.
10. The polar-angle branch cut is repaired when the start/end key-point head
    misfires.  Upstream takes the cut from those key points unconditionally, and
    a cut landing *inside* the run of numbers folds the scale in half -- the fit
    stays confident and returns a wrong value.  ``_dial_zero_angle`` keeps the
    key-point cut whenever it already lies in a gap, and otherwise moves it to
    the middle of the widest gap between the numbers actually read.
11. A needle that lands outside the labelled part of the scale is reported with
    ``status="out_of_range"`` instead of a plain ``stable`` value, so an
    extrapolated reading cannot masquerade as a measured one.
12. Every result carries ``screen_angle_deg`` beside upstream's
    ``needle_angle_deg``.  Upstream's polar angle is measured in the fitted
    ellipse's frame, which for a near-circular dial is an arbitrary rotation of
    the photo's -- correct for the angle->value fit, meaningless on a screen.
    The screen angle (0 deg at 3 o'clock, growing clockwise as displayed) is the
    one an operator can check against the image, and it is what the studio page
    prints beside the value.
"""

from __future__ import annotations

import math
import os
import threading
from itertools import combinations
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

__all__ = [
    "EthzGaugeReader",
    "OCRReading",
    "UNIT_LIST",
    "DEFAULT_MODEL_DIR",
]

# --- constants, mirroring upstream `pipeline.py` ---------------------------
YOLO_INPUT = 640
GAUGE_RESOLUTION = (448, 448)
DET_CONFIDENCE = 0.25
DET_IOU = 0.45
KEYPOINT_THRESHOLDS = tuple(
    float(value.strip())
    for value in os.getenv(
        "RVCLAW_GAUGE_KEYPOINT_THRESHOLDS", "0.60,0.50,0.40,0.35,0.30"
    ).split(",")
    if value.strip()
)
if not KEYPOINT_THRESHOLDS:
    KEYPOINT_THRESHOLDS = (0.6,)
# Kept for callers and diagnostics that used the old single-value constant.
KEYPOINT_THRESHOLD = KEYPOINT_THRESHOLDS[0]
MEANSHIFT_BANDWIDTH = 20.0
OCR_THRESHOLD = 0.7
# Upstream's serial-number heuristics: a dial label is neither implausibly large
# nor a two-plus digit number that is not a round ten.
MAX_LABEL = 10_000
MAX_PLAIN_LABEL = 100
# Rotate the rectified dial so the zero point sits at the bottom before OCR.
OCR_DESIRED_ANGLE = 90.0
# Upstream `ocr_reading.UNIT_LIST`, plus the SI pressure units a dial is far more
# likely to carry than the Fahrenheit symbols (deviation 9).
UNIT_LIST = ("bar", "mbar", "millibars", "Pa", "kPa", "hPa", "MPa", "psi", "inHg",
             "kg/cm2", "kgf/cm2", "C", "\u00b0C", "F", "\u00b0F", "%")

# Beside the other vision models on the K3; RVCLAW_GAUGE_MODEL_DIR overrides it.
DEFAULT_MODEL_DIR = os.path.expanduser("~/.cache/models/vision/analog_gauge_reader")

MODEL_FILES = {
    "detection": "gauge_detection_model.onnx",
    "segmentation": "segmentation_model.onnx",
    "keypoint": "key_point_model.onnx",
}


# ---------------------------------------------------------------------------
# OCR readings -- port of upstream `ocr/ocr_reading.py`
# ---------------------------------------------------------------------------


class OCRReading:
    """One recognised text box on the dial."""

    def __init__(self, polygon: Any, reading: str, confidence: float):
        self.polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        self.reading = (reading or "").strip()
        self.confidence = float(confidence)
        self.number: Optional[float] = None
        if self.is_number():
            self.number = float(self.reading)
        self.center: Tuple[float, float] = self._get_centroid()
        self.theta: Optional[float] = None

    def _get_centroid(self) -> Tuple[float, float]:
        return (float(np.mean(self.polygon[:, 0])), float(np.mean(self.polygon[:, 1])))

    def is_number(self) -> bool:
        try:
            float(self.reading)
            return True
        except ValueError:
            return False

    def is_unit(self) -> bool:
        return self.reading.lower() in [unit.lower() for unit in UNIT_LIST]

    def set_polygon(self, polygon: Any) -> None:
        self.polygon = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        self.center = self._get_centroid()

    def set_theta(self, theta: float) -> None:
        self.theta = theta

    def get_bounding_box(self):
        return (
            float(np.min(self.polygon[:, 0])),
            float(np.min(self.polygon[:, 1])),
            float(np.max(self.polygon[:, 0])),
            float(np.max(self.polygon[:, 1])),
        )


# ---------------------------------------------------------------------------
# ellipse geometry -- port of upstream `geometry/ellipse.py`
# ---------------------------------------------------------------------------


def _fit_ellipse(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Direct least squares ellipse fit (Halir & Flusser, as upstream)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    d1 = np.vstack([x ** 2, x * y, y ** 2]).T
    d2 = np.vstack([x, y, np.ones(len(x))]).T
    s1 = d1.T @ d1
    s2 = d1.T @ d2
    s3 = d2.T @ d2
    t = -np.linalg.inv(s3) @ s2.T
    m = s1 + s2 @ t
    c = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    m = np.linalg.inv(c) @ m
    _eigval, eigvec = np.linalg.eig(m)
    con = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, t @ ak)).ravel()


def _cart_to_pol(coeffs: np.ndarray):
    """Conic coefficients -> ``(x0, y0, ap, bp, phi)``."""
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b ** 2 - a * c
    if den > 0:
        raise ValueError("coeffs do not represent an ellipse: b^2 - 4ac must be negative")

    x0, y0 = (c * d - b * f) / den, (a * f - b * d) / den
    num = 2 * (a * f ** 2 + c * d ** 2 + g * b ** 2 - 2 * b * d * f - a * c * g)
    fac = np.sqrt((a - c) ** 2 + 4 * b ** 2)
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    r = (bp / ap) ** 2
    if r > 1:
        r = 1 / r

    if b == 0:
        phi = 0 if a < c else np.pi / 2
    else:
        phi = np.arctan((2.0 * b) / (a - c)) / 2
        if a > c:
            phi += np.pi / 2
    if not width_gt_height:
        phi += np.pi / 2
    phi = phi % np.pi

    return float(x0), float(y0), float(ap), float(bp), float(phi)


def _polar_angle(point: Sequence[float], params) -> float:
    """Parametric angle of ``point`` on the ellipse, in [0, 2*pi)."""
    x0, y0, ap, bp, phi = params
    x_shift = float(point[0]) - x0
    y_shift = float(point[1]) - y0
    cos_p, sin_p = math.cos(-phi), math.sin(-phi)
    x_rot = cos_p * x_shift - sin_p * y_shift
    y_rot = sin_p * x_shift + cos_p * y_shift
    theta = math.atan2(ap * y_rot, bp * x_rot)
    if theta < 0:
        theta += 2 * math.pi
    return theta


def _point_from_angle(theta: float, params):
    x0, y0, ap, bp, phi = params
    x_p, y_p = ap * math.cos(theta), bp * math.sin(theta)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    return (cos_p * x_p - sin_p * y_p + x0, sin_p * x_p + cos_p * y_p + y0)


def _theta_middle(theta_1: float, theta_2: float) -> float:
    """The angle between two angles, on the short way round."""
    candidate_1 = (theta_2 + theta_1) / 2
    if theta_2 + theta_1 > 2 * math.pi:
        candidate_2 = (theta_2 + theta_1 - 2 * math.pi) / 2
    else:
        candidate_2 = (theta_2 + theta_1 + 2 * math.pi) / 2
    distance_1 = min(abs(candidate_1 - theta_1), abs(candidate_1 - theta_2))
    distance_2 = min(abs(candidate_2 - theta_1), abs(candidate_2 - theta_2))
    return candidate_1 if distance_1 < distance_2 else candidate_2


def _dial_zero_angle(label_thetas: Sequence[float], keypoint_zero: float):
    """Place the polar-angle branch cut in the empty part of the dial.

    Upstream sets the cut from the start/end key-point head, via
    ``get_theta_middle``.  That head is the weak link: on a dial whose zero
    marks it has not learned, it fires on noise (its two channels can peak on the
    same pixel), and a cut placed *inside* the run of numbers folds the scale in
    half -- every label after the cut gets a 2*pi offset and the linear fit
    becomes meaningless while still returning a confident number.

    Keep the key-point cut whenever it already sits in a gap, and otherwise move
    it to the middle of the widest gap between the recognised numbers.  That is
    the same geometric intent -- a cut where the dial has no scale -- derived
    from the labels actually visible in the photograph.
    """
    angles = sorted(float(t) % (2 * math.pi) for t in label_thetas)
    count = len(angles)
    if count < 2:
        return float(keypoint_zero), "keypoints"

    widest, gap_start, gap_end = 0.0, 0.0, 0.0
    for index in range(count):
        start = angles[index]
        end = angles[(index + 1) % count] + (2 * math.pi if index == count - 1 else 0.0)
        if end - start > widest:
            widest, gap_start, gap_end = end - start, start, end

    normalised = float(keypoint_zero) % (2 * math.pi)
    if widest > 0 and (normalised - gap_start) % (2 * math.pi) <= widest:
        return normalised, "keypoints"
    return (gap_start + gap_end) / 2.0 % (2 * math.pi), "widest_gap"


def _ellipse_error(points: np.ndarray, params) -> float:
    """Mean distance from the fitted points to the ellipse (upstream metric)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return float("nan")
    total = 0.0
    for point in points:
        theta = _polar_angle(point, params)
        proj = np.asarray(_point_from_angle(theta, params))
        total += float(np.linalg.norm(proj - point))
    return total / points.shape[0]


def _line_ellipse_intersections(point: Sequence[float], direction: Sequence[float], params):
    """Intersect a line with the ellipse; returns 0, 1 or 2 image-space points.

    Works in the ellipse's own frame, where the ellipse is the unit circle, so
    the quadratic stays well conditioned for every needle direction.  This
    replaces upstream's ``y = m * x + c`` form (deviation 5).
    """
    x0, y0, ap, bp, phi = params
    cos_m, sin_m = math.cos(-phi), math.sin(-phi)
    cos_p, sin_p = math.cos(phi), math.sin(phi)

    def to_frame(dx, dy):
        rx = cos_m * dx - sin_m * dy
        ry = sin_m * dx + cos_m * dy
        return rx / ap, ry / bp

    ax, ay = to_frame(point[0] - x0, point[1] - y0)
    dx2, dy2 = to_frame(direction[0], direction[1])

    b_term = dx2 * dx2 + dy2 * dy2
    if b_term < 1e-18:
        return []
    a_term = ax * dx2 + ay * dy2
    c_term = ax * ax + ay * ay - 1.0
    disc = a_term * a_term - b_term * c_term
    if disc < 0:
        return []

    root = math.sqrt(disc)
    results = []
    for t in ((-a_term - root) / b_term, (-a_term + root) / b_term):
        u, v = ax + t * dx2, ay + t * dy2
        rx, ry = u * ap, v * bp
        results.append((cos_p * rx - sin_p * ry + x0, sin_p * rx + cos_p * ry + y0))
    return results


# ---------------------------------------------------------------------------
# needle line -- total least squares (deviation 1)
# ---------------------------------------------------------------------------


def _needle_line(x: np.ndarray, y: np.ndarray):
    """Orthogonal-distance line fit of the needle mask pixels.

    Returns ``(origin, direction, rms_residual)`` with ``direction`` a unit
    vector; ``origin`` is the pixel centroid.
    """
    points = np.column_stack([np.asarray(x, dtype=np.float64).ravel(),
                              np.asarray(y, dtype=np.float64).ravel()])
    if points.shape[0] < 2:
        raise ValueError("needle mask has fewer than 2 pixels")
    origin = points.mean(axis=0)
    centred = points - origin
    covariance = (centred.T @ centred) / points.shape[0]
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("needle mask is degenerate")
    direction = direction / norm
    # Residual spread orthogonal to the line: the smaller eigenvalue.
    residual = math.sqrt(max(0.0, float(values[0])))
    return origin, direction, residual


def _needle_extent(x: np.ndarray, y: np.ndarray, origin, direction):
    """Parameter range of the mask along the fitted line."""
    points = np.column_stack([np.asarray(x, float).ravel(), np.asarray(y, float).ravel()])
    projections = (points - origin) @ direction
    return float(projections.min()), float(projections.max())


def _choose_ellipse_point(candidates, origin, direction, t_min, t_max):
    """Pick the needle/ellipse intersection on the needle's own side.

    Upstream prefers the intersection lying between the two ends of the needle
    segment and otherwise falls back to the one nearest an end; the same rule is
    applied here on the along-line parameter ``t``.
    """
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    params = [float((np.asarray(p) - origin) @ direction) for p in candidates]
    inside = [i for i, t in enumerate(params) if t_min - 1e-6 <= t <= t_max + 1e-6]
    if len(inside) == 1:
        return candidates[inside[0]]
    ends = [origin + t_min * direction, origin + t_max * direction]
    best, best_distance = None, None
    for candidate in candidates:
        for end in ends:
            distance = float(np.linalg.norm(np.asarray(candidate) - end))
            if best_distance is None or distance < best_distance:
                best, best_distance = candidate, distance
    return best


# ---------------------------------------------------------------------------
# key-point extraction -- port of upstream `key_point_extraction.py`
# ---------------------------------------------------------------------------


def _mean_shift_modes(points: np.ndarray, bandwidth: float,
                     max_iter: int = 300, tolerance: float = 1e-3) -> np.ndarray:
    """``sklearn.cluster.MeanShift(bandwidth, bin_seeding=False)`` in numpy.

    Phase 1 seeds from every point and walks each one uphill to a mode of the
    empirical density (flat kernel of the given bandwidth), recording how many
    seeds land on each mode.  Phase 2 walks the modes densest first; each claims
    every still-unclaimed point within one bandwidth, and a cluster centre is the
    mean of the points it claimed.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return pts.copy()

    keys = np.floor(pts / bandwidth).astype(np.int64)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for index, (kx, ky) in enumerate(keys):
        buckets.setdefault((int(kx), int(ky)), []).append(index)
    index_of = {key: np.asarray(value, dtype=np.int64) for key, value in buckets.items()}

    def neighbours(point) -> Optional[np.ndarray]:
        kx = int(math.floor(point[0] / bandwidth))
        ky = int(math.floor(point[1] / bandwidth))
        chunks = []
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                chunk = index_of.get((kx + ddx, ky + ddy))
                if chunk is not None:
                    chunks.append(chunk)
        return np.concatenate(chunks) if chunks else None

    modes: Dict[Tuple[float, float], int] = {}
    for seed in pts:
        centre = seed
        for _ in range(max_iter):
            near = neighbours(centre)
            if near is None or near.size == 0:
                break
            shifted = pts[near].mean(axis=0)
            if math.hypot(float(shifted[0] - centre[0]), float(shifted[1] - centre[1])) < tolerance:
                centre = shifted
                break
            centre = shifted
        key = (round(float(centre[0]), 3), round(float(centre[1]), 3))
        modes[key] = modes.get(key, 0) + 1

    claimed = np.zeros(pts.shape[0], dtype=bool)
    centres: List[np.ndarray] = []
    for key, _intensity in sorted(modes.items(), key=lambda kv: kv[1], reverse=True):
        near = neighbours(np.asarray(key, dtype=np.float64))
        if near is None:
            continue
        fresh = near[~claimed[near]]
        if fresh.size == 0:
            continue
        claimed[fresh] = True
        centres.append(pts[fresh].mean(axis=0))
    return np.asarray(centres, dtype=np.float64) if centres else np.empty((0, 2))


def _centroid(points: np.ndarray) -> np.ndarray:
    """``KMeans(n_clusters=1)`` returns exactly the centroid (deviation 3)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts.mean(axis=0, keepdims=True)


def _extract_key_points(heatmaps: np.ndarray, threshold: float, bandwidth: float):
    """Split the 3 heat-map channels into (notch points, start, end)."""
    notch = np.empty((0, 2), dtype=np.float64)
    start = np.empty((0, 2), dtype=np.float64)
    end = np.empty((0, 2), dtype=np.float64)
    for index in range(heatmaps.shape[0]):
        channel = np.asarray(heatmaps[index], dtype=np.float64)
        peak = float(channel.max())
        if peak <= 0:
            continue
        normalised = channel / peak
        coords = np.argwhere(normalised > threshold)
        if coords.shape[0] == 0:
            continue
        # np.argwhere yields (row, col); upstream swaps to (x, y).
        swapped = coords[:, ::-1].astype(np.float64)
        if index == 1:
            notch = _mean_shift_modes(swapped, bandwidth)
        elif index == 0:
            start = _centroid(swapped)
        else:
            end = _centroid(swapped)
    return notch, start, end


# ---------------------------------------------------------------------------
# RANSAC angle->value fit (deviation 4)
# ---------------------------------------------------------------------------


def _ransac_line(x: np.ndarray, y: np.ndarray, max_trials: int = 100, seed: int = 0):
    """``RANSACRegressor(loss='absolute_error')`` analogue, plus the inlier mask.

    sklearn's default residual threshold is the median absolute deviation of the
    targets, and the returned model is refitted on the inliers; both are kept.
    For the handful of numbers a dial carries, every sample pair is enumerated
    (up to ``max_trials``) rather than sampled, which makes the result stable.
    """
    xs = np.asarray(x, dtype=np.float64).ravel()
    ys = np.asarray(y, dtype=np.float64).ravel()
    count = xs.size
    if count < 2:
        raise ValueError("need at least 2 dial numbers to fit a line")
    threshold = float(np.median(np.abs(ys - np.median(ys))))
    if not np.isfinite(threshold) or threshold <= 0:
        threshold = float(np.std(ys)) or 1.0

    pairs = list(combinations(range(count), 2))
    if len(pairs) > max_trials:
        rng = np.random.default_rng(seed)
        pairs = [tuple(int(v) for v in rng.choice(count, 2, replace=False))
                 for _ in range(max_trials)]

    best_inliers, best_coeffs = -1, None
    for i, j in pairs:
        if xs[i] == xs[j]:
            continue
        slope = (ys[j] - ys[i]) / (xs[j] - xs[i])
        intercept = ys[i] - slope * xs[i]
        inliers = int(np.count_nonzero(np.abs(slope * xs + intercept - ys) <= threshold))
        if inliers > best_inliers:
            best_inliers, best_coeffs = inliers, (slope, intercept)
    if best_coeffs is None:
        raise ValueError("all dial numbers share one angle; cannot fit")

    slope, intercept = best_coeffs
    mask = np.abs(slope * xs + intercept - ys) <= threshold
    if int(mask.sum()) >= 2:
        coeffs = np.polyfit(xs[mask], ys[mask], 1)
    else:
        coeffs = np.asarray(best_coeffs, dtype=np.float64)
    return coeffs, mask


# ---------------------------------------------------------------------------
# warp / rotate helpers used by the OCR path
# ---------------------------------------------------------------------------


def _move_point_resize(point, original_resolution, resized_resolution):
    """Upstream ``pipeline.move_point_resize``."""
    return (float(point[0]) * resized_resolution[0] / original_resolution[0],
            float(point[1]) * resized_resolution[1] / original_resolution[1])


def _rescale_ellipse(params, original_resolution, resized_resolution):
    """Upstream ``pipeline.rescale_ellipse_resize``."""
    x0, y0, ap, bp, phi = params
    x0_new, y0_new = _move_point_resize((x0, y0), original_resolution,
                                        resized_resolution)
    factor = resized_resolution[0] / original_resolution[0]
    return x0_new, y0_new, ap * factor, bp * factor, phi


def _warp_ellipse_to_circle(image: np.ndarray, params):
    """Perspective-map the fitted ellipse onto a circle (upstream ``warp_ellipse``).

    ``params`` must be in *this image's* resolution.  The ellipse is expressed in
    the 448x448 frame the key-point model works in, so callers have to rescale it
    through ``_rescale_ellipse`` before warping the full-resolution crop;
    handing the 448-frame numbers to a 1083-pixel image throws the dial off the
    canvas entirely.
    """
    height, width = image.shape[:2]
    x0, y0, ap, bp, phi = params
    major_axis, minor_axis = ap, bp

    source = np.array([[-major_axis / 2, -minor_axis / 2],
                       [major_axis / 2, -minor_axis / 2],
                       [major_axis / 2, minor_axis / 2],
                       [-major_axis / 2, minor_axis / 2]], dtype=np.float32)
    square = max(major_axis, minor_axis)
    destination = np.array([[-square / 2, -square / 2],
                            [square / 2, -square / 2],
                            [square / 2, square / 2],
                            [-square / 2, square / 2]], dtype=np.float32)

    rotation = np.array([[math.cos(phi), math.sin(phi)],
                         [-math.sin(phi), math.cos(phi)]])
    source = source @ rotation
    destination = destination @ rotation
    source[:, 0] += x0
    source[:, 1] += y0
    destination[:, 0] += x0
    destination[:, 1] += y0

    matrix = cv2.getPerspectiveTransform(source.astype(np.float32),
                                         destination.astype(np.float32))
    warped = cv2.warpPerspective(image, matrix, (width, height))
    return warped, matrix


def _transform_point(point, matrix: np.ndarray):
    """Apply a 3x3 perspective transform to one point."""
    packed = np.array([[point]], dtype=np.float32)
    return tuple(float(v) for v in cv2.perspectiveTransform(packed, matrix)[0][0])


def _transform_point_inverse(point, matrix: np.ndarray):
    """Apply the inverse of a 3x3 perspective transform to one point."""
    return _transform_point(point, np.linalg.inv(matrix))


def _transform_polygon_inverse(polygon: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(matrix)
    packed = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(packed, inverse).reshape(-1, 2)


def _rotate_about(image: np.ndarray, centre, angle_deg: float):
    """Rotate about a point; returns the image and the affine matrix used."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                     float(angle_deg), 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height))
    return rotated, matrix


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.hstack([pts, np.ones((pts.shape[0], 1))])
    return (matrix @ homogeneous.T).T


# ---------------------------------------------------------------------------
# YOLO ONNX inference (ultralytics conventions, reimplemented in numpy)
# ---------------------------------------------------------------------------


def _letterbox(image: np.ndarray, size: int = YOLO_INPUT, pad: int = 114):
    height, width = image.shape[:2]
    ratio = min(size / width, size / height)
    new_w, new_h = int(round(width * ratio)), int(round(height * ratio))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(pad, pad, pad))
    return padded, ratio, left, top


def _blob(padded: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0:
        best = order[0]
        keep.append(int(best))
        xx1 = np.maximum(x1[best], x1[order[1:]])
        yy1 = np.maximum(y1[best], y1[order[1:]])
        xx2 = np.minimum(x2[best], x2[order[1:]])
        yy2 = np.minimum(y2[best], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[best] + areas[order[1:]] - inter
        iou = inter / np.maximum(1e-9, union)
        order = order[1:][iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def _decode_boxes(prediction: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """xywh (input pixels) -> xyxy."""
    xywh = prediction[:4, keep].T
    boxes = np.empty((keep.size, 4), dtype=np.float64)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2
    return boxes


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def _is_dial_number(reading: "OCRReading") -> bool:
    """Upstream's filter for "this reading is a scale label".

    Confident parseable numbers, excluding the implausible magnitudes and the
    two-plus digit non-round numbers that are really a serial or a model code
    printed on the case.
    """
    if not reading.is_number() or reading.confidence <= OCR_THRESHOLD:
        return False
    magnitude = abs(reading.number)
    if magnitude > MAX_LABEL:
        return False
    if magnitude > MAX_PLAIN_LABEL and reading.number % 10 != 0:
        return False
    return True


def _count_dial_numbers(readings: Sequence["OCRReading"]) -> int:
    return sum(1 for reading in readings if _is_dial_number(reading))


# ---------------------------------------------------------------------------
# public reader
# ---------------------------------------------------------------------------


def configured_scale_value(screen_angle_deg: float, calibration: Mapping[str, Any]):
    """Map a detected screen angle through a known instrument scale.

    ``screen_angle_deg`` uses the reader's documented convention: zero at
    3 o'clock and increasing clockwise.  A configured scale is safer than
    fitting from two accidentally recognised dial labels, and is the normal
    deployment mode for a known inspection point.
    """

    def finite(name: str, default=None) -> float:
        raw = calibration.get(name, default)
        if raw is None:
            raise ValueError(f"pointer calibration requires {name}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"pointer calibration {name} must be finite")
        return value

    zero = finite("zero_angle_deg") % 360.0
    sweep = finite("sweep_deg", 360.0)
    minimum = finite("min_value")
    maximum = finite("max_value")
    if not 0.0 < sweep <= 360.0:
        raise ValueError("pointer calibration sweep_deg must be in (0, 360]")
    if maximum <= minimum:
        raise ValueError("pointer calibration max_value must exceed min_value")
    direction = str(calibration.get("direction", "clockwise")).strip().lower()
    if direction not in {"clockwise", "counterclockwise"}:
        raise ValueError("pointer calibration direction must be clockwise or counterclockwise")
    angle = float(screen_angle_deg) % 360.0
    offset = (angle - zero) % 360.0 if direction == "clockwise" else (zero - angle) % 360.0
    tolerance = max(0.0, finite("out_of_scale_tolerance_deg", 4.0))
    if sweep < 360.0 and offset > sweep + tolerance:
        return {
            "status": "unreadable",
            "value": None,
            "reason": "needle angle is outside the configured scale arc",
            "offset_deg": round(offset, 3),
            "ratio": None,
        }
    bounded = min(sweep, max(0.0, offset))
    ratio = bounded / sweep
    value = minimum + ratio * (maximum - minimum)
    decimals = int(calibration.get("decimal_places", 2))
    decimals = max(0, min(decimals, 6))
    return {
        "status": "stable",
        "value": round(value, decimals),
        "reason": "",
        "offset_deg": round(offset, 3),
        "ratio": round(ratio, 6),
    }


class EthzGaugeReader:
    """Runs the three ETH Zurich ONNX models and reconstructs the dial reading.

    Sessions are created lazily and cached on the instance, so a warm reader
    pays the model load once rather than per photograph.  ``read`` is safe to
    call from several threads; inference itself is guarded by ``_infer_lock``
    because the K3's CPU provider is the only execution provider available.
    """

    def __init__(self, model_dir: Optional[str] = None,
                 provider: str = "CPUExecutionProvider", num_threads: int = 4):
        self.model_dir = model_dir or os.environ.get("RVCLAW_GAUGE_MODEL_DIR") or DEFAULT_MODEL_DIR
        self.provider = provider
        self.num_threads = int(num_threads)
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # -- model plumbing ---------------------------------------------------

    def model_path(self, kind: str) -> str:
        return os.path.join(self.model_dir, MODEL_FILES[kind])

    def models_present(self, model_dir: Optional[str] = None) -> bool:
        """Cheap existence check, so callers can gate before a real load."""
        base = model_dir or self.model_dir
        return all(os.path.isfile(os.path.join(base, name)) for name in MODEL_FILES.values())

    def session(self, kind: str):
        with self._lock:
            cached = self._sessions.get(kind)
            if cached is not None:
                return cached
            import onnxruntime as ort

            path = self.model_path(kind)
            if not os.path.isfile(path):
                raise FileNotFoundError("gauge model not found: %s" % path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = max(1, self.num_threads)
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            created = ort.InferenceSession(path, options, providers=[self.provider])
            self._sessions[kind] = created
            return created

    # -- individual stages ------------------------------------------------

    def detect_gauge_face(self, image: np.ndarray):
        """Highest-confidence gauge-face box plus every box found."""
        session = self.session("detection")
        padded, ratio, pad_x, pad_y = _letterbox(image)
        name = session.get_inputs()[0].name
        with self._infer_lock:
            output = session.run(None, {name: _blob(padded)})[0][0]
        scores = output[4:].max(axis=0)
        keep = np.nonzero(scores > DET_CONFIDENCE)[0]
        if keep.size == 0:
            return None, np.empty((0, 4)), np.empty((0,))
        boxes = _decode_boxes(output, keep)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        found = scores[keep]
        order = _nms(boxes, found, DET_IOU)
        return boxes[order][0], boxes[order], found[order]

    def keypoint_heatmaps(self, image448: np.ndarray) -> np.ndarray:
        """3 x 448 x 448 heat maps; upstream preprocessing is RGB and /255 only."""
        session = self.session("keypoint")
        rgb = cv2.cvtColor(image448, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob.transpose(2, 0, 1)[None, ...])
        name = session.get_inputs()[0].name
        with self._infer_lock:
            output = session.run(None, {name: blob})[0]
        return np.asarray(output)[0]

    def needle_mask(self, image: np.ndarray, resolution) -> Optional[np.ndarray]:
        """Boolean needle mask at ``resolution``, mirroring ultralytics post-processing."""
        session = self.session("segmentation")
        padded, ratio, pad_x, pad_y = _letterbox(image)
        name = session.get_inputs()[0].name
        with self._infer_lock:
            outputs = session.run(None, {name: _blob(padded)})
        detections = np.asarray(outputs[0])[0]
        prototypes = np.asarray(outputs[1])[0]
        n_coeff = prototypes.shape[0]

        scores = detections[4:4 + 1].max(axis=0)
        keep = np.nonzero(scores > DET_CONFIDENCE)[0]
        if keep.size == 0:
            return None
        boxes = _decode_boxes(detections, keep)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio
        found = scores[keep]
        order = _nms(boxes, found, DET_IOU)
        if order.size == 0:
            return None

        best_index = int(keep[order[0]])
        best_box = boxes[order][0]
        coefficients = detections[4 + 1:4 + 1 + n_coeff, best_index]
        proto_h, proto_w = prototypes.shape[1:]
        mask = _sigmoid(coefficients @ prototypes.reshape(n_coeff, -1)).reshape(proto_h, proto_w)

        # ultralytics crops the proto-resolution mask to the box, then upsamples.
        height, width = image.shape[:2]
        scale_x, scale_y = proto_w / width, proto_h / height
        x1, y1, x2, y2 = (best_box[0] * scale_x, best_box[1] * scale_y,
                          best_box[2] * scale_x, best_box[3] * scale_y)
        columns = np.arange(proto_w)[None, :]
        rows = np.arange(proto_h)[:, None]
        mask = mask * ((columns >= x1) & (columns < x2)) * ((rows >= y1) & (rows < y2))

        target_w, target_h = int(resolution[0]), int(resolution[1])
        upsampled = cv2.resize(mask.astype(np.float32), (target_w, target_h),
                               interpolation=cv2.INTER_LINEAR)
        return upsampled > 0.5

    # -- OCR ---------------------------------------------------------------

    def _ocr_readings(self, cropped: np.ndarray, ocr: Callable,
                      params, theta_zero: float):
        """OCR the dial, returning readings in the cropped image's coordinates.

        Upstream's default is to rectify the ellipse to a circle and rotate the
        zero point to the bottom before reading.  That is kept, but only as a
        second attempt.  The rotation is a *global* one, while the digits around
        a dial each carry their own orientation, so it can turn a perfectly
        readable dial upside down just as easily as it can straighten a squashed
        one; the plain crop preserves whatever orientation the digits were
        photographed in and needs no polygon back-mapping at all.  The rectified
        view is fallen back on when the plain read does not turn up a usable
        scale.

        Either way the polygons end up in the cropped image's coordinates, which
        is what the caller scales into the 448x448 frame the ellipse lives in.
        """
        attempts: List[Dict[str, Any]] = []

        def run(frame: np.ndarray, back_map, label: str) -> None:
            readings = [OCRReading(item["poly"], item.get("text", ""), item.get("score", 0.0))
                        for item in (ocr(frame) or [])]
            if back_map is not None:
                for reading in readings:
                    reading.set_polygon(back_map(reading.polygon))
            attempts.append({"mode": label, "readings": readings,
                             "numbers": _count_dial_numbers(readings)})

        run(cropped, None, "direct")
        if attempts[0]["numbers"] < 2:
            try:
                # The warp runs on the full-resolution crop, so the ellipse --
                # fitted in the 448x448 frame -- has to be rescaled into it
                # first, exactly as upstream ``ocr_warp`` does.
                crop_h, crop_w = cropped.shape[:2]
                local = _rescale_ellipse(params, GAUGE_RESOLUTION, (crop_w, crop_h))
                zero_local = _move_point_resize(_point_from_angle(theta_zero, params),
                                                GAUGE_RESOLUTION, (crop_w, crop_h))
                warped, matrix = _warp_ellipse_to_circle(cropped, local)
                centre_warped = _transform_point(local[:2], matrix)
                zero_warped = _transform_point(zero_local, matrix)
                angle = math.degrees(math.atan2(zero_warped[1] - centre_warped[1],
                                                zero_warped[0] - centre_warped[0]))
                rotated, affine = _rotate_about(warped, centre_warped,
                                                angle - OCR_DESIRED_ANGLE)
                inverse_affine = cv2.invertAffineTransform(affine)

                def back_map(polygon):
                    moved = _apply_affine(inverse_affine, polygon)
                    return _transform_polygon_inverse(moved, matrix)

                run(rotated, back_map, "rectified")
            except Exception as exc:  # noqa: BLE001 - a failed warp must not abort the read
                attempts.append({"mode": "rectified", "readings": [], "numbers": 0,
                                 "error": "%s: %s" % (type(exc).__name__, exc)})

        best = max(attempts, key=lambda item: item["numbers"])
        summary = [{"mode": item["mode"], "numbers": item["numbers"],
                    "texts": [r.reading for r in item["readings"]]}
                   if "error" not in item else
                   {"mode": item["mode"], "numbers": 0, "error": item["error"]}
                   for item in attempts]
        return best["readings"], summary

    # -- top level ---------------------------------------------------------

    def read(
        self,
        image: np.ndarray,
        ocr: Optional[Callable] = None,
        calibration: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Read one dial.

        ``ocr`` is a callable taking a BGR image and returning a sequence of
        ``{"text", "score", "poly"}`` mappings -- pass
        ``lambda frame: ppocr.recognize(frame)["items"]``.  Without it the
        geometry still runs and the result reports ``ocr_unavailable``.
        """
        notes: Dict[str, Any] = {"stage": "start"}
        if image is None or getattr(image, "size", 0) == 0:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "empty image", "notes": notes}
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # --- gauge face -------------------------------------------------
        try:
            box, all_boxes, all_scores = self.detect_gauge_face(image)
        except FileNotFoundError as exc:
            return {"kind": "pointer", "status": "unsupported", "value": None,
                    "reason": str(exc), "notes": notes}
        notes["stage"] = "detection"
        notes["gauge_boxes"] = int(all_boxes.shape[0])
        if box is None:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "no gauge face detected", "notes": notes}
        notes["gauge_box"] = [round(float(v), 1) for v in box]
        notes["gauge_score"] = round(float(all_scores[0]), 4)

        cropped = _crop_square(image, box)
        gauge_frame = cv2.resize(cropped, GAUGE_RESOLUTION, interpolation=cv2.INTER_CUBIC)
        frame_h, frame_w = gauge_frame.shape[:2]

        # --- key points -------------------------------------------------
        notes["stage"] = "keypoints"
        heatmaps = self.keypoint_heatmaps(gauge_frame)
        candidates = []
        for threshold in KEYPOINT_THRESHOLDS:
            current = _extract_key_points(heatmaps, threshold, MEANSHIFT_BANDWIDTH)
            candidates.append((float(threshold), *current))
        threshold, notch, start, end = max(
            candidates,
            key=lambda item: (item[1].shape[0], item[0]),
        )
        notes["keypoint_threshold"] = round(threshold, 3)
        notes["keypoint_candidates"] = [
            {"threshold": round(item[0], 3), "notch_points": int(item[1].shape[0])}
            for item in candidates
        ]
        notes["notch_points"] = int(notch.shape[0])
        notes["start_found"] = start.shape == (1, 2)
        notes["end_found"] = end.shape == (1, 2)
        if notch.shape[0] < 5:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "only %d key points on the dial arc" % notch.shape[0],
                    "notes": notes}

        # --- ellipse ----------------------------------------------------
        notes["stage"] = "ellipse"
        try:
            params = _cart_to_pol(_fit_ellipse(notch[:, 0], notch[:, 1]))
        except Exception as exc:  # noqa: BLE001
            notes["ellipse_error"] = "%s: %s" % (type(exc).__name__, exc)
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "ellipse fit failed", "notes": notes}
        notes["ellipse"] = [round(float(v), 3) for v in params]
        notes["ellipse_fit_error_px"] = round(_ellipse_error(notch, params), 4)

        if start.shape == (1, 2) and end.shape == (1, 2):
            theta_zero = _theta_middle(_polar_angle(start.ravel(), params),
                                       _polar_angle(end.ravel(), params))
        else:
            theta_zero = _polar_angle((frame_w / 2.0, float(frame_h)), params)
        notes["theta_zero_deg"] = round(math.degrees(theta_zero) % 360.0, 2)

        # --- needle -----------------------------------------------------
        notes["stage"] = "needle"
        mask = self.needle_mask(gauge_frame, GAUGE_RESOLUTION)
        if mask is None:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "no needle found", "notes": notes}
        ys, xs = np.where(mask)
        notes["needle_pixels"] = int(xs.size)
        if xs.size < 20:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "needle mask only %d pixels" % xs.size, "notes": notes}
        try:
            origin, direction, residual = _needle_line(xs, ys)
        except ValueError as exc:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": str(exc), "notes": notes}
        t_min, t_max = _needle_extent(xs, ys, origin, direction)
        notes["needle_line_rms_px"] = round(residual, 3)

        candidates = _line_ellipse_intersections(origin, direction, params)
        needle_point = _choose_ellipse_point(candidates, origin, direction, t_min, t_max)
        if needle_point is None:
            return {"kind": "pointer", "status": "unreadable", "value": None,
                    "reason": "needle line does not meet the dial arc", "notes": notes}
        needle_angle = _polar_angle(needle_point, params)
        # `needle_angle` lives in the fitted ellipse's own frame.  For a
        # near-circular dial that frame is an arbitrary rotation of the photo's
        # -- which is harmless, because the angle->value fit below uses the very
        # same frame, but it makes the number meaningless to an operator staring
        # at the dial.  The needle's direction *on the dial face* is the one that
        # can be checked against the photo: 0 deg is 3 o'clock and the angle
        # grows clockwise on screen.  The square crop is resized isotropically,
        # so this angle is identical in the uploaded image.
        screen_angle = math.atan2(needle_point[1] - params[1],
                                  needle_point[0] - params[0]) % (2.0 * math.pi)
        notes["needle_screen_deg"] = round(math.degrees(screen_angle), 2)
        needle_screen_deg = round(math.degrees(screen_angle), 2)

        if calibration is not None:
            try:
                calibrated = configured_scale_value(needle_screen_deg, calibration)
            except (TypeError, ValueError) as error:
                return {
                    "kind": "pointer", "status": "unreadable", "value": None,
                    "unit": calibration.get("unit") if isinstance(calibration, Mapping) else None,
                    "reason": "invalid pointer calibration: %s" % error,
                    "needle_angle_deg": round(math.degrees(needle_angle) % 360.0, 2),
                    "screen_angle_deg": needle_screen_deg,
                    "notes": notes,
                }
            notes["configured_scale"] = {
                "zero_angle_deg": float(calibration["zero_angle_deg"]),
                "sweep_deg": float(calibration.get("sweep_deg", 360.0)),
                "min_value": float(calibration["min_value"]),
                "max_value": float(calibration["max_value"]),
                "direction": str(calibration.get("direction", "clockwise")),
                "offset_deg": calibrated["offset_deg"],
                "ratio": calibrated["ratio"],
            }
            confidence = min(0.95, 0.52 + 0.035 * min(int(notch.shape[0]), 10))
            return {
                "kind": "pointer",
                "status": calibrated["status"],
                "value": calibrated["value"],
                "unit": calibration.get("unit"),
                "confidence": round(confidence, 4) if calibrated["value"] is not None else 0.0,
                "reason": calibrated["reason"],
                "needle_angle_deg": round(math.degrees(needle_angle) % 360.0, 2),
                "screen_angle_deg": needle_screen_deg,
                "scale_span": float(calibration["max_value"]) - float(calibration["min_value"]),
                "gauge_box": [round(float(v), 1) for v in box],
                "gauge_score": round(float(all_scores[0]), 4),
                "candidates": [],
                "value_method": "configured_pointer_scale",
                "notes": notes,
            }

        # --- OCR --------------------------------------------------------
        if ocr is None:
            return {"kind": "pointer", "status": "unsupported", "value": None,
                    "reason": "ocr_unavailable",
                    "needle_angle_deg": round(math.degrees(needle_angle) % 360.0, 2),
                    "screen_angle_deg": needle_screen_deg,
                    "notes": notes}
        notes["stage"] = "ocr"
        readings, ocr_attempts = self._ocr_readings(cropped, ocr, params, theta_zero)
        notes["ocr_attempts"] = ocr_attempts
        notes["ocr_readings"] = len(readings)

        # Upstream carries every OCR polygon from the cropped image into the
        # 448x448 frame the ellipse lives in.
        crop_h, crop_w = cropped.shape[:2]
        for reading in readings:
            polygon = reading.polygon.copy()
            polygon[:, 0] = polygon[:, 0] * GAUGE_RESOLUTION[1] / float(crop_w)
            polygon[:, 1] = polygon[:, 1] * GAUGE_RESOLUTION[0] / float(crop_h)
            reading.set_polygon(polygon)

        units = [r for r in readings if r.is_unit()]
        unit = units[0].reading if len(units) == 1 else None
        notes["unit_candidates"] = [r.reading for r in units]

        numbers = [reading for reading in readings if _is_dial_number(reading)]
        notes["dial_numbers"] = [r.reading for r in numbers]

        if len(numbers) < 2:
            return {"kind": "pointer", "status": "unreadable", "value": None, "unit": unit,
                    "reason": "only %d dial number(s) recognised, need 2" % len(numbers),
                    "needle_angle_deg": round(math.degrees(needle_angle) % 360.0, 2),
                    "screen_angle_deg": needle_screen_deg,
                    "notes": notes}

        # --- angle -> value fit -----------------------------------------
        notes["stage"] = "fit"
        for number in numbers:
            number.set_theta(_polar_angle(number.center, params))
        # The cut is only trustworthy if it falls in a gap between numbers.
        theta_zero, cut_source = _dial_zero_angle([n.theta for n in numbers], theta_zero)
        notes["theta_zero_keypoint_deg"] = notes["theta_zero_deg"]
        notes["theta_zero_deg"] = round(math.degrees(theta_zero) % 360.0, 2)
        notes["theta_zero_source"] = cut_source
        offset = _angle_offset(theta_zero)
        angle_values = np.array([[offset(number.theta), number.number] for number in numbers])
        try:
            coeffs, inlier_mask = _ransac_line(angle_values[:, 0], angle_values[:, 1])
        except Exception as exc:  # noqa: BLE001
            return {"kind": "pointer", "status": "unreadable", "value": None, "unit": unit,
                    "reason": "angle/value fit failed: %s" % exc, "notes": notes}

        reading_line = np.poly1d(coeffs)
        needle_offset = offset(needle_angle)
        value = float(reading_line(needle_offset))
        inliers = int(np.count_nonzero(inlier_mask))
        notes["fit_coeffs"] = [round(float(c), 6) for c in coeffs]
        notes["fit_inliers"] = inliers
        notes["fit_of"] = int(len(numbers))
        notes["angle_value_pairs"] = [[round(float(a), 2), float(v)]
                                      for a, v in angle_values]

        if not np.isfinite(value) or inliers < 2:
            return {"kind": "pointer", "status": "unreadable", "value": None, "unit": unit,
                    "reason": "fit collapsed (inliers=%d)" % inliers, "notes": notes}

        # A needle pointing past the last label means the linear fit is being
        # extrapolated, which is exactly how a confident wrong number appears.
        # Half a label step of slack keeps a needle that merely touches the end
        # of the scale in range.
        label_angles = np.sort(angle_values[:, 0])
        if label_angles.size >= 3:
            margin = 0.5 * float(np.median(np.diff(label_angles)))
        else:
            margin = math.radians(20.0)
        low, high = float(label_angles[0]), float(label_angles[-1])
        outside = not (low - margin <= needle_offset <= high + margin)
        notes["needle_offset_deg"] = round(math.degrees(needle_offset), 2)
        notes["label_span_deg"] = round(math.degrees(high - low), 2)

        span = float(np.ptp(angle_values[:, 1]))
        candidates = [{
            "text": number.reading,
            "value": float(number.number),
            "score": round(number.confidence, 4),
            "angle_deg": round(math.degrees(number.theta) % 360.0, 2),
            "offset_deg": round(math.degrees(offset(number.theta)), 2),
            "bbox": [round(v, 1) for v in number.get_bounding_box()],
            "inlier": bool(inlier_mask[index]),
        } for index, number in enumerate(numbers)]
        return {
            "kind": "pointer",
            "status": "out_of_range" if outside else "stable",
            "value": round(value, 4),
            "unit": unit,
            "confidence": round(min(0.99, 0.35 + 0.08 * inliers) * (0.75 if outside else 1.0), 4),
            "needle_angle_deg": round(math.degrees(needle_angle) % 360.0, 2),
            "screen_angle_deg": needle_screen_deg,
            "scale_span": span,
            "gauge_box": [round(float(v), 1) for v in box],
            "gauge_score": round(float(all_scores[0]), 4),
            "candidates": candidates,
            "notes": notes,
        }


def _angle_offset(theta_zero: float) -> Callable[[float], float]:
    """``AngleConverter`` from upstream, as a closure."""

    def convert(theta: float) -> float:
        shifted = theta - theta_zero
        if shifted < 0:
            shifted += 2 * math.pi
        return shifted

    return convert


def _crop_square(image: np.ndarray, box) -> np.ndarray:
    """Square crop of the gauge face, black-padded (upstream ``crop_image``).

    The box is clipped first: upstream slices with the raw box, and a negative
    index silently yields an empty crop when the gauge touches the border
    (deviation 7).
    """
    height, width = image.shape[:2]
    x1 = int(max(0, min(width - 1, math.floor(float(box[0])))))
    y1 = int(max(0, min(height - 1, math.floor(float(box[1])))))
    x2 = int(max(x1 + 1, min(width, math.ceil(float(box[2])))))
    y2 = int(max(y1 + 1, min(height, math.ceil(float(box[3])))))
    cropped = image[y1:y2, x1:x2]

    box_h, box_w = cropped.shape[:2]
    if box_h > box_w:
        delta = box_h - box_w
        left, right = delta // 2, delta - delta // 2
        top = bottom = 0
    else:
        delta = box_w - box_h
        top, bottom = delta // 2, delta - delta // 2
        left = right = 0
    return cv2.copyMakeBorder(cropped, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))
