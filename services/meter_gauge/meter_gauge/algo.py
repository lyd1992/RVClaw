"""NanoDet + YOLOv8-Pose 指针表读数 —— 向量化 Python/ncnn 实现

上游：https://github.com/zhahoi/Nanodet-YoloV8-Pose-MeterReader
本文件是 `tools/nanodet_pose_meter_py.py`（逐函数对齐上游 C++ 的移植版）的
**等价向量化版**。三处原为纯 Python 三重循环的热点改为 numpy 批量运算，
否则在 K3（RISC-V 8 核）上单张耗时会到十秒级：

  - nd_decode                 2125 个 prior × 4 次 softmax
  - pose_generate_proposals   2100 个格子 × sigmoid/softmax
  - thinning_rosenfeld        Rosenfeld 细化，逐像素 × 4 方向

等价性判据（见 tests/test_algo_equiv.py）：同一输入下 `status` 与 `ratio`
与原实现**逐位一致**（不设容差）。

与上游的差异（沿用评测适配器口径）：
  - 不调用 cv::imshow/waitKey 弹窗；
  - 不叠加按读数 +0.012/+0.008 的经验偏移；
  - 量程/单位由外部配置给出，模型只输出 [0,1] 比例。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

import ncnn

# ---------------------------------------------------------------- 常量（对应 yolov8_pose.h）
MAX_STRIDE = 32
SCALE_BEGINNING = 0.0
SCALE_END = 1.0
TARGET_SIZE = 320
PROB_THRESHOLD = 0.25
NMS_THRESHOLD = 0.45
ASPECT_RATIO_THRESHOLD = 0.90

CLASS_NAMES = ("pointer_rect", "left_rect", "right_rect")

# nanodet.h
ND_INPUT = 320
ND_NUM_CLASS = 1
ND_REG_MAX = 7
ND_STRIDES = (8, 16, 32, 64)
ND_SCORE_THRESHOLD = 0.3
ND_NMS_THRESHOLD = 0.3

POSE_REG_MAX = 16
POSE_NUM_CLASS = 3
POSE_KPS_NUM = 2
POSE_OUT_WIDTH = POSE_NUM_CLASS + 4 * POSE_REG_MAX + POSE_KPS_NUM * 3  # = 73


@dataclass
class BoxInfo:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    label: int


@dataclass
class Det:
    rect: np.ndarray  # [x, y, w, h] float32
    label: int
    prob: float
    kps: list = field(default_factory=list)


# ================================================================ ncnn 封装
class Net:
    """薄封装：加载 param/bin，按名字抽取输出。"""

    def __init__(self, param: str, bin_path: str, threads: int = 4):
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.num_threads = threads
        if self.net.load_param(param) != 0:
            raise RuntimeError(f"load_param failed: {param}")
        if self.net.load_model(bin_path) != 0:
            raise RuntimeError(f"load_model failed: {bin_path}")

    def run(self, input_name: str, mat: "ncnn.Mat", outputs):
        ex = self.net.create_extractor()
        ex.set_light_mode(False)
        if ex.input(input_name, mat) != 0:
            raise RuntimeError(f"input failed: {input_name}")
        out = {}
        for name in outputs:
            m = ncnn.Mat()
            if ex.extract(name, m) != 0:
                raise RuntimeError(f"extract failed: {name}")
            out[name] = np.array(m)
        return out


def mat_from_uint8(img: np.ndarray, pixel_type) -> "ncnn.Mat":
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    return ncnn.Mat.from_pixels(img.tobytes(), pixel_type, w, h)


# ================================================================ NanoDet（表盘检测）
def nd_resize_uniform(src: np.ndarray):
    """对应 NanoDet::resize_uniform —— 黑边居中 letterbox 到 320x320。"""
    h, w = src.shape[:2]
    dst_w = dst_h = ND_INPUT
    dst = np.zeros((dst_h, dst_w, 3), np.uint8)

    ratio_src = w / h
    ratio_dst = dst_w / dst_h

    if abs(ratio_src - ratio_dst) < 1e-12:
        return cv2.resize(src, (dst_w, dst_h)), (0, 0, dst_w, dst_h)

    if ratio_src > ratio_dst:
        tmp_w = dst_w
        tmp_h = int(math.floor((dst_w / w) * h))
    else:
        tmp_h = dst_h
        tmp_w = int(math.floor((dst_h / h) * w))

    tmp = cv2.resize(src, (tmp_w, tmp_h))

    if tmp_w != dst_w:
        index_w = int(math.floor((dst_w - tmp_w) / 2.0))
        dst[:, index_w:index_w + tmp_w] = tmp
        return dst, (index_w, 0, tmp_w, tmp_h)
    if tmp_h != dst_h:
        index_h = int(math.floor((dst_h - tmp_h) / 2.0))
        dst[index_h:index_h + tmp_h, :] = tmp
        return dst, (0, index_h, tmp_w, tmp_h)
    raise RuntimeError("resize_uniform: unreachable")


def nd_preprocess(letterboxed_bgr: np.ndarray) -> "ncnn.Mat":
    """对应 NanoDet::preprocess —— PIXEL_BGR + ImageNet BGR mean/norm。"""
    mat = mat_from_uint8(letterboxed_bgr, ncnn.Mat.PixelType.PIXEL_BGR)
    mat.substract_mean_normalize([103.53, 116.28, 123.675],
                                 [0.017429, 0.017507, 0.017125])
    return mat


_PRIORS = None


def _nd_priors():
    """生成顺序与上游一致：stride 外层、y 中层、x 内层。"""
    global _PRIORS
    if _PRIORS is None:
        xs, ys, ss = [], [], []
        for stride in ND_STRIDES:
            n = math.ceil(ND_INPUT / stride)
            yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            xs.append(xx.ravel())
            ys.append(yy.ravel())
            ss.append(np.full(n * n, stride, np.float32))
        _PRIORS = (np.concatenate(xs).astype(np.float32),
                   np.concatenate(ys).astype(np.float32),
                   np.concatenate(ss))
    return _PRIORS


ND_NUM_PRIORS = 40 * 40 + 20 * 20 + 10 * 10 + 5 * 5  # 2125


def nd_decode(feats: np.ndarray, threshold: float) -> list:
    """对应 NanoDet::decode_infer + disPred2Bbox（DFL, reg_max=7）—— 向量化。"""
    if feats.shape[0] != ND_NUM_PRIORS and feats.shape[1] == ND_NUM_PRIORS:
        feats = feats.T
    if feats.shape[0] != ND_NUM_PRIORS or feats.shape[1] != ND_NUM_CLASS + 4 * (ND_REG_MAX + 1):
        raise ValueError(f"unexpected nanodet output shape: {feats.shape}")

    feats = np.ascontiguousarray(feats, dtype=np.float32)
    xs, ys, ss = _nd_priors()

    scores = feats[:, :ND_NUM_CLASS]
    label = np.argmax(scores, axis=1).astype(np.int32)
    score = scores[np.arange(scores.shape[0]), label]

    keep = score > threshold
    if not keep.any():
        return []

    dfl = feats[:, ND_NUM_CLASS:ND_NUM_CLASS + 4 * (ND_REG_MAX + 1)].reshape(-1, 4, ND_REG_MAX + 1)
    m = dfl.max(axis=2, keepdims=True)
    e = np.exp(dfl - m)
    sm = e / e.sum(axis=2, keepdims=True)
    proj = np.arange(ND_REG_MAX + 1, dtype=np.float32)
    dis = (sm * proj).sum(axis=2) * ss[:, None]        # (N, 4)

    ct_x = xs * ss
    ct_y = ys * ss
    x1 = np.maximum(ct_x - dis[:, 0], 0.0)
    y1 = np.maximum(ct_y - dis[:, 1], 0.0)
    x2 = np.minimum(ct_x + dis[:, 2], float(ND_INPUT))
    y2 = np.minimum(ct_y + dis[:, 3], float(ND_INPUT))

    idx = np.nonzero(keep)[0]
    return [BoxInfo(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                    float(score[i]), int(label[i])) for i in idx]


def nd_nms(boxes: list, thresh: float) -> list:
    """对应 NanoDet::nms —— 逐类别、按 score 降序、面积含 +1。"""
    boxes = sorted(boxes, key=lambda b: -b.score)
    area = [(b.x2 - b.x1 + 1) * (b.y2 - b.y1 + 1) for b in boxes]
    i = 0
    while i < len(boxes):
        j = i + 1
        while j < len(boxes):
            xx1 = max(boxes[i].x1, boxes[j].x1)
            yy1 = max(boxes[i].y1, boxes[j].y1)
            xx2 = min(boxes[i].x2, boxes[j].x2)
            yy2 = min(boxes[i].y2, boxes[j].y2)
            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (area[i] + area[j] - inter)
            if ovr >= thresh:
                boxes.pop(j)
                area.pop(j)
            else:
                j += 1
        i += 1
    return boxes


def nd_convert_boxes(boxes: list, roi, src_w: int, src_h: int) -> list:
    """对应 NanoDet::convert_boxes —— 去 letterbox，映射回原图坐标。"""
    rx, ry, rw, rh = roi
    wr = src_w / rw
    hr = src_h / rh
    dets = []
    for b in boxes:
        x1 = (b.x1 - rx) * wr
        y1 = (b.y1 - ry) * hr
        x2 = (b.x2 - rx) * wr
        y2 = (b.y2 - ry) * hr
        dets.append(Det(rect=np.array([x1, y1, x2 - x1, y2 - y1], np.float32),
                        label=b.label, prob=b.score))
    return dets


def detect_meters(net: Net, image_bgr: np.ndarray) -> list:
    lb, roi = nd_resize_uniform(image_bgr)
    mat = nd_preprocess(lb)
    feats = net.run("data", mat, ["output"])["output"]
    boxes = nd_decode(feats, ND_SCORE_THRESHOLD)
    per_class = {}
    for b in boxes:
        per_class.setdefault(b.label, []).append(b)
    keep = []
    for lab in per_class:
        keep.extend(nd_nms(per_class[lab], ND_NMS_THRESHOLD))
    return nd_convert_boxes(keep, roi, image_bgr.shape[1], image_bgr.shape[0])


# ================================================================ YOLOv8-Pose（关键点）
def _pose_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _pose_softmax_weight(seg: np.ndarray) -> np.ndarray:
    """对应 Yolov8Pose::softmax —— sum(i * softmax_i)，沿最后一维。"""
    m = seg.max(axis=-1, keepdims=True)
    e = np.exp(seg - m)
    p = e / e.sum(axis=-1, keepdims=True)
    return (p * np.arange(seg.shape[-1], dtype=np.float32)).sum(axis=-1)


def pose_generate_proposals(stride: int, blob: np.ndarray, prob_threshold: float) -> list:
    """对应 Yolov8Pose::generate_proposals（reg_max=16, num_class=3, kps_num=2）。

    blob 形状 (grid_y, grid_x, 73)。

    性能与保真的折中：sigmoid/argmax 部分整批向量化（原版对全部 2100 个格子
    都算一遍），但**几何量逐格照抄原表达式**（不换成 float32 数组运算）——
    两者 dtype 提升路径不同，会让 ratio 出现 1e-2 量级偏差。
    存活格子通常只有几十个，逐点代价可忽略。
    """
    if blob.ndim != 3 or blob.shape[2] != POSE_OUT_WIDTH:
        raise ValueError(f"unexpected pose output shape: {blob.shape}")

    cls = 1.0 / (1.0 + np.exp(-blob[:, :, :POSE_NUM_CLASS]))
    label = np.argmax(cls, axis=2)
    score = np.max(cls, axis=2)

    ys, xs = np.nonzero(score >= prob_threshold)   # 上游：score < thresh 才 continue
    if ys.size == 0:
        return []

    num_class = POSE_NUM_CLASS
    base = num_class + 4 * POSE_REG_MAX

    objs = []
    for gy_i, gx_j in zip(ys.tolist(), xs.tolist()):
        matat = blob[gy_i, gx_j]

        x0 = gx_j + 0.5 - _pose_softmax_weight(matat[num_class:num_class + POSE_REG_MAX])
        y0 = gy_i + 0.5 - _pose_softmax_weight(matat[num_class + POSE_REG_MAX:num_class + 2 * POSE_REG_MAX])
        x1 = gx_j + 0.5 + _pose_softmax_weight(matat[num_class + 2 * POSE_REG_MAX:num_class + 3 * POSE_REG_MAX])
        y1 = gy_i + 0.5 + _pose_softmax_weight(matat[num_class + 3 * POSE_REG_MAX:num_class + 4 * POSE_REG_MAX])
        x0 *= stride
        y0 *= stride
        x1 *= stride
        y1 *= stride

        kps = []
        for k in range(POSE_KPS_NUM):
            kx = (matat[base + k * 3] * 2.0 + gx_j) * stride
            ky = (matat[base + k * 3 + 1] * 2.0 + gy_i) * stride
            ks = 1.0 / (1.0 + np.exp(-matat[base + k * 3 + 2]))
            kps += [float(kx), float(ky), float(ks)]

        objs.append(Det(rect=np.array([x0, y0, x1 - x0, y1 - y0], np.float32),
                        label=int(label[gy_i, gx_j]),
                        prob=float(score[gy_i, gx_j]), kps=kps))
    return objs


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def pose_nms(proposals: list, orin_h: int, orin_w: int, dh: float, dw: float,
             ratio_h: float, ratio_w: float, conf: float, iou: float) -> list:
    """对应 Yolov8Pose::non_max_suppression —— cv::dnn::NMSBoxes + 去 letterbox。"""
    if not proposals:
        return []
    boxes = [[int(p.rect[0]), int(p.rect[1]), int(p.rect[2]), int(p.rect[3])] for p in proposals]
    scores = [p.prob for p in proposals]
    idx = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    if len(idx) == 0:
        return []
    idx = np.array(idx).reshape(-1).tolist()

    results = []
    for i in idx:
        p = proposals[i]
        x0, y0, w, h = p.rect
        x1, y1 = x0 + w, y0 + h
        x0 = _clamp((x0 - dw) / ratio_w, 0.0, orin_w)
        y0 = _clamp((y0 - dh) / ratio_h, 0.0, orin_h)
        x1 = _clamp((x1 - dw) / ratio_w, 0.0, orin_w)
        y1 = _clamp((y1 - dh) / ratio_h, 0.0, orin_h)
        kps = list(p.kps)
        for n in range(0, len(kps), 3):
            kps[n] = _clamp((kps[n] - dw) / ratio_w, 0.0, orin_w)
            kps[n + 1] = _clamp((kps[n + 1] - dh) / ratio_h, 0.0, orin_h)
        results.append(Det(rect=np.array([x0, y0, x1 - x0, y1 - y0], np.float32),
                           label=p.label, prob=p.prob, kps=kps))
    return results


def pose_letterbox(bgr: np.ndarray):
    """对应 Yolov8Pose::detect_yolov8 的前处理（letterbox 到 320，pad 114）。"""
    img_h, img_w = bgr.shape[:2]
    w, h = img_w, img_h
    if w > h:
        scale = TARGET_SIZE / w
        w, h = TARGET_SIZE, int(h * scale)
    else:
        scale = TARGET_SIZE / h
        h, w = TARGET_SIZE, int(w * scale)

    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)

    wpad = (w + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE - w
    hpad = (h + MAX_STRIDE - 1) // MAX_STRIDE * MAX_STRIDE - h
    top, bottom = hpad // 2, hpad - hpad // 2
    left, right = wpad // 2, wpad - wpad // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    mat = mat_from_uint8(rgb, ncnn.Mat.PixelType.PIXEL_RGB)
    mat.substract_mean_normalize([0.0, 0.0, 0.0], [1 / 255.0, 1 / 255.0, 1 / 255.0])
    return mat, (top, left, hpad, wpad)


POSE_OUTPUTS = (("output0", 8), ("378", 16), ("403", 32))


def detect_pose(net: Net, bgr: np.ndarray) -> list:
    mat, (dh, dw, _hpad, _wpad) = pose_letterbox(bgr)
    raw = net.run("images", mat, [n for n, _ in POSE_OUTPUTS])
    orin_h, orin_w = bgr.shape[:2]
    scale = (TARGET_SIZE / orin_w) if orin_w > orin_h else (TARGET_SIZE / orin_h)

    proposals = []
    for name, stride in POSE_OUTPUTS:
        proposals += pose_generate_proposals(stride, raw[name], PROB_THRESHOLD)
    return pose_nms(proposals, orin_h, orin_w, dh, dw, scale, scale, PROB_THRESHOLD, NMS_THRESHOLD)


# ================================================================ 指针细化（Rosenfeld 细化 + Hough）
def thinning_rosenfeld(binary: np.ndarray) -> np.ndarray:
    """对应 Yolov8Pose::Thining_Rosenfeld —— 向量化版本。

    上游按 4 个方向轮流扫描，每个方向开始时对当前结果取快照 `src`，扫描中
    只读 `src`、只写 `dst` ⇒ 同一方向内多个像素的删除互不影响，因此可以
    整批用 numpy 计算。等价性由 tests/test_algo_equiv.py 逐字节验证。

    含上游 `p3 == 9` 的笔误（恒假 ⇒ 该分支等效于去掉），此处按等价式移植。
    """
    if binary.dtype != np.uint8:
        raise TypeError("binary must be uint8")
    dst = binary.copy()

    # 4 个方向，与上游 dirs = {(-1,0),(1,0),(0,1),(0,-1)} 一一对应：
    #   (-1,0) 北 → 邻居 P2 ；(1,0) 南 → P6 ；(0,1) 东 → P4 ；(0,-1) 西 → P8
    dirs = ("p2", "p6", "p4", "p8")

    while True:
        changed = False
        for which in dirs:
            src = dst
            b = src > 0
            C = b[1:-1, 1:-1]
            P2 = b[:-2, 1:-1]
            P3 = b[:-2, 2:]
            P4 = b[1:-1, 2:]
            P5 = b[2:, 2:]
            P6 = b[2:, 1:-1]
            P7 = b[2:, :-2]
            P8 = b[1:-1, :-2]
            P9 = b[:-2, :-2]

            neigh = {"p2": P2, "p6": P6, "p4": P4, "p8": P8}[which]
            cand = C & ~neigh
            if not cand.any():
                continue

            n2, n3, n4 = ~P2, ~P3, ~P4
            n5, n6, n7 = ~P5, ~P6, ~P7
            n8, n9 = ~P8, ~P9

            not_simple = (
                (n2 & n6 & (P9 | P8 | P7) & (P3 | P4 | P5))
                | (n4 & n8 & (P9 | P2 | P3) & (P5 | P6 | P7))
                | (n8 & n2 & P9 & (P3 | P4 | P5 | P6 | P7))
                | (n4 & n2 & P3 & (P5 | P6 | P7 | P8 | P9))
                | (n8 & n6 & P7 & (P2 | P3 | P4 | P5))
                | (n4 & n6 & P5 & (P7 | P8 | P9 | P2 | P3))
            )

            adjsum = (P2.astype(np.int8) + P3 + P4 + P5 + P6 + P7 + P8 + P9)

            delete = cand & ~not_simple & (adjsum >= 2)
            if delete.any():
                sub = dst[1:-1, 1:-1]
                sub[delete] = 0
                changed = True

        if not changed:
            break
    return dst


def get_pointer_lines(roi_bgr: np.ndarray, rect_xywh):
    """对应 Yolov8Pose::getPointerLines —— 返回 line4（已加 rect 偏移）或 None。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    inv = cv2.bitwise_not(hist)
    blur = cv2.medianBlur(inv, 3)
    erode = cv2.erode(blur, kernel)
    opened = cv2.morphologyEx(erode, cv2.MORPH_CLOSE, kernel)
    _, thres = cv2.threshold(opened, 210, 255, cv2.THRESH_BINARY)

    thinned = thinning_rosenfeld(thres)
    lines = cv2.HoughLinesP(thinned, 1, math.pi / 180, 50, minLineLength=100, maxLineGap=150)
    if lines is None or len(lines) < 1:
        return None
    l = lines[0][0]
    rx, ry = float(rect_xywh[0]), float(rect_xywh[1])
    return [float(l[0]) + rx, float(l[1]) + ry, float(l[2]) + rx, float(l[3]) + ry]


def get_pointer_point(center: np.ndarray, line) -> np.ndarray:
    """对应 Yolov8Pose::getPointerPoint —— 取距 center 更远的端点。"""
    p1 = np.array([line[0], line[1]], np.float32)
    p2 = np.array([line[2], line[3]], np.float32)
    return p1 if float(np.linalg.norm(p1 - center)) > float(np.linalg.norm(p2 - center)) else p2


def get_angle_ratio(start: np.ndarray, end: np.ndarray, tip: np.ndarray, center: np.ndarray) -> float:
    """对应 Yolov8Pose::getAngleRatio（逐行等价，含 y/x 判据分支）。"""
    beg = math.atan2(center[1] - start[1], start[0] - center[0])
    enda = math.atan2(center[1] - end[1], end[0] - center[0])
    beg_end = 2 * math.pi - (enda - beg)

    pt = math.atan2(center[1] - tip[1], tip[0] - center[0])
    if tip[1] > center[1] and tip[0] < center[0]:
        beg_ptr = pt - beg
    else:
        beg_ptr = 2 * math.pi - (pt - beg)
    return beg_ptr / beg_end


def get_scale_value(ratio: float) -> float:
    return (SCALE_END - SCALE_BEGINNING) * ratio + SCALE_BEGINNING


# ================================================================ 单表整体流程
def read_meter(pose_net: Net, meter_bgr: np.ndarray, want_vis: bool = True):
    """对应 Yolov8Pose::get_results + process_objects（不弹窗、不加经验偏移）。"""
    objs = detect_pose(pose_net, meter_bgr)
    if not objs:
        return {"status": "missing_pose_class", "reason": "Doesn't detect any objects", "objects": []}

    pointer_line = None
    start = end = center_ori = tip_ori = None

    for o in objs:
        if CLASS_NAMES[o.label] == "pointer_rect":
            rx, ry, rw, rh = (int(o.rect[0]), int(o.rect[1]), int(o.rect[2]), int(o.rect[3]))
            if rw > 0 and rh > 0:
                sub = meter_bgr[max(0, ry):ry + rh, max(0, rx):rx + rw]
                if sub.size:
                    pointer_line = get_pointer_lines(sub, o.rect)
        for k in range(POSE_KPS_NUM):
            kx, ky, ks = o.kps[k * 3], o.kps[k * 3 + 1], o.kps[k * 3 + 2]
            if ks <= 0.5:
                continue
            if CLASS_NAMES[o.label] == "pointer_rect":
                if k == 0:
                    center_ori = np.array([kx, ky], np.float32)
                else:
                    tip_ori = np.array([kx, ky], np.float32)
            if CLASS_NAMES[o.label] == "left_rect":
                start = np.array([kx, ky], np.float32)
            if CLASS_NAMES[o.label] == "right_rect":
                end = np.array([kx, ky], np.float32)

    missing = [n for n, v in (("start", start), ("end", end), ("center", center_ori)) if v is None]
    if missing:
        return {"status": "missing_pose_class", "reason": f"missing keypoint(s): {missing}",
                "objects": [o.__dict__ for o in objs]}

    center = center_ori
    if pointer_line is not None:
        tip = get_pointer_point(center, pointer_line)
        needle_method = "upstream_skeleton_hough"
    elif tip_ori is not None:
        tip = tip_ori
        needle_method = "pose_tip_fallback"
    else:
        return {"status": "missing_pose_class", "reason": "no pointer tip",
                "objects": [o.__dict__ for o in objs]}

    ratio = get_angle_ratio(start, end, tip, center)

    vis = None
    if want_vis:
        vis = meter_bgr.copy()
        for o in objs:
            x, y, w, h = o.rect
            cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)
            cv2.putText(vis, f"{CLASS_NAMES[o.label]} {o.prob:.2f}", (int(x), max(12, int(y) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            for k in range(POSE_KPS_NUM):
                kx, ky, ks = o.kps[k * 3], o.kps[k * 3 + 1], o.kps[k * 3 + 2]
                if ks > 0.5:
                    cv2.circle(vis, (int(round(kx)), int(round(ky))), 3, (255, 128, 0), -1)
        if pointer_line is not None:
            cv2.line(vis, (int(pointer_line[0]), int(pointer_line[1])),
                     (int(pointer_line[2]), int(pointer_line[3])), (255, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(tip[0])), int(round(tip[1]))), 5, (255, 0, 0), -1)
        cv2.circle(vis, (int(round(center[0])), int(round(center[1]))), 5, (0, 0, 255), -1)
        cv2.circle(vis, (int(round(start[0])), int(round(start[1]))), 5, (235, 242, 15), -1)
        cv2.circle(vis, (int(round(end[0])), int(round(end[1]))), 5, (235, 242, 15), -1)
        cv2.putText(vis, f"ratio={ratio:.4f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return {
        "status": "readable",
        "reason": "ok",
        "ratio": ratio,
        "value_unit_scale": get_scale_value(ratio),
        "needle_method": needle_method,
        "pointer_line": pointer_line,
        "keypoints": {"center": center.tolist(), "tip": tip.tolist(),
                      "start": start.tolist(), "end": end.tolist()},
        "objects": [{"label": CLASS_NAMES[o.label], "prob": o.prob,
                     "rect": o.rect.tolist(), "kps": o.kps} for o in objs],
        "vis": vis,
    }


def read_image(det_net: Net, pose_net: Net, image_bgr: np.ndarray,
               relax_gate: bool = False, want_vis: bool = True):
    """整图流程：NanoDet 定位表盘 → 裁剪 → 关键点 + 细化 → 读数。

    relax_gate=False（默认）严格复刻上游 isValidROI（宽高比 ∈ [0.90, 1/0.90]）。
    relax_gate=True 时对越界框围绕中心扩成正方形再试 —— 上游会直接拒识，
    放宽后能出读数，但实测存在"自信读错"的样本，故默认关闭，仅作诊断。
    """
    meters = detect_meters(det_net, image_bgr)
    if not meters:
        return {"status": "no_meter", "reason": "NanoDet found no meter", "meters": [], "reads": []}

    H, W = image_bgr.shape[:2]
    strict, relaxed = [], []
    for m in meters:
        w, h = float(m.rect[2]), float(m.rect[3])
        if w <= 0 or h <= 0:
            continue
        ar = w / h
        if ASPECT_RATIO_THRESHOLD <= ar <= 1.0 / ASPECT_RATIO_THRESHOLD:
            strict.append((m, False))
        else:
            relaxed.append((m, True))

    chosen = list(strict)
    if not chosen and relax_gate:
        chosen = list(relaxed)

    if not chosen:
        return {"status": "pointer_outside_scale",
                "reason": "ROI aspect ratio out of bounds (upstream isValidROI)",
                "meters": [{"rect": m.rect.tolist(), "score": m.prob} for m in meters],
                "reads": []}

    reads = []
    for m, was_relaxed in chosen:
        x, y, w, h = (float(m.rect[0]), float(m.rect[1]), float(m.rect[2]), float(m.rect[3]))
        if was_relaxed:
            cx, cy = x + w / 2.0, y + h / 2.0
            side = max(w, h)
            x, y, w, h = cx - side / 2.0, cy - side / 2.0, side, side
        # 取整必须与上游一致：(int) 截断，且 x1 = (int)x + (int)w（不是 round(x+w)）
        xi, yi, wi, hi = int(x), int(y), int(w), int(h)
        x0, y0 = max(0, xi), max(0, yi)
        x1, y1 = min(W, xi + wi), min(H, yi + hi)
        if x1 <= x0 or y1 <= y0:
            continue
        crop = image_bgr[y0:y1, x0:x1].copy()
        r = read_meter(pose_net, crop, want_vis=want_vis)
        r["meter_box"] = {"rect": m.rect.tolist(), "score": m.prob}
        r["crop_origin"] = [x0, y0]
        r["gate_relaxed"] = bool(was_relaxed)
        reads.append(r)

    if not reads:
        return {"status": "unsupported", "reason": "no valid crop",
                "meters": [{"rect": m.rect.tolist(), "score": m.prob} for m in meters], "reads": []}

    return {"status": "ok",
            "meters": [{"rect": m.rect.tolist(), "score": m.prob} for m in meters],
            "reads": reads}
