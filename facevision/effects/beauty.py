"""美颜: 磨皮、美白、瘦脸、大眼。

1. 皮肤掩码 skin_mask
   脸部外轮廓 (face_oval) 填满 = 1, 再把眼睛、眉毛、嘴唇挖掉 = 0, 最后高斯模糊 "羽化" 边缘,
   得到 0~1 的软掩码。后面所有颜色处理都按它加权, 所以五官保持清晰, 边界不会出现硬接缝。

2. 磨皮 smooth_skin —— 保边平滑
   普通高斯模糊会把所有边缘一起抹掉 (脸变成一团)。"保边滤波" 只抹平小起伏 (痘印、毛孔), 保留大边缘:
   - 双边滤波: 邻居的权重 = 空间高斯 × 颜色高斯, 颜色差太大 (跨过边缘) 的邻居权重≈0。
   - 导向滤波 (He et al. 2010): 假设输出在每个小窗口里是引导图 I 的线性函数 q = a·I + b,
         a = var(I) / (var(I) + ε),   b = mean(p) - a·mean(I)
     平坦区域 var(I) << ε -> a≈0, q≈均值 (被抹平);
     边缘处 var(I) >> ε -> a≈1, q≈I (被保留)。
     ε 就是 "多大的起伏算瑕疵" 的阈值; 只用盒式滤波实现, 复杂度 O(N) 与半径无关, 很快。
   结果按 strength·mask 与原图混合。

3. 美白 whiten —— LAB 空间的亮度曲线
   LAB 把颜色拆成 亮度 L 和两个色度 a (绿-红), b (蓝-黄)。只改 L 就能变亮而不偏色。
   用对数曲线  L' = 255 · log(1 + (β-1)·L/255) / log(β)   (β>1)
   它把 0 映射到 0、255 映射到 255, 中间调被抬高; 做成 256 项查找表 (LUT), 每个像素查表即可。

4. 瘦脸 / 大眼 —— Gustafsson 局部变形 (Interactive Image Warping, 1993)
   都用 "反向映射": 对输出图的每个像素 x, 算出它应该从原图哪个位置 u 取颜色, 然后 cv2.remap 插值。
   (正向映射会留下空洞, 反向映射不会。)
   - 局部平移 (瘦脸): 圆心 c、半径 r, 想把 c 处的内容往 m 推:
         u = x - ((r² - |x-c|²) / (r² - |x-c|² + |m-c|²))² · (m - c)      (|x-c| < r)
     圆心处移动最多, 越靠近圆周移动越少, 圆外不动 -> 变形平滑无接缝。
     沿下颌线取几个点, 都往脸中心推, 脸就 "瘦" 了。
   - 局部缩放 (大眼): 以眼睛中心 c、半径 r 的圆内:
         u = c + (x - c) · (1 - a·(|x-c|/r - 1)²)       (a ∈ [0,1) 为强度)
     圆心附近从更靠近中心的地方取像素 = 放大; 圆周上系数 = 1 不变形。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..mesh import KEY, MESH, MeshFace
from .common import blend, face_roi, fill_mask


# ---------------------------------------------------------------------------
# 皮肤掩码
# ---------------------------------------------------------------------------
def skin_mask(face: MeshFace, shape, feather: float = 0.12) -> np.ndarray:
    """返回整幅图大小的 float32 掩码 (H, W), 皮肤≈1, 五官和背景≈0。feather 以 IOD 为单位。"""
    h, w = shape[:2]
    iod = face.eye_distance()
    x0, y0, x1, y1 = face_roi(face, shape, pad=0.3)
    full = np.zeros((h, w), np.float32)
    if x1 <= x0 or y1 <= y0:
        return full
    off = (x0, y0)
    hw = (y1 - y0, x1 - x0)
    # 脸部外轮廓: 大半径羽化, 和头发/背景之间过渡柔和
    oval = fill_mask(hw, [face.pts("face_oval")], off).astype(np.float32) / 255.0
    oval = cv2.GaussianBlur(oval, (0, 0), max(1.0, feather * iod))
    # 五官: 先填多边形, 再沿边描一圈粗线 (相当于向外扩一点), 小半径羽化 -> 五官内部严格为 0, 不会被磨糊
    holes = [face.pts(g) for g in ("left_eye", "right_eye", "left_brow", "right_brow", "lips_outer")]
    hole = fill_mask(hw, holes, off)
    cv2.polylines(hole, [np.round(p - off).astype(np.int32) for p in holes], True, 255,
                  max(2, int(0.16 * iod)), cv2.LINE_AA)
    hole = cv2.GaussianBlur(hole.astype(np.float32) / 255.0, (0, 0), max(0.8, 0.04 * iod))
    full[y0:y1, x0:x1] = np.clip(oval * (1.0 - hole), 0.0, 1.0)
    return full


# ---------------------------------------------------------------------------
# 磨皮 / 美白
# ---------------------------------------------------------------------------
def guided_filter_self(p: np.ndarray, r: int, eps: float, sub: int = 2) -> np.ndarray:
    """自引导的导向滤波 (引导图 I = 输入 p, 每个通道独立), p 为 float32, 取值 0~1。

    公式 (每个 (2r+1)² 窗口内): a = var(p) / (var(p) + ε),  b = (1 - a)·mean(p),  q = mean(a)·p + mean(b)
    只用到盒式滤波 (窗口均值), 所以和半径无关、很快。
    sub>1 时用 "快速导向滤波" (He & Sun 2015): 在缩小 sub 倍的图上算 a、b, 放大回来再算 q,
    a、b 本身很平滑, 缩小计算几乎不损失质量, 速度快 sub² 倍。
    """
    h, w = p.shape[:2]
    small = cv2.resize(p, (max(1, w // sub), max(1, h // sub)), interpolation=cv2.INTER_AREA) if sub > 1 else p
    rs = max(1, r // sub)
    box = lambda x: cv2.boxFilter(x, -1, (2 * rs + 1, 2 * rs + 1))  # noqa: E731  窗口均值
    mean = box(small)
    var = box(small * small) - mean * mean
    a = var / (var + eps)
    b = (1.0 - a) * mean
    mean_a, mean_b = box(a), box(b)
    if sub > 1:
        mean_a = cv2.resize(mean_a, (w, h), interpolation=cv2.INTER_LINEAR)
        mean_b = cv2.resize(mean_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return mean_a * p + mean_b


def smooth_skin(img: np.ndarray, face: MeshFace, strength: float = 0.7, method: str = "guided",
                mask: np.ndarray | None = None, eps: float = 0.03 ** 2) -> np.ndarray:
    """保边磨皮 (原地修改并返回 img)。method: "guided" (导向滤波) 或 "bilateral" (双边滤波)。

    eps: 导向滤波的 ε (像素归一化到 0~1), 标准差小于 sqrt(ε)≈0.03 (约 8 个灰度级) 的起伏被抹平。
    """
    if strength <= 0:
        return img
    mask = skin_mask(face, img.shape) if mask is None else mask
    x0, y0, x1, y1 = face_roi(face, img.shape, pad=0.3)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img
    iod = face.eye_distance()
    r = max(2, int(round(0.10 * iod)))           # 滤波半径跟着脸的大小走
    if method == "guided":
        sub = 2 if r >= 6 else 1
        smooth = guided_filter_self(crop.astype(np.float32) / 255.0, r, eps, sub) * 255.0
    else:
        smooth = cv2.bilateralFilter(crop, d=2 * r + 1, sigmaColor=35, sigmaSpace=r)
    img[y0:y1, x0:x1] = blend(crop, smooth, strength * mask[y0:y1, x0:x1])
    return img


def whiten_lut(strength: float) -> np.ndarray:
    """对数提亮曲线做成的 256 项查找表。strength=0 时是恒等映射。"""
    x = np.arange(256, dtype=np.float64)
    beta = 1.0 + 4.0 * strength
    if beta <= 1.0001:
        return x.astype(np.uint8)
    y = 255.0 * np.log1p((beta - 1) * x / 255.0) / np.log(beta)
    return np.clip(y + 0.5, 0, 255).astype(np.uint8)


def whiten(img: np.ndarray, face: MeshFace, strength: float = 0.4,
           mask: np.ndarray | None = None) -> np.ndarray:
    """美白: 只提亮 LAB 的 L 通道, 并按皮肤掩码混合 (原地修改并返回 img)。"""
    if strength <= 0:
        return img
    mask = skin_mask(face, img.shape) if mask is None else mask
    x0, y0, x1, y1 = face_roi(face, img.shape, pad=0.3)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return img
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.LUT(lab[..., 0], whiten_lut(strength))
    # 顺便把色度往中性 (128) 收一点点, 减少泛红/泛黄, 看起来更 "透"
    ab = lab[..., 1:].astype(np.float32)
    lab[..., 1:] = np.clip(128 + (ab - 128) * (1 - 0.15 * strength), 0, 255).astype(np.uint8)
    bright = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img[y0:y1, x0:x1] = blend(crop, bright, mask[y0:y1, x0:x1])
    return img


# ---------------------------------------------------------------------------
# 局部变形 (Gustafsson): 在 "坐标图" 上操作, 最后统一 remap 一次
# ---------------------------------------------------------------------------
def local_translate(map_x: np.ndarray, map_y: np.ndarray, c, m, r: float, strength: float = 1.0):
    """局部平移变形, 原地修改坐标图 map_x/map_y (它们存的是 "去原图哪里取像素")。

    注意 |m-c| 和 r 的比例决定了衰减曲线的形状:
      |m-c| << r 时, 圆内几乎整体平移, 只在圆周附近极窄的一圈里突然降到 0 -> 看得见 "台阶";
      |m-c| ≈ r 时, 从圆心到圆周平滑衰减。
    所以这里固定让 |m-c| 与 r 同量级, 再用 strength (0~1) 把位移整体按比例缩小来控制力度
    (位移场整体缩小不会产生折叠, strength=1 就是 Gustafsson 原公式)。
    """
    cx, cy = c
    mx, my = m
    dx, dy = map_x - cx, map_y - cy
    d2 = dx * dx + dy * dy
    r2 = r * r
    mc2 = (mx - cx) ** 2 + (my - cy) ** 2
    inside = d2 < r2
    ratio = np.zeros_like(d2)
    ratio[inside] = ((r2 - d2[inside]) / (r2 - d2[inside] + mc2)) ** 2
    map_x -= strength * ratio * (mx - cx)
    map_y -= strength * ratio * (my - cy)


def local_scale(map_x: np.ndarray, map_y: np.ndarray, c, r: float, a: float):
    """局部缩放变形 (a>0 放大, a<0 缩小), 原地修改坐标图。"""
    cx, cy = c
    dx, dy = map_x - cx, map_y - cy
    d = np.sqrt(dx * dx + dy * dy)
    t = d / r
    k = np.where(t < 1.0, 1.0 - a * (t - 1.0) ** 2, 1.0)
    map_x[:] = cx + dx * k
    map_y[:] = cy + dy * k


def warp_face(img: np.ndarray, face: MeshFace, slim: float = 0.0, eyes: float = 0.0) -> np.ndarray:
    """瘦脸 + 大眼, 两种变形叠加到同一张坐标图上, 只做一次 remap (只插值一次, 画面更清晰)。

    slim / eyes: 强度 0~1, 0 表示不变形。
    """
    if slim <= 0 and eyes <= 0:
        return img
    iod = face.eye_distance()
    # 先列出所有变形 (圆心, 半径, 参数), 再统一处理
    warps: list[tuple] = []
    if slim > 0:
        # 下颌线上 3 个点往 "鼻尖与下巴中点" 方向拖; 越靠近耳朵拖得越多, 下巴附近少拖。
        # 拖动距离 |m-c| = r (衰减平滑), 实际位移再乘 strength; 圆心处最大位移 = 0.25·r·strength。
        target = (face.xy[KEY["nose_tip"]] + face.xy[KEY["chin"]]) / 2
        r = 0.45 * iod
        for side in ("left_jaw", "right_jaw"):
            idx = [MESH[side][k] for k in (1, 3, 4)]     # 下颌角上方 / 下颌角 / 下颌中段
            for i, wgt in zip(idx, (1.0, 0.9, 0.6)):
                c = face.xy[i].astype(np.float64)
                v = target - c
                m = c + v / (np.linalg.norm(v) + 1e-6) * r
                warps.append(("translate", c, r, (m, 0.5 * slim * wgt)))
    if eyes > 0:
        for eye in ("left_eye", "right_eye"):
            pts = face.pts(eye)
            eye_w = np.linalg.norm(pts[0] - pts[8])       # 外眼角 -> 内眼角
            warps.append(("scale", pts.mean(0).astype(np.float64), 1.1 * eye_w, 0.4 * eyes))

    # 只有圆内的像素会动, 所以 ROI = 所有圆的外接矩形 (ROI 太小会在边界留下 "断层")
    lo = np.min([c - r for _, c, r, _ in warps], axis=0)
    hi = np.max([c + r for _, c, r, _ in warps], axis=0)
    h, w = img.shape[:2]
    x0, y0 = max(0, int(np.floor(lo[0])) - 1), max(0, int(np.floor(lo[1])) - 1)
    x1, y1 = min(w, int(np.ceil(hi[0])) + 2), min(h, int(np.ceil(hi[1])) + 2)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return img
    # 坐标图初始化为恒等映射 (每个像素从自己的位置取色), 用整幅图的绝对坐标
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    map_x, map_y = xs.copy(), ys.copy()
    for kind, c, r, arg in warps:
        if kind == "translate":
            local_translate(map_x, map_y, c, arg[0], r, strength=arg[1])
        else:
            local_scale(map_x, map_y, c, r, arg)
    img[y0:y1, x0:x1] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return img


def slim_face(img: np.ndarray, face: MeshFace, strength: float = 0.5) -> np.ndarray:
    return warp_face(img, face, slim=strength)


def big_eyes(img: np.ndarray, face: MeshFace, strength: float = 0.5) -> np.ndarray:
    return warp_face(img, face, eyes=strength)
