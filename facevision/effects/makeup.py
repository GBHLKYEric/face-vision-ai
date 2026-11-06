"""上妆: 口红、腮红、眼线。

口红 —— 为什么不直接把嘴唇涂成纯色?
    纯色会抹掉嘴唇的纹理和高光, 像贴了一块塑料。思路是在 LAB 空间里:
        a, b (颜色)  -> 换成口红的颜色 (保留一点原来的起伏)
        L    (亮度)  -> 保留原来的明暗起伏 (L - 平均L), 只把整体亮度往口红亮度挪一点
    即  L' = L + k·(L_口红 - mean(L)),   a' = a_口红 + 0.3·(a - mean(a)),   b' 同理
    唇纹、高光、阴影都在 L 的起伏里, 所以被保留下来。
    掩码 = 外唇轮廓 - 内唇轮廓 (张嘴时不涂到牙齿), 再轻微羽化。

腮红 —— 柔和的径向渐变
    在两颊中心放一个 2D 高斯 "光斑": w(p) = exp(-½·|A⁻¹(p - c)|²),
    A 的两列是脸坐标系的 X/Y 轴 (投影到图像上) 乘以腮红半径,
    所以光斑是一个随歪头旋转、随转头压扁的椭圆。再乘皮肤掩码, 不会涂到背景上。
    颜色混合: out = (1-α)·原图 + α·腮红色, α = strength·w。

眼线 —— 沿上眼睑画一条抗锯齿粗线, 外眼角向外上方拉出一个小 "眼尾"。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..mesh import MeshFace
from .common import blend, face_frame, fill_mask, roi_from_points


def lip_mask(face: MeshFace, shape, feather: float = 0.025) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """嘴唇掩码 (只在嘴唇 ROI 内), 返回 (mask float32 0~1, roi)。"""
    iod = face.eye_distance()
    outer = face.pts("lips_outer")
    x0, y0, x1, y1 = roi_from_points(outer, shape, pad=0.15 * iod)
    hw = (y1 - y0, x1 - x0)
    if hw[0] <= 0 or hw[1] <= 0:
        return np.zeros((0, 0), np.float32), (x0, y0, x1, y1)
    m = fill_mask(hw, [outer], (x0, y0))
    fill_mask(hw, [face.pts("lips_inner")], (x0, y0), value=0, mask=m)
    m = m.astype(np.float32) / 255.0
    m = cv2.GaussianBlur(m, (0, 0), max(0.6, feather * iod))
    return np.clip(m, 0, 1), (x0, y0, x1, y1)


def lipstick(img: np.ndarray, face: MeshFace, color=(60, 30, 200), strength: float = 0.6) -> np.ndarray:
    """口红 (原地修改并返回 img)。color 是 BGR。"""
    if strength <= 0:
        return img
    m, (x0, y0, x1, y1) = lip_mask(face, img.shape)
    if m.size == 0 or m.max() < 1e-3:
        return img
    crop = img[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    w = m / m.sum()
    mean = (lab * w[..., None]).sum((0, 1))             # 嘴唇区域的加权平均 L, a, b
    new = np.empty_like(lab)
    new[..., 0] = lab[..., 0] + 0.5 * (target[0] - mean[0])           # 亮度: 保留起伏, 整体挪一半
    new[..., 1] = target[1] + 0.3 * (lab[..., 1] - mean[1])           # 色度: 换成口红色
    new[..., 2] = target[2] + 0.3 * (lab[..., 2] - mean[2])
    colored = cv2.cvtColor(np.clip(new, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    img[y0:y1, x0:x1] = blend(crop, colored, strength * m)
    return img


def blush(img: np.ndarray, face: MeshFace, color=(150, 120, 245), strength: float = 0.5,
          skin: np.ndarray | None = None, radius: tuple[float, float] = (0.42, 0.30)) -> np.ndarray:
    """腮红 (原地修改并返回 img)。radius = 椭圆半轴 (IOD 单位), skin = 可选皮肤掩码 (H, W)。"""
    if strength <= 0:
        return img
    _, X, Y, _ = face_frame(face)
    A = np.column_stack([X[:2] * radius[0], Y[:2] * radius[1]])   # 2x2: 脸坐标 -> 图像
    if abs(np.linalg.det(A)) < 1e-6:
        return img
    A_inv = np.linalg.inv(A).astype(np.float32)
    for side, outer in (("left_cheek", 234), ("right_cheek", 454)):
        # 中心: 脸颊一圈的中心稍微往外 (颧骨方向) 挪一点
        c = face.center(side) * 0.85 + face.xy[outer] * 0.15
        ext = np.linalg.norm(A, axis=1) * 2.2                          # 椭圆外接框, 2.2σ 以外权重 < 0.09, 忽略
        x0, y0, x1, y1 = roi_from_points(np.array([c - ext, c + ext]), img.shape)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(c[0])   # (1, w)
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(c[1])   # (h, 1)
        u = A_inv[0, 0] * dx + A_inv[0, 1] * dy                        # 变到 "腮红坐标" (广播成 h×w)
        v = A_inv[1, 0] * dx + A_inv[1, 1] * dy
        wgt = np.exp(-0.5 * (u * u + v * v)).astype(np.float32)
        if skin is not None:
            wgt *= skin[y0:y1, x0:x1]
        crop = img[y0:y1, x0:x1]
        tint = np.empty_like(crop)
        tint[:] = color
        img[y0:y1, x0:x1] = blend(crop, tint, 0.45 * strength * wgt)
    return img


def eyeliner(img: np.ndarray, face: MeshFace, color=(25, 20, 20), strength: float = 0.8) -> np.ndarray:
    """眼线 (原地修改并返回 img)。"""
    if strength <= 0:
        return img
    iod = face.eye_distance()
    thick = max(1, int(round(0.022 * iod)))
    for side in ("left_upper_lid", "right_upper_lid"):
        lid = face.pts(side)                      # 外眼角 -> 内眼角
        outer, nxt = lid[0], lid[1]
        # 眼尾: 从外眼角沿 "外眼角 - 相邻点" 方向延长, 再往上翘一点
        out_dir = outer - nxt
        out_dir /= np.linalg.norm(out_dir) + 1e-6
        _, X, Y, _ = face_frame(face)
        up = -Y[:2] / (np.linalg.norm(Y[:2]) + 1e-6)
        wing = outer + (out_dir * 0.9 + up * 0.35) * 0.12 * iod
        line = np.vstack([wing, lid])
        x0, y0, x1, y1 = roi_from_points(line, img.shape, pad=thick * 3)
        if x1 <= x0 or y1 <= y0:
            continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts = np.round((line - [x0, y0]) * 4).astype(np.int32)   # shift=2: 1/4 像素精度画线
        cv2.polylines(m, [pts], False, 255, thick, cv2.LINE_AA, shift=2)
        m = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0), 0.6)
        crop = img[y0:y1, x0:x1]
        tint = np.empty_like(crop)
        tint[:] = color
        img[y0:y1, x0:x1] = blend(crop, tint, strength * m)
    return img
