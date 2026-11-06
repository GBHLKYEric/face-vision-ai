"""人脸对齐: 用 5 个关键点把任意角度、大小的人脸"摆正"成 112x112 的标准脸。

为什么要对齐? 识别网络在训练时看到的都是眼睛、鼻子、嘴巴位置固定的标准脸,
推理时也必须把脸摆成同样的姿势, 否则同一个人的特征会差很多。

做法: 求一个相似变换 (旋转 + 等比缩放 + 平移, 4 个自由度),
使检测到的 5 个点经过变换后与标准模板的 5 个点尽量重合 (最小二乘意义下)。
这个最小二乘问题有闭式解, 即 Umeyama (1991) 算法, 下面手写实现。
"""
from __future__ import annotations

import cv2
import numpy as np

# ArcFace 标准模板: 112x112 图中 左眼、右眼、鼻尖、左嘴角、右嘴角 的位置
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """求相似变换矩阵 T (3x3), 使得 dst ≈ T @ src。

    步骤 (都是线性代数):
      1. 两组点各自减去重心 -> 去掉平移
      2. 协方差矩阵 Σ = dstᵀ·src / n, 做 SVD: Σ = U·S·Vᵀ
      3. 最优旋转 R = U·D·Vᵀ (D 用来防止出现镜像翻转)
      4. 最优缩放 s = trace(S·D) / src 的方差
      5. 平移 t = dst重心 - s·R·src重心
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_c, dst_c = src - src_mean, dst - dst_mean

    cov = dst_c.T @ src_c / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(dim)
    # 若 U·Vᵀ 是镜像 (行列式为 -1), 把最后一个奇异方向反号, 保证得到真正的旋转。
    # 用 det(U)·det(Vᵀ) 判断而不是 det(cov): 只有 2 个点时 cov 秩为 1、行列式恒为 0, 会漏判
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    if estimate_scale:
        scale = (S * d).sum() / src_c.var(0).sum()
    else:
        scale = 1.0
    T = np.eye(dim + 1)
    T[:dim, :dim] = scale * R
    T[:dim, dim] = dst_mean - scale * R @ src_mean
    return T


def estimate_norm(kps: np.ndarray, image_size: int = 112) -> np.ndarray:
    """返回 2x3 仿射矩阵 M, 把原图坐标映射到 image_size 的对齐图上。"""
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio, diff_x = image_size / 112.0, 0.0
    else:
        ratio, diff_x = image_size / 128.0, 8.0 * image_size / 128.0
    dst = ARCFACE_TEMPLATE * ratio
    dst[:, 0] += diff_x
    return umeyama(kps, dst)[:2].astype(np.float32)


def norm_crop(img: np.ndarray, kps: np.ndarray, image_size: int = 112):
    """对齐并裁剪出标准人脸, 同时返回变换矩阵 (方便把结果贴回原图)。"""
    M = estimate_norm(kps, image_size)
    aligned = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return aligned, M


def crop_transform(center, output_size: int, scale: float, rotation_deg: float = 0.0) -> np.ndarray:
    """以 center 为中心, 缩放 scale 倍、旋转 rotation 度, 输出 output_size 的裁剪矩阵 (2x3)。

    关键点模型、性别年龄模型都用这种"以框中心裁一块正方形"的方式取输入。
    """
    rot = np.deg2rad(rotation_deg)
    cos, sin = np.cos(rot) * scale, np.sin(rot) * scale
    cx, cy = center
    # 先平移到原点 -> 缩放旋转 -> 再平移到输出图中心
    M = np.array([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=np.float64)
    M[:, 2] = np.array([output_size / 2, output_size / 2]) - M[:, :2] @ np.array([cx, cy])
    return M.astype(np.float32)


def transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """对 (N,2) 或 (N,3) 的点做 2x3 仿射变换 (第三维 z 只按缩放比例变化)。"""
    pts = np.asarray(pts, dtype=np.float32)
    out = pts.copy()
    out[:, :2] = pts[:, :2] @ M[:, :2].T + M[:, 2]
    if pts.shape[1] == 3:
        scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
        out[:, 2] = pts[:, 2] * scale
    return out
