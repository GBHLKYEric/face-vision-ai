"""特效模块共用的小工具: 人脸局部坐标系、ROI、多边形掩码、按掩码混合。

关键思想 —— "以脸为尺子":
    特效的位置和大小都用 "两眼距离 (IOD, inter-ocular distance)" 做单位,
    这样同一套参数对大脸/小脸、640x480/1280x720 都适用。
"""
from __future__ import annotations

import cv2
import numpy as np

from ..mesh import KEY, MESH, MeshFace


def face_frame(face: MeshFace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """建立 "脸的局部坐标系": 原点 O 和三根轴 X/Y/Z (都在 3D 像素坐标里, 图像 x 右、y 下、z 向画面里)。

        O: 两眼中心的中点 (来自关键点)
        X: 脸的水平方向 (左眼 -> 右眼)       |
        Y: 脸的竖直方向 (向下, 额头 -> 下巴)   | 方向来自头部姿态矩阵的旋转 R, 长度都 = 1 个 IOD
        Z: 脸的前后方向 (指向脑后)            |
    脸上任意一点都可以写成 O + u·X + v·Y + w·Z, (u, v, w) 就是它的 "脸坐标",
    不随平移、缩放、转头而变 —— 贴纸、腮红都在这个坐标系里定义。

    为什么方向用 R 而不直接用 "额头->下巴" 向量? 因为张嘴时下巴会动, 那样帽子会跟着嘴上下跳;
    R 是 MediaPipe 用全部 468 个点拟合出来的刚体旋转, 不受表情影响, 稳定得多。
    MediaPipe 相机坐标是 x 右、y 上、z 指向观众, 换到图像坐标 (y 下、z 向里) 要把 y、z 取反。
    """
    P = face.landmarks.astype(np.float64)
    le = P[MESH["left_eye"]].mean(0)
    re = P[MESH["right_eye"]].mean(0)
    O = (le + re) / 2
    iod = float(np.linalg.norm(re - le)) + 1e-6
    R = np.asarray(face.matrix, dtype=np.float64)[:3, :3]
    if np.allclose(R, np.eye(3)):           # 没有姿态矩阵时退化为用关键点估计
        xn = (re - le) / iod
        d = P[KEY["chin"]] - P[KEY["forehead_top"]]
        y = d - (d @ xn) * xn                # Gram-Schmidt 正交化
        yn = y / (np.linalg.norm(y) + 1e-6)
        zn = np.cross(xn, yn)
    else:
        flip = np.array([1.0, -1.0, -1.0])   # 相机坐标 -> 图像坐标
        xn = R[:, 0] * flip                  # 模型 +x (脸的水平方向)
        yn = -R[:, 1] * flip                 # 模型 +y 是 "向上", 取反得到 "向下"
        zn = -R[:, 2] * flip                 # 模型 +z 是 "脸的正前方", 取反得到 "脑后"
    return O, xn * iod, yn * iod, zn * iod


def face_to_image(face_pts_uvw: np.ndarray, O, X, Y, Z) -> np.ndarray:
    """脸坐标 (N,3) 的 (u,v,w) -> 3D 像素坐标 (N,3)。"""
    uvw = np.asarray(face_pts_uvw, dtype=np.float64).reshape(-1, 3)
    return O + uvw[:, :1] * X + uvw[:, 1:2] * Y + uvw[:, 2:3] * Z


def roi_from_points(pts: np.ndarray, shape, pad: float = 0.0) -> tuple[int, int, int, int]:
    """包住一组 2D 点的矩形 (向外扩 pad 像素), 裁剪到图像范围内。返回 x0, y0, x1, y1。"""
    h, w = shape[:2]
    x0, y0 = np.floor(pts.min(0) - pad).astype(int)
    x1, y1 = np.ceil(pts.max(0) + pad).astype(int)
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def face_roi(face: MeshFace, shape, pad: float = 0.3) -> tuple[int, int, int, int]:
    """整张脸的 ROI (感兴趣区域), pad 以 IOD 为单位。只在 ROI 里计算可以省很多时间。"""
    return roi_from_points(face.xy, shape, pad * face.eye_distance())


def fill_mask(hw: tuple[int, int], polys: list[np.ndarray], offset=(0, 0), value: int = 255,
              mask: np.ndarray | None = None) -> np.ndarray:
    """把若干多边形 (像素坐标) 填进一个 uint8 掩码; offset 是 ROI 左上角。"""
    if mask is None:
        mask = np.zeros(hw, np.uint8)
    pts = [np.round(p - np.asarray(offset)).astype(np.int32) for p in polys]
    cv2.fillPoly(mask, pts, value, cv2.LINE_AA)
    return mask


def blend(dst: np.ndarray, src: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """按逐像素权重混合: out = dst·(1-α) + src·α。dst/src 为 uint8 或 float, α 形状 (h,w), 取值 0~1。"""
    a = alpha[..., None].astype(np.float32)
    out = dst.astype(np.float32) * (1 - a) + src.astype(np.float32) * a
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
