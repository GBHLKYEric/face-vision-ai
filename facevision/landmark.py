"""人脸关键点: 106 点 2D (美颜/贴纸用) 和 68 点 3D (姿态估计/评测用)。

两种模型都是"坐标回归": 网络直接输出每个点的 (x, y[, z]), 取值范围 [-1, 1],
再映射回 192x192 的裁剪图, 最后用逆仿射变换映射回原图。
(另一类方法是"热图回归": 每个点输出一张概率图取峰值, 精度高但更慢, 见教学文档对比。)
"""
from __future__ import annotations

import cv2
import numpy as np
import onnx

from .align import crop_transform, transform_points
from .onnx_utils import create_session
from .paths import LMK106_MODEL, LMK3D68_MODEL


def guess_normalization(model_path) -> tuple[float, float]:
    """InsightFace 的老模型 (MXNet 转来的) 把归一化写进了网络里 (bn_data 层),
    这种模型输入直接给 0~255 像素; 否则需要手动 (x-127.5)/128。"""
    graph = onnx.load(str(model_path)).graph
    find_sub = find_mul = False
    for nid, node in enumerate(graph.node[:8]):
        if node.name.startswith("Sub") or node.name.startswith("_minus"):
            find_sub = True
        if node.name.startswith("Mul") or node.name.startswith("_mul"):
            find_mul = True
        if nid < 3 and node.name == "bn_data":
            find_sub = find_mul = True
    return (0.0, 1.0) if (find_sub and find_mul) else (127.5, 128.0)


class LandmarkModel:
    def __init__(self, model_path, num_points: int, providers=None, threads=None):
        self.session = create_session(model_path, providers, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2])  # 192
        self.output_name = self.session.get_outputs()[0].name
        self.num_points = num_points
        self.mean, self.std = guess_normalization(model_path)

    def predict(self, img: np.ndarray, bbox) -> np.ndarray:
        x1, y1, x2, y2 = bbox[:4]
        w, h = x2 - x1, y2 - y1
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        # 框放大 1.5 倍再缩放到 192, 让下巴、额头也在图里
        scale = self.input_size / (max(w, h) * 1.5)
        M = crop_transform(center, self.input_size, scale)
        crop = cv2.warpAffine(img, M, (self.input_size, self.input_size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / self.std, (self.input_size, self.input_size),
                                     (self.mean,) * 3, swapRB=True)
        pred = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        dim = 3 if pred.shape[0] >= 3000 else 2
        pred = pred.reshape(-1, dim)[-self.num_points:].copy()
        # [-1, 1] -> [0, 192]
        pred[:, :2] = (pred[:, :2] + 1) * (self.input_size // 2)
        if dim == 3:
            pred[:, 2] *= self.input_size // 2
        IM = cv2.invertAffineTransform(M)
        return transform_points(pred, IM)


class Landmark106(LandmarkModel):
    def __init__(self, model_path=LMK106_MODEL, **kw):
        super().__init__(model_path, 106, **kw)


class Landmark3D68(LandmarkModel):
    def __init__(self, model_path=LMK3D68_MODEL, **kw):
        super().__init__(model_path, 68, **kw)


# 106 点里各部位的下标 (InsightFace 2d106 定义, 已逐点画图核对), 按轮廓顺序排列。
# "左/右" 指图像中的左/右。
LMK106 = {
    "contour": [1, 9, 10, 11, 12, 13, 14, 15, 16, 2, 3, 4, 5, 6, 7, 8, 0,
                24, 23, 22, 21, 20, 19, 18, 32, 31, 30, 29, 28, 27, 26, 25, 17],
    "left_brow": [43, 48, 49, 51, 50, 46, 47, 45, 44],
    "right_brow": [97, 98, 99, 100, 101, 105, 104, 103, 102],
    "left_eye": [35, 41, 40, 42, 39, 37, 33, 36],
    "right_eye": [89, 95, 94, 96, 93, 91, 87, 90],
    "left_pupil": [38],
    "right_pupil": [88],
    "nose_bridge": [72, 73, 74, 86],
    "nose_bottom": [75, 76, 77, 78, 79, 80, 85, 84, 83, 82, 81],
    "outer_lip": [52, 64, 63, 71, 67, 68, 61, 58, 59, 53, 56, 55],
    "inner_lip": [65, 66, 62, 70, 69, 57, 60, 54],
}
