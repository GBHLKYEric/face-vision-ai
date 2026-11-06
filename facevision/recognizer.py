"""人脸识别: ArcFace 特征提取。

对齐后的 112x112 人脸 -> ResNet50 -> 512 维特征向量 (embedding)。
同一个人的两张照片, 特征向量方向接近 (余弦相似度高); 不同人则方向差得远。
"""
from __future__ import annotations

import cv2
import numpy as np

from .onnx_utils import create_session
from .paths import REC_MODEL


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """把向量缩放到长度 1, 之后两向量的点积就等于余弦相似度。"""
    return x / np.linalg.norm(x, axis=axis, keepdims=True).clip(min=1e-12)


class ArcFaceONNX:
    def __init__(self, model_path=REC_MODEL, providers=None, threads=None):
        self.session = create_session(model_path, providers, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = (int(inp.shape[3]), int(inp.shape[2]))  # (112, 112)
        self.output_name = self.session.get_outputs()[0].name

    def _blob(self, faces: list[np.ndarray]) -> np.ndarray:
        # ArcFace 训练时的归一化: (x - 127.5) / 127.5 -> [-1, 1], 并转为 RGB
        return cv2.dnn.blobFromImages(faces, 1.0 / 127.5, self.input_size, (127.5,) * 3, swapRB=True)

    def embed(self, aligned_faces, flip_test: bool = True, normalize: bool = True) -> np.ndarray:
        """aligned_faces: 一张或一批 112x112 BGR 对齐人脸。

        flip_test: 把水平翻转的图也提一次特征, 两者相加。人脸左右近似对称,
        这样相当于免费做了一次"测试时增强", LFW 上通常能再提升 0.05~0.1 个点。
        """
        if isinstance(aligned_faces, np.ndarray) and aligned_faces.ndim == 3:
            aligned_faces = [aligned_faces]
        feats = self.session.run([self.output_name], {self.input_name: self._blob(list(aligned_faces))})[0]
        if flip_test:
            flipped = [cv2.flip(f, 1) for f in aligned_faces]
            feats = feats + self.session.run([self.output_name], {self.input_name: self._blob(flipped)})[0]
        return l2_normalize(feats) if normalize else feats


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = l2_normalize(np.asarray(a).ravel()), l2_normalize(np.asarray(b).ravel())
    return float(a @ b)
