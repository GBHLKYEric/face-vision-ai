"""完整的人脸分析流水线: 检测 -> 关键点 -> 对齐 -> 特征 -> 属性。"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .align import norm_crop
from .attributes import GenderAge
from .detector import SCRFD
from .landmark import Landmark106, Landmark3D68
from .recognizer import ArcFaceONNX


@dataclass
class Face:
    bbox: np.ndarray                  # x1, y1, x2, y2
    score: float
    kps: np.ndarray                   # 5 点 (左眼, 右眼, 鼻尖, 左嘴角, 右嘴角)
    landmark_106: np.ndarray | None = None
    landmark_3d_68: np.ndarray | None = None
    embedding: np.ndarray | None = None
    gender: str | None = None
    age: int | None = None
    extra: dict = field(default_factory=dict)


class FaceAnalyzer:
    """modules 可选: "landmark_106", "landmark_3d_68", "recognition", "genderage"。"""

    def __init__(self, modules=("landmark_106", "recognition"), det_size=(640, 640),
                 det_thresh=0.5, providers=None, threads=None):
        kw = dict(providers=providers, threads=threads)
        self.detector = SCRFD(input_size=det_size, conf_thresh=det_thresh, **kw)
        self.lmk106 = Landmark106(**kw) if "landmark_106" in modules else None
        self.lmk3d = Landmark3D68(**kw) if "landmark_3d_68" in modules else None
        self.recognizer = ArcFaceONNX(**kw) if "recognition" in modules else None
        self.genderage = GenderAge(**kw) if "genderage" in modules else None

    def analyze(self, img: np.ndarray, max_num: int = 0, flip_test: bool = True) -> list[Face]:
        dets, kpss = self.detector.detect(img, max_num=max_num)
        faces = []
        for det, kps in zip(dets, kpss):
            face = Face(bbox=det[:4], score=float(det[4]), kps=kps)
            if self.lmk106:
                face.landmark_106 = self.lmk106.predict(img, face.bbox)
            if self.lmk3d:
                face.landmark_3d_68 = self.lmk3d.predict(img, face.bbox)
            if self.recognizer:
                aligned, _ = norm_crop(img, kps)
                face.embedding = self.recognizer.embed(aligned, flip_test=flip_test)[0]
            if self.genderage:
                face.gender, face.age = self.genderage.predict(img, face.bbox)
            faces.append(face)
        return faces
