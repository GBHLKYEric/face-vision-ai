"""性别、年龄估计 (InsightFace genderage.onnx, 在 CelebA 上训练)。"""
from __future__ import annotations

import cv2
import numpy as np

from .align import crop_transform
from .landmark import guess_normalization
from .onnx_utils import create_session
from .paths import GENDERAGE_MODEL


class GenderAge:
    def __init__(self, model_path=GENDERAGE_MODEL, providers=None, threads=None):
        self.session = create_session(model_path, providers, threads)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = int(inp.shape[2])  # 96
        self.output_name = self.session.get_outputs()[0].name
        self.mean, self.std = guess_normalization(model_path)

    def predict(self, img: np.ndarray, bbox) -> tuple[str, int]:
        x1, y1, x2, y2 = bbox[:4]
        scale = self.input_size / (max(x2 - x1, y2 - y1) * 1.5)
        M = crop_transform(((x1 + x2) / 2, (y1 + y2) / 2), self.input_size, scale)
        crop = cv2.warpAffine(img, M, (self.input_size, self.input_size), borderValue=0.0)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / self.std, (self.input_size,) * 2, (self.mean,) * 3, swapRB=True)
        pred = self.session.run([self.output_name], {self.input_name: blob})[0][0]
        gender = "M" if np.argmax(pred[:2]) == 1 else "F"
        age = int(np.round(pred[2] * 100))
        return gender, age
