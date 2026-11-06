"""可视化工具: 画框、关键点、文字。"""
from __future__ import annotations

import cv2
import numpy as np

KPS_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]


def draw_faces(img: np.ndarray, faces, show_106: bool = True, show_kps: bool = True, labels=None) -> np.ndarray:
    out = img.copy()
    scale = max(1, round(max(img.shape[:2]) / 500))
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox.astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), scale + 1)
        if show_106 and f.landmark_106 is not None:
            for x, y in f.landmark_106.astype(int):
                cv2.circle(out, (x, y), scale, (255, 200, 0), -1)
        if show_kps:
            for (x, y), c in zip(f.kps.astype(int), KPS_COLORS):
                cv2.circle(out, (x, y), scale + 2, c, -1)
        text = labels[i] if labels else f"{f.score:.2f}"
        if f.gender is not None and labels is None:
            text += f" {f.gender}/{f.age}"
        put_label(out, text, (x1, y1), scale)
    return out


def put_label(img, text, org, scale=1, color=(0, 220, 0)):
    x, y = org
    fs = 0.5 * scale
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, scale)
    y = max(y, th + 6)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
    cv2.putText(img, text, (x + 2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), scale, cv2.LINE_AA)
