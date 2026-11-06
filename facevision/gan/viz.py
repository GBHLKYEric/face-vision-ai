"""把若干行图片拼成带列标题的大图 (训练采样、评测展示共用)。"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _fit(img: np.ndarray, h: int) -> np.ndarray:
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * h / img.shape[0]))
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA if img.shape[0] > h else cv2.INTER_LINEAR)


def save_grid(rows: Sequence[Sequence[np.ndarray]], path: Path | str, titles: Sequence[str] | None = None,
              cell_h: int | None = None, quality: int = 92) -> np.ndarray:
    """rows: 每行若干张 BGR uint8 图 (同一行内高度会统一到 cell_h); titles: 每列的标题。"""
    cell_h = cell_h or rows[0][0].shape[0]
    lines = [np.hstack([_fit(im, cell_h) for im in row]) for row in rows]
    width = max(line.shape[1] for line in lines)
    lines = [np.pad(line, ((0, 0), (0, width - line.shape[1]), (0, 0)), constant_values=255) for line in lines]
    grid = np.vstack(lines)
    if titles:
        bar = np.full((22, width, 3), 255, np.uint8)
        col_w = [_fit(im, cell_h).shape[1] for im in rows[0]]
        x = 0
        for t, w in zip(titles, col_w):
            scale = 0.42 if cell_h >= 120 else 0.33
            (tw, _), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(bar, t, (x + max(2, (w - tw) // 2), 15), cv2.FONT_HERSHEY_SIMPLEX, scale,
                        (30, 30, 30), 1, cv2.LINE_AA)
            x += w
        grid = np.vstack([bar, grid])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return grid
