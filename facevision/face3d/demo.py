"""给 Gradio 界面用的一站式函数: 一张图 -> (网格叠加图, 多视角图, .obj 路径)。"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..paths import OUTPUTS
from . import io, render
from .tddfa import TDDFA3D

UI_DIR = OUTPUTS / "face3d" / "ui"
_models: dict = {}


def _get_models():
    """第一次调用时才加载模型 (检测器 + 3DDFA), 之后复用。"""
    if not _models:
        from ..detector import SCRFD
        _models["det"] = SCRFD()
        _models["tddfa"] = TDDFA3D()
    return _models["det"], _models["tddfa"]


def reconstruct_for_ui(img_bgr: np.ndarray, max_faces: int = 5, view_size: int = 192):
    """返回 (overlay_bgr, multiview_grid_bgr, obj_path_str)。

    overlay: 所有人脸的网格叠加 + 头部姿态坐标轴; multiview: 最大那张脸"摆正"后
    在 yaw = -60..60 下的纹理渲染 (上排) 和石膏像 (下排); obj: 最大那张脸的带颜色网格。
    没检测到人脸时返回 (原图, 空白图, None)。
    """
    det, tddfa = _get_models()
    dets, _ = det.detect(img_bgr, max_num=max_faces)
    if len(dets) == 0:
        blank = np.full((view_size, view_size * 5, 3), 250, np.uint8)
        render.put_label(blank, "no face detected")
        return img_bgr.copy(), blank, None
    results = [tddfa.fit(img_bgr, box) for box in dets]

    # 小图先放大再画, 网格和文字更清楚
    k = max(1.0, 500 / min(img_bgr.shape[:2]))
    disp = cv2.resize(img_bgr, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1 else img_bgr.copy()
    overlay = render.render_mesh_overlay(disp, [r.vertices * k for r in results], tddfa.tri)
    for r in results:
        lmk = r.landmarks68 * k
        overlay = render.draw_pose_axes(overlay, r.R, lmk[30], np.ptp(lmk[:, 1]) * 0.6)
    if len(results) == 1:
        render.put_label(overlay, "yaw %+.1f  pitch %+.1f  roll %+.1f" % results[0].pose_deg, bottom=True)

    area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    main = results[int(area.argmax())]
    colors = render.texture_from_image(img_bgr, main.vertices, tddfa.tri, tddfa.mirror_index)
    fv = tddfa.frontal_vertices(main)
    rows = []
    for tex in (colors, None):
        tiles = [render.render_view(fv, tddfa.tri, tex, yaw, 0, view_size) for yaw in (-60, -30, 0, 30, 60)]
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)

    UI_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = UI_DIR / f"face_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.obj"
    centered = main.vertices - main.vertices.mean(axis=1, keepdims=True)  # 移到原点, 网页 3D 查看器好对焦
    io.save_obj(obj_path, centered, tddfa.tri, colors)
    return overlay, grid, str(obj_path)
