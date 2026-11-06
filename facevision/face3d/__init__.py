"""3D 人脸重建: 单张图 -> 3DMM 参数 -> 稠密 3D 网格 -> 渲染 / 多视角可视化。

    from facevision.face3d import TDDFA3D, render
    tddfa = TDDFA3D()
    res = tddfa.fit(img, bbox)          # bbox 来自 SCRFD
    overlay = render.render_mesh_overlay(img, res.vertices, tddfa.tri)
"""
from . import io, render
from .tddfa import Face3DResult, TDDFA3D, roi_from_bbox, roi_from_landmarks

__all__ = ["TDDFA3D", "Face3DResult", "roi_from_bbox", "roi_from_landmarks", "render", "io"]
