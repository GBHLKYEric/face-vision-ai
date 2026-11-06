"""导出 3D 网格: .obj (带顶点颜色) 和 .ply, 可以用 MeshLab / Blender / Windows 3D 查看器打开。

两种都是文本格式, 用记事本打开就能看懂:
    OBJ:  v x y z r g b      一行一个顶点 (颜色 0~1, RGB)
          f i j k            一行一个三角形 (顶点下标从 1 开始!)
    PLY:  文件头声明顶点数/面数和每个属性的类型, 然后是顶点表和面表 (下标从 0 开始)

3D 软件习惯 y 轴朝上, 而图像 y 轴朝下, 所以导出时 y -> height - y。
y 翻转是一次"镜像", 会把三角形的绕序 (顺/逆时针) 也翻过来, 所以导出前要重新统一绕序:
约定从外面看是逆时针 (CCW), 这样法向朝外, 软件里光照才正确。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .render import orient_triangles


def _to_export_frame(vertices: np.ndarray, tri: np.ndarray, height: float | None):
    """(3,N) 图像坐标 -> (N,3) y 朝上的坐标, 并让三角形从相机一侧看是逆时针。"""
    v = vertices.T.astype(np.float64).copy()
    v[:, 1] = (height if height is not None else 0.0) - v[:, 1]
    return v, orient_triangles(v, tri)  # y 朝上的右手系里, 叉积 z>0 <=> 从 +z 看是逆时针


def _colors_rgb01(colors: np.ndarray | None, n: int) -> np.ndarray:
    if colors is None:
        return np.full((n, 3), 0.8)
    c = np.asarray(colors, np.float64)
    c = c / 255.0 if c.max() > 1.0 else c
    return np.clip(c[:, ::-1], 0, 1)  # BGR -> RGB


def save_obj(path, vertices: np.ndarray, tri: np.ndarray, colors: np.ndarray | None = None,
             height: float | None = None) -> Path:
    """vertices (3,N) 图像坐标; colors (N,3) BGR (0~255 或 0~1); height 为原图高度 (用于 y 翻转)。"""
    path = Path(path)
    v, f = _to_export_frame(vertices, tri, height)
    rgb = _colors_rgb01(colors, len(v))
    with open(path, "w", encoding="ascii") as fp:
        fp.write(f"# 3DDFA_V2 reconstruction: {len(v)} vertices, {len(f)} faces\n")
        np.savetxt(fp, np.hstack([v, rgb]), fmt="v %.3f %.3f %.3f %.4f %.4f %.4f")
        np.savetxt(fp, f + 1, fmt="f %d %d %d")
    return path


def save_ply(path, vertices: np.ndarray, tri: np.ndarray, colors: np.ndarray | None = None,
             height: float | None = None) -> Path:
    path = Path(path)
    v, f = _to_export_frame(vertices, tri, height)
    rgb = np.round(_colors_rgb01(colors, len(v)) * 255).astype(np.int64)
    header = ("ply\nformat ascii 1.0\n"
              f"element vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n")
    with open(path, "w", encoding="ascii") as fp:
        fp.write(header)
        np.savetxt(fp, np.hstack([v, rgb]), fmt="%.3f %.3f %.3f %d %d %d")
        np.savetxt(fp, np.hstack([np.full((len(f), 1), 3), f]), fmt="%d %d %d %d")
    return path


def load_obj(path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """读回 save_obj 写的文件: 返回 vertices (N,3), faces (T,3) (0 起始), colors (N,3) RGB 0~1 或 None。"""
    verts, faces = [], []
    with open(path, encoding="ascii") as fp:
        for line in fp:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:]])
            elif line.startswith("f "):
                faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    verts = np.array(verts, np.float64)
    colors = verts[:, 3:6] if verts.shape[1] >= 6 else None
    return verts[:, :3], np.array(faces, np.int64), colors
