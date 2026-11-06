"""3D 人脸重建模块的快速测试:  python -m pytest tests/test_face3d.py -q"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facevision.face3d import TDDFA3D, io, render  # noqa: E402
from facevision.face3d.tddfa import matrix2angle  # noqa: E402
from facevision.paths import DATA  # noqa: E402

IMG_PATH = DATA / "lfw" / "Colin_Powell" / "Colin_Powell_0001.jpg"
BOX = np.array([70.77, 63.67, 160.08, 185.44, 0.88], np.float32)  # SCRFD 在这张图上的检测框


@pytest.fixture(scope="module")
def tddfa() -> TDDFA3D:
    return TDDFA3D()


@pytest.fixture(scope="module")
def fitted(tddfa):
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        pytest.skip(f"缺少测试图片 {IMG_PATH}")
    return img, tddfa.fit(img, BOX)


def test_output_shapes(tddfa, fitted):
    img, r = fitted
    assert r.params.shape == (62,)
    assert r.vertices.shape == (3, 38365) and r.vertices.dtype == np.float32
    assert r.landmarks68.shape == (68, 3)
    assert tddfa.tri.shape == (76073, 3) and tddfa.tri.max() == r.vertices.shape[1] - 1
    # 关键点应落在检测框附近, 鼻尖 (30 号点) 在框内
    x1, y1, x2, y2 = BOX[:4]
    assert x1 < r.landmarks68[30, 0] < x2 and y1 < r.landmarks68[30, 1] < y2
    # 稀疏 68 点就是稠密网格中的 68 个顶点 (z 的零点不同, 只比 xy)。
    # 原版数据的小瑕疵: 第 44 号点的 z 下标取自相邻顶点 (12087 而不是 12383), 所以它差约 0.06 px
    idx = tddfa.keypoints[::3] // 3
    diff = np.abs(r.vertices[:2, idx] - r.landmarks68[:, :2].T)
    assert np.delete(diff, 44, axis=1).max() < 1e-3 and diff[:, 44].max() < 0.2


def test_rotation_matrix(fitted):
    R = fitted[1].R
    # 前两行是归一化过的; 第三行 = 叉积, 模长 = sin(前两行夹角), 网络输出只是"近似"正交
    np.testing.assert_allclose(np.linalg.norm(R[:2], axis=1), 1.0, atol=1e-5)
    assert abs(np.linalg.det(R) - 1) < 0.02
    assert np.abs(R @ R.T - np.eye(3)).max() < 0.05


def test_matrix2angle_roundtrip():
    yaw, pitch, roll = 0.4, -0.2, 0.1
    c, s = np.cos, np.sin
    Rx = np.array([[1, 0, 0], [0, c(pitch), -s(pitch)], [0, s(pitch), c(pitch)]])
    Ry = np.array([[c(-yaw), 0, s(-yaw)], [0, 1, 0], [-s(-yaw), 0, c(-yaw)]])
    Rz = np.array([[c(roll), -s(roll), 0], [s(roll), c(roll), 0], [0, 0, 1]])
    np.testing.assert_allclose(matrix2angle(Rz @ Ry @ Rx), (yaw, pitch, roll), atol=1e-9)


def test_mirror_flips_yaw_and_roll(tddfa, fitted):
    img, r = fitted
    w = img.shape[1]
    box = BOX.copy()
    box[[0, 2]] = w - 1 - BOX[[2, 0]]
    rm = tddfa.fit(cv2.flip(img, 1), box)
    assert np.sign(rm.pose_deg[0]) == -np.sign(r.pose_deg[0])
    assert abs(rm.pose_deg[0] + r.pose_deg[0]) < 5      # yaw 取反
    assert abs(rm.pose_deg[1] - r.pose_deg[1]) < 5      # pitch 不变


def test_renderers_return_correct_size(tddfa, fitted):
    img, r = fitted
    overlay = render.render_mesh_overlay(img, r.vertices, tddfa.tri)
    assert overlay.shape == img.shape and overlay.dtype == np.uint8
    assert np.abs(overlay.astype(int) - img).sum() > 0
    depth = render.render_depth(img.shape, r.vertices, tddfa.tri)
    assert depth.shape == img.shape and depth.max() > 200 and depth[0, 0].max() == 0
    colors = render.sample_colors(img, r.vertices)
    view = render.render_view(tddfa.frontal_vertices(r), tddfa.tri, colors, 30, 10, size=128)
    assert view.shape == (128, 128, 3)


def test_zbuffer_keeps_nearest_triangle():
    # 两个重叠三角形: 近的 (z=1) 红, 远的 (z=0) 蓝; 不管顺序如何, 中心像素都应是红色
    verts = np.array([[2, 2, 0], [18, 2, 0], [2, 18, 0], [2, 2, 1], [18, 2, 1], [2, 18, 1]], np.float32)
    colors = np.array([[255, 0, 0]] * 3 + [[0, 0, 255]] * 3, np.float32)
    for tri in (np.array([[0, 1, 2], [3, 4, 5]]), np.array([[3, 4, 5], [0, 1, 2]])):
        img, depth, mask = render.rasterize(verts, render.orient_triangles(verts, tri), colors, 20, 20)
        assert mask[6, 6] and abs(depth[6, 6] - 1) < 1e-5
        np.testing.assert_allclose(img[6, 6], (0, 0, 255), atol=1e-3)
        assert not mask[17, 17]


def test_zbuffer_matches_painter(tddfa, fitted):
    img, r = fitted
    v = r.vertices.T
    tri = render.orient_triangles(v, tddfa.tri)
    _, _, mask = render.rasterize(v, tri, np.ones((len(v), 1), np.float32), *img.shape[:2])
    canvas = np.zeros(img.shape[:2], np.uint8)
    render.rasterize_painter(v, tri, np.full((len(tri), 1), 255), canvas)
    painted = canvas > 0
    iou = (mask & painted).sum() / (mask | painted).sum()
    assert iou > 0.95


def test_obj_ply_roundtrip(tmp_path, tddfa, fitted):
    img, r = fitted
    colors = render.sample_colors(img, r.vertices)
    path = io.save_obj(tmp_path / "face.obj", r.vertices, tddfa.tri, colors, height=img.shape[0])
    lines = path.read_text().splitlines()
    assert sum(ln.startswith("v ") for ln in lines) == r.vertices.shape[1]
    assert sum(ln.startswith("f ") for ln in lines) == len(tddfa.tri)
    v, f, c = io.load_obj(path)
    np.testing.assert_allclose(v[:, 0], r.vertices[0], atol=1e-3)
    np.testing.assert_allclose(v[:, 1], img.shape[0] - r.vertices[1], atol=1e-3)   # y 轴翻转
    assert np.array_equal(np.sort(f, axis=1), np.sort(tddfa.tri, axis=1))           # 同一组三角形
    np.testing.assert_allclose(c, colors[:, ::-1] / 255, atol=1e-3)                 # BGR -> RGB
    ply = io.save_ply(tmp_path / "face.ply", r.vertices, tddfa.tri, colors, height=img.shape[0])
    text = ply.read_text().splitlines()
    assert f"element vertex {r.vertices.shape[1]}" in text
    assert len(text) == text.index("end_header") + 1 + r.vertices.shape[1] + len(tddfa.tri)
