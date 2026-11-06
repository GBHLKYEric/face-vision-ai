"""特效模块的快速测试: pytest tests/test_effects.py -q

只在模块级别跑一次 MediaPipe (LFW 照片), 其余都是纯 numpy/OpenCV 计算, 几秒内跑完。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from facevision.effects import beauty, makeup
from facevision.effects.avatar import AvatarRenderer, avatar_panel
from facevision.effects.filters import FaceSmoother, OneEuroFilter
from facevision.effects.stickers import STICKERS, StickerLibrary, alpha_composite_premul
from facevision.mesh import MESH, FaceMesh, pose_from_matrix
from facevision.paths import DATA

LFW_IMG = DATA / "lfw" / "Angelina_Jolie" / "Angelina_Jolie_0001.jpg"


@pytest.fixture(scope="module")
def img_face():
    if not LFW_IMG.exists():
        pytest.skip("缺少 LFW 测试图片")
    img = cv2.resize(cv2.imread(str(LFW_IMG)), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    with FaceMesh(mode="image") as mesh:
        faces = mesh.detect(img)
    assert len(faces) == 1
    return img, faces[0]


# ---------------- mesh ----------------
def test_mesh_output(img_face):
    img, f = img_face
    assert f.landmarks.shape == (478, 3) and f.landmarks.dtype == np.float32
    assert len(f.blendshapes) == 52 and all(0 <= v <= 1 for v in f.blendshapes.values())
    assert f.matrix.shape == (4, 4)
    # 图像左边的眼睛 x 坐标更小; 虹膜中心落在对应的眼睛轮廓里
    assert f.center("left_eye")[0] < f.center("right_eye")[0]
    for eye, iris in (("left_eye", "left_iris"), ("right_eye", "right_iris")):
        poly = f.pts(eye).astype(np.float32)
        assert cv2.pointPolygonTest(poly, tuple(map(float, f.xy[MESH[iris][0]])), False) >= 0


def test_pose_from_identity_matrix():
    assert np.allclose(pose_from_matrix(np.eye(4)), (0.0, 0.0, 0.0))


# ---------------- One Euro ----------------
def test_one_euro_converges_to_constant():
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    target = np.array([10.0, -3.0])
    y = None
    for i in range(300):
        y = f(target, i / 30)
    assert np.allclose(y, target, atol=1e-6)


def test_one_euro_reduces_jitter_and_tracks_steps():
    rng = np.random.default_rng(0)
    t = np.arange(300) / 30
    clean = np.where(t < 5, 0.0, 10.0)                     # 一个阶跃
    noisy = clean + rng.normal(0, 0.5, t.shape)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    out = np.array([f(np.array([x]), ti)[0] for x, ti in zip(noisy, t)])
    still = slice(30, 140)                                  # 静止段: 抖动明显变小
    assert out[still].std() < 0.5 * noisy[still].std()
    assert abs(out[-1] - 10.0) < 0.5                        # 阶跃之后能跟上


def test_face_smoother_keeps_count(img_face):
    _, f = img_face
    sm = FaceSmoother()
    for i in range(3):
        out = sm([f], i / 30)
    assert len(out) == 1 and out[0].landmarks.shape == (478, 3)
    R = out[0].matrix[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-4)     # 平滑后仍是旋转矩阵


# ---------------- 贴纸 ----------------
def test_alpha_composite_extremes():
    bg = np.full((4, 4, 3), 100, np.uint8)
    clear = np.zeros((4, 4, 4), np.float32)
    assert np.array_equal(alpha_composite_premul(bg, clear), bg)        # α=0: 背景不变
    solid = np.dstack([np.full((4, 4, 3), 200, np.float32), np.full((4, 4), 255, np.float32)])
    assert np.all(alpha_composite_premul(bg, solid) == 200)            # α=1: 全是贴纸


@pytest.mark.parametrize("mode", ["2d", "3d"])
def test_stickers_keep_shape_and_dtype(img_face, mode):
    img, f = img_face
    lib = StickerLibrary()
    for name in STICKERS:
        out = lib.apply(img.copy(), [f], name, mode=mode)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert not np.array_equal(out, img)                 # 确实贴上去了


def test_sticker_outside_image_is_noop(img_face):
    img, f = img_face
    small = img[:20, :20].copy()                           # 人脸完全不在这块小图里
    out = StickerLibrary().apply(small.copy(), [f], "sunglasses")
    assert np.array_equal(out, small)


# ---------------- 掩码 ----------------
def test_masks_in_unit_range(img_face):
    img, f = img_face
    skin = beauty.skin_mask(f, img.shape)
    assert skin.shape == img.shape[:2] and skin.dtype == np.float32
    assert skin.min() >= 0 and skin.max() <= 1 and skin.max() > 0.9
    nose = f.xy[MESH["nose_bridge"][4]].astype(int)        # 鼻梁上是皮肤
    eye = f.center("left_eye").astype(int)                 # 眼睛被挖掉
    mouth = f.center("lips_outer").astype(int)              # 嘴巴也被挖掉
    assert skin[nose[1], nose[0]] > 0.8 and skin[eye[1], eye[0]] < 0.05 and skin[mouth[1], mouth[0]] < 0.05
    lips, _ = makeup.lip_mask(f, img.shape)
    assert lips.min() >= 0 and lips.max() <= 1 and lips.max() > 0.9


# ---------------- 颜色特效 ----------------
@pytest.mark.parametrize("fn", [beauty.smooth_skin, beauty.whiten, makeup.lipstick, makeup.blush, makeup.eyeliner])
def test_color_effects_shape_and_zero_strength(img_face, fn):
    img, f = img_face
    out = fn(img.copy(), f, strength=0.8)
    assert out.shape == img.shape and out.dtype == np.uint8 and not np.array_equal(out, img)
    assert np.array_equal(fn(img.copy(), f, strength=0.0), img)


# ---------------- 局部变形 ----------------
def test_warps_identity_at_zero_strength(img_face):
    img, f = img_face
    assert np.array_equal(beauty.warp_face(img.copy(), f, slim=0.0, eyes=0.0), img)
    ys, xs = np.mgrid[0:50, 0:60].astype(np.float32)
    mx, my = xs.copy(), ys.copy()
    beauty.local_scale(mx, my, (30, 25), 20, a=0.0)
    beauty.local_translate(mx, my, (30, 25), (40, 25), 20, strength=0.0)
    assert np.allclose(mx, xs) and np.allclose(my, ys)


def test_warps_are_local_and_continuous(img_face):
    img, f = img_face
    out = beauty.warp_face(img.copy(), f, slim=1.0, eyes=1.0)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert np.array_equal(out[:40], img[:40])               # 远离人脸的地方不动
    # Gustafsson 局部缩放: 圆心不动, 圆周上恒等, 圆内向圆心取样 (放大)
    ys, xs = np.mgrid[0:101, 0:101].astype(np.float32)
    mx, my = xs.copy(), ys.copy()
    beauty.local_scale(mx, my, (50, 50), 30, a=0.5)
    assert abs(mx[50, 50] - 50) < 1e-4
    assert abs(mx[50, 80] - 80) < 1e-4 and abs(mx[50, 65] - 50) < 15


# ---------------- 头像 / 流水线 ----------------
def test_avatar_render_shapes(img_face):
    _, f = img_face
    r = AvatarRenderer(size=200)
    assert r.render({}, 0, 0, 0).shape == (200, 200, 3)
    assert avatar_panel(r, f, 240).shape == (240, 240, 3)
    assert avatar_panel(r, None, 240).shape == (240, 240, 3)


def test_pipeline_no_face_is_graceful():
    from facevision.effects import EffectPipeline
    pipe = EffectPipeline(["sunglasses", "smooth", "lipstick", "avatar"], mode="video")
    blank = np.full((240, 320, 3), 127, np.uint8)
    out = pipe.process(blank, 0)
    assert out.shape == (240, 320 + 240, 3)                 # avatar 面板拼在右边
    assert np.array_equal(out[:, :320], blank)
    pipe.close()
