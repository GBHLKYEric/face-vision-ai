"""纯 numpy + OpenCV 的三角网格渲染器 (不需要 OpenGL / GPU / Cython)。

渲染 = 把 3D 三角形"画"到 2D 图像上, 要解决两个问题:

1. 可见性: 多个三角形落在同一个像素上时, 只显示离相机最近的那个。
   - 画家算法: 按深度从远到近排序依次画, 近的自然盖住远的 (像油画家先画背景再画前景)。
     直观, 但三角形互相穿插时会出错, 而且只能整个三角形一个颜色。见 rasterize_painter。
   - Z-buffer (深度缓冲): 每个像素记住"目前最近的深度", 新来的点更近才覆盖。
     GPU 就是这么做的; 这里用 numpy 一次性处理所有 (三角形, 像素) 对, 见 rasterize。
   我们用正交投影 (和 3DDFA 的弱透视一致), 屏幕坐标 = 顶点的 (x, y), 深度 = z (越大越近)。

2. 明暗: 顶点法向 n 与光线方向 L 的夹角决定亮度。
   Lambert 漫反射 (余弦定律):  I = ka + kd·max(0, n·L)
   Blinn-Phong 高光:          + ks·max(0, n·H)^p,  H = normalize(L + V), V 是看向相机的方向
   每个顶点算出亮度后, 在三角形内部用重心坐标插值 (Gouraud 着色), 看起来是平滑的曲面。

坐标约定 (和 Face3DResult.vertices 一致): x 向右, y 向下 (像素), z 越大越靠近相机。
公开函数的 vertices 参数都是 (3, N), 内部转成 (N, 3) 方便按三角形下标取顶点。
"""
from __future__ import annotations

import cv2
import numpy as np

LIGHT_DIR = (-0.35, -0.45, 1.0)   # 光从左上方前面打过来 (y 向下, 所以"上"是 -y)
MESH_COLOR = (200, 208, 218)      # BGR, 略带暖色的"石膏像"
LMK68_PARTS = [(0, 17, False), (17, 22, False), (22, 27, False), (27, 31, False), (31, 36, False),
               (36, 42, True), (42, 48, True), (48, 60, True), (60, 68, True)]  # (起, 止, 是否闭合)


# ---------------------------------------------------------------------------
# 法向与光照
# ---------------------------------------------------------------------------
def orient_triangles(verts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """统一三角形的顶点顺序, 使叉积 (v1-v0)×(v2-v0) 指向相机 (+z), 即"正面朝外"。

    叉积方向取决于三个顶点的排列顺序 (顺时针/逆时针)。网格整体朝着相机,
    所以按面积加权的法向之和应该 z > 0; 如果不是, 就把所有三角形的顺序反过来。
    verts: (N, 3)。
    """
    v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    nz = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    return tri if nz.sum() >= 0 else tri[:, ::-1].copy()


def vertex_normals(verts: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """顶点法向 = 周围所有三角形法向的 (面积加权) 平均, 再归一化。verts (N,3) -> (N,3)。"""
    v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)  # 面法向, 长度 = 2 倍面积, 累加时大三角形权重自然更大
    idx = tri.ravel()
    vn = np.stack([np.bincount(idx, np.repeat(fn[:, c], 3), len(verts)) for c in range(3)], axis=1)
    return (vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def lambert_phong(normals: np.ndarray, light_dir=LIGHT_DIR, ambient: float = 0.35, diffuse: float = 0.6,
                  specular: float = 0.15, shininess: float = 20.0) -> np.ndarray:
    """每个顶点的亮度 I = ka + kd·max(0, n·L) + ks·max(0, n·H)^p。返回 (N,)。"""
    L = np.asarray(light_dir, np.float32)
    L = L / np.linalg.norm(L)
    V = np.array([0, 0, 1], np.float32)        # 正交投影: 视线方向处处相同
    H = (L + V) / np.linalg.norm(L + V)        # 半程向量
    ndl = np.clip(normals @ L, 0, None)
    spec = np.clip(normals @ H, 0, None) ** shininess * (ndl > 0)
    return (ambient + diffuse * ndl + specular * spec).astype(np.float32)


# ---------------------------------------------------------------------------
# 光栅化
# ---------------------------------------------------------------------------
def rasterize(verts: np.ndarray, tri: np.ndarray, attrs: np.ndarray, height: int, width: int,
              cull_back: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """向量化 Z-buffer 光栅化 + 重心坐标插值。

    verts: (N,3) 屏幕坐标 (像素中心在整数坐标上), attrs: (N,C) 顶点属性 (颜色 / 亮度 / 深度…)。
    返回 image (H,W,C) float32, depth (H,W) (背景 -inf), mask (H,W) bool。

    步骤: 对每个三角形枚举包围盒内的整数像素 -> 用重心坐标判断是否在三角形内
    -> 插值出深度 z -> 每个像素取 z 最大 (最近) 的那个 -> 用它的重心坐标插值属性。
    """
    attrs = attrs.reshape(len(verts), -1).astype(np.float32)
    v = verts[tri].astype(np.float64)                       # (T, 3 个顶点, xyz)
    x, y = v[..., 0], v[..., 1]
    # 2 倍有向面积; 三角形已 orient_triangles, 面积 > 0 表示正面朝向相机
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    x0 = np.maximum(np.ceil(x.min(1)), 0).astype(np.int64)
    x1 = np.minimum(np.floor(x.max(1)), width - 1).astype(np.int64)
    y0 = np.maximum(np.ceil(y.min(1)), 0).astype(np.int64)
    y1 = np.minimum(np.floor(y.max(1)), height - 1).astype(np.int64)
    keep = (x1 >= x0) & (y1 >= y0) & ((area > 1e-9) if cull_back else (np.abs(area) > 1e-9))
    ids = np.nonzero(keep)[0]
    # 按包围盒大小分桶 (1, 2, 4, 8, ... 像素), 每桶用 k×k 的像素网格一次算完
    span = np.maximum(x1 - x0, y1 - y0)[ids] + 1
    bucket = np.ceil(np.log2(span)).astype(np.int64)

    pix_all, tid_all, w_all = [], [], []
    for b in np.unique(bucket):
        k = 1 << int(b)
        oy, ox = np.divmod(np.arange(k * k), k)
        sel = ids[bucket == b]
        step = max(1, 2_000_000 // (k * k))                 # 分块, 控制内存
        for s in range(0, len(sel), step):
            t = sel[s:s + step]
            px = x0[t, None] + ox[None]                     # (Tb, k*k)
            py = y0[t, None] + oy[None]
            xa, xb, xc = x[t, 0:1], x[t, 1:2], x[t, 2:3]
            ya, yb, yc = y[t, 0:1], y[t, 1:2], y[t, 2:3]
            # 重心坐标 w_i = 对边与 p 组成的小三角形面积 / 大三角形面积
            w0 = ((xb - px) * (yc - py) - (xc - px) * (yb - py)) / area[t, None]
            w1 = ((xc - px) * (ya - py) - (xa - px) * (yc - py)) / area[t, None]
            w2 = 1.0 - w0 - w1
            eps = -1e-7
            inside = (px <= x1[t, None]) & (py <= y1[t, None]) & (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
            r, c = np.nonzero(inside)
            pix_all.append(py[r, c] * width + px[r, c])
            tid_all.append(t[r])
            w_all.append(np.stack([w0[r, c], w1[r, c], w2[r, c]], 1))

    image = np.zeros((height * width, attrs.shape[1]), np.float32)
    depth = np.full(height * width, -np.inf, np.float32)
    if not pix_all or sum(len(p) for p in pix_all) == 0:
        return image.reshape(height, width, -1), depth.reshape(height, width), np.zeros((height, width), bool)
    pix = np.concatenate(pix_all)
    tid = np.concatenate(tid_all)
    w = np.concatenate(w_all).astype(np.float32)
    z = np.einsum("ij,ij->i", w, v[tid, :, 2].astype(np.float32))

    # Z-buffer: 每个像素保留最大的 z
    np.maximum.at(depth, pix, z)
    win = z >= depth[pix]                                   # 共享边上可能并列, 任取其一即可
    pix, tid, w = pix[win], tid[win], w[win]
    vid = tri[tid]                                          # (M, 3) 三个顶点的下标
    image[pix] = np.einsum("ij,ijc->ic", w, attrs[vid])     # 属性 = Σ w_i · attr_i
    mask = np.isfinite(depth)
    return image.reshape(height, width, -1), depth.reshape(height, width), mask.reshape(height, width)


def rasterize_painter(verts: np.ndarray, tri: np.ndarray, tri_colors: np.ndarray, canvas: np.ndarray,
                      cull_back: bool = True) -> np.ndarray:
    """画家算法: 三角形按平均深度从远到近排序, 逐个用 cv2.fillConvexPoly 填充 (每个三角形一个颜色)。

    教学对照用: 和 rasterize 结果几乎一样 (每个三角形只有 1~2 个像素大时),
    但它不能插值颜色, 速度也受 Python 循环限制 (约几百毫秒)。
    """
    if cull_back:
        v0, v1, v2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
        area = (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
        tri, tri_colors = tri[area > 0], tri_colors[area > 0]
    order = np.argsort(verts[tri, 2].mean(1))              # z 小 = 远, 先画
    pts = np.round(verts[:, :2] * 16).astype(np.int32)[tri]  # shift=4: 坐标乘 16, 保留亚像素精度
    colors = tri_colors.tolist()
    for i in order:
        cv2.fillConvexPoly(canvas, pts[i], colors[i], cv2.LINE_8, 4)
    return canvas


# ---------------------------------------------------------------------------
# 高层渲染函数
# ---------------------------------------------------------------------------
def _as_list(vertices) -> list[np.ndarray]:
    return list(vertices) if isinstance(vertices, (list, tuple)) else [vertices]


def render_shaded(vertices: np.ndarray, tri: np.ndarray, height: int, width: int,
                  colors: np.ndarray | None = None, light_dir=LIGHT_DIR, **light_kw):
    """在 (height, width) 画布上渲染一个网格, 返回 (BGR float 图, mask)。

    colors=None: 灰白石膏色 × 光照; colors (N,3) BGR 0~255: 纹理色 × 光照。
    """
    v = vertices.T.astype(np.float32)
    tri = orient_triangles(v, tri)
    shade = lambert_phong(vertex_normals(v, tri), light_dir, **light_kw)
    base = np.broadcast_to(np.asarray(MESH_COLOR, np.float32), v.shape) if colors is None else colors
    img, _, mask = rasterize(v, tri, base * shade[:, None], height, width)
    return np.clip(img, 0, 255), mask


def render_mesh_overlay(img: np.ndarray, vertices, tri: np.ndarray, alpha: float = 0.65) -> np.ndarray:
    """把带光照的网格半透明地叠加在原图上 (检查重建是否对齐)。vertices: (3,N) 或其列表。"""
    out = img.astype(np.float32)
    for ver in _as_list(vertices):
        shaded, mask = render_shaded(ver, tri, img.shape[0], img.shape[1])
        out[mask] = (1 - alpha) * out[mask] + alpha * shaded[mask]
    return out.astype(np.uint8)


def render_depth(shape_hw, vertices, tri: np.ndarray, colormap: int | None = None) -> np.ndarray:
    """深度图: 每个像素的 z 线性映射到 20~255 (越亮越近), 背景黑色。
    shape_hw: 输出尺寸 (h, w[, c]); colormap: 如 cv2.COLORMAP_TURBO, 伪彩色更容易看出起伏。"""
    h, w = shape_hw[:2]
    out = np.zeros((h, w), np.float32)
    for ver in _as_list(vertices):
        v = ver.T.astype(np.float32)
        _, depth, mask = rasterize(v, orient_triangles(v, tri), v[:, 2:3], h, w)
        if not mask.any():
            continue
        z = depth[mask]
        out[mask] = (z - z.min()) / (np.ptp(z) + 1e-6) * 235 + 20  # 留一点亮度和背景区分
    out = out.astype(np.uint8)
    if colormap is not None:
        color = cv2.applyColorMap(out, colormap)
        color[out == 0] = 0
        return color
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def sample_colors(img: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """纹理采样: 每个顶点投影到原图上的 (x, y) 处取颜色。返回 (N,3) BGR float32。

    (x, y) 一般不是整数, 用双线性插值: 周围 4 个像素按距离加权平均。
    """
    h, w = img.shape[:2]
    x = np.clip(vertices[0], 0, w - 1)
    y = np.clip(vertices[1], 0, h - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    im = img.astype(np.float32)
    return (im[y0, x0] * (1 - fx) * (1 - fy) + im[y0, x1] * fx * (1 - fy)
            + im[y1, x0] * (1 - fx) * fy + im[y1, x1] * fx * fy).astype(np.float32)


def vertex_visibility(vertices: np.ndarray, tri: np.ndarray, height: int, width: int,
                      tol: float = 0.02) -> np.ndarray:
    """哪些顶点在原图里是可见的 (没被鼻子等挡住)? 和 z-buffer 比较深度即可。

    被挡住的顶点采样到的是挡住它的那块皮肤的颜色, 从侧面看会出现"重影", 需要特殊处理。
    """
    v = vertices.T.astype(np.float32)
    _, depth, mask = rasterize(v, orient_triangles(v, tri), v[:, 2:3], height, width)
    xi = np.clip(np.round(v[:, 0]).astype(int), 0, width - 1)
    yi = np.clip(np.round(v[:, 1]).astype(int), 0, height - 1)
    zbuf = depth[yi, xi]
    return mask[yi, xi] & (v[:, 2] >= zbuf - tol * np.ptp(v[:, 2]))


def texture_from_image(img: np.ndarray, vertices: np.ndarray, tri: np.ndarray,
                       mirror_index: np.ndarray | None = None, min_conf: float = 0.35) -> np.ndarray:
    """给每个顶点取颜色, 并用左右对称补全照片里看不清的部分。返回 (N,3) BGR float32。

    可信度 conf = 顶点法向朝向相机的程度 n_z (正对相机 = 1, 侧面 = 0), 被遮挡的顶点 conf = 0。
    侧面的皮肤在照片里被严重压缩, 采到的颜色会被拉伸成条纹, 甚至采到背景;
    若对称点更可信, 就按比例借用对称点的颜色 (人脸大致左右对称)。
    """
    colors = sample_colors(img, vertices)
    if mirror_index is None:
        return colors
    v = vertices.T.astype(np.float32)
    n = vertex_normals(v, orient_triangles(v, tri))
    vis = vertex_visibility(vertices, tri, img.shape[0], img.shape[1])
    w = np.clip(n[:, 2] * vis / min_conf, 0, 1)          # 自己的可信度 (0~1)
    borrow = ((1 - w) * w[mirror_index])[:, None]         # 自己不可信、对称点可信 -> 借用
    return (1 - borrow) * colors + borrow * colors[mirror_index]


def rotation_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """图像坐标系 (x 右, y 下, z 朝相机) 下的旋转: yaw > 0 鼻子转向画面左侧, pitch > 0 低头。
    (和 Face3DResult.pose_deg 的符号约定一致)"""
    a, b = np.radians(yaw_deg), np.radians(pitch_deg)
    Ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]], np.float32)
    Rx = np.array([[1, 0, 0], [0, np.cos(b), np.sin(b)], [0, -np.sin(b), np.cos(b)]], np.float32)
    return Rx @ Ry


def render_view(vertices: np.ndarray, tri: np.ndarray, colors: np.ndarray | None = None,
                yaw_deg: float = 0.0, pitch_deg: float = 0.0, size: int = 512, bg=(250, 250, 250),
                fill: float = 0.92, shade_strength: float = 0.5) -> np.ndarray:
    """从任意角度看网格: 绕网格中心旋转 -> 正交投影到 size×size 的干净背景上。

    colors: (N,3) BGR 0~255 的顶点颜色 (纹理), None 则渲染灰白石膏像。
    shade_strength: 纹理模式下光照的强度 (照片本身已经带光照, 不宜再加太强)。
    缩放只由网格本身决定 (不随角度变化), 所以连续角度渲染出来的视频不会抖动。
    """
    v = vertices.T.astype(np.float32)
    tri = orient_triangles(v, tri)                 # 先在原始 (朝向相机) 姿态下统一朝向
    v = v - (v.min(0) + v.max(0)) / 2              # 以包围盒中心为旋转中心
    radius = np.linalg.norm(v, axis=1).max()
    v = v @ rotation_yaw_pitch(yaw_deg, pitch_deg).T
    normals = vertex_normals(v, tri)
    screen = v * (fill * size / 2 / radius)
    screen[:, :2] += size / 2

    if colors is None:
        attrs = np.asarray(MESH_COLOR, np.float32) * lambert_phong(normals)[:, None]
    else:
        light = lambert_phong(normals, ambient=0.3, diffuse=0.75, specular=0.08)
        light = (1 - shade_strength) + shade_strength * light / 0.9   # 正对光源时亮度约为 1
        attrs = colors.astype(np.float32) * light[:, None]
    img, _, mask = rasterize(screen, tri, attrs, size, size)
    out = np.empty((size, size, 3), np.uint8)
    out[:] = bg
    out[mask] = np.clip(img[mask], 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# 2D 标注
# ---------------------------------------------------------------------------
def draw_landmarks68(img: np.ndarray, lmk: np.ndarray, color=(80, 255, 80), radius: int | None = None) -> np.ndarray:
    """画 68 点: 轮廓 / 眉毛 / 鼻子 / 眼睛 / 嘴巴 分组连线。lmk: (68, 2 或 3)。"""
    out = img.copy()
    scale = max(1.0, (np.ptp(lmk[:, 0]) + np.ptp(lmk[:, 1])) / 300)
    r = radius or max(1, int(round(1.5 * scale)))
    th = max(1, int(round(scale)))
    for a, b, closed in LMK68_PARTS:
        pts = np.round(lmk[a:b, :2] * 16).astype(np.int32)
        cv2.polylines(out, [pts], closed, (255, 255, 255), th, cv2.LINE_AA, 4)
    for x, y in lmk[:, :2]:
        cv2.circle(out, (int(round(x * 16)), int(round(y * 16))), r * 16, color, -1, cv2.LINE_AA, 4)
    return out


def draw_pose_axes(img: np.ndarray, R: np.ndarray, origin, length: float, pose_deg=None) -> np.ndarray:
    """在 origin (通常是鼻尖) 画出头部坐标系的三根轴: x 红 (脸的左侧), y 绿 (头顶), z 蓝 (鼻子朝向)。

    R 是 y 轴朝上的模型坐标系 -> 相机坐标系的旋转; 图像 y 轴朝下, 所以投影时 y 分量取负。
    正脸时 z 轴正对相机, 投影成一个点; 转头越厉害, 蓝线越长。
    """
    out = img.copy()
    o = np.asarray(origin, np.float64)[:2]
    th = max(2, int(round(length / 40)))
    for k, color in [(2, (255, 120, 0)), (0, (0, 0, 255)), (1, (0, 200, 0))]:
        d = np.array([R[0, k], -R[1, k]]) * length
        cv2.arrowedLine(out, tuple(np.round(o).astype(int)), tuple(np.round(o + d).astype(int)),
                        color, th, cv2.LINE_AA, tipLength=0.15)
    if pose_deg is not None:
        put_label(out, "yaw %+.1f  pitch %+.1f  roll %+.1f" % tuple(pose_deg), bottom=True)
    return out


def put_label(img: np.ndarray, text: str, org=(6, 6), scale: float | None = None,
              bottom: bool = False) -> np.ndarray:
    """带深色底的文字标签 (直接画在 img 上); bottom=True 时放在左下角。"""
    scale = scale or max(0.4, img.shape[1] / 900)
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, th)
    x, y = org
    if bottom:
        y = img.shape[0] - (tht + base + 8) - org[1]
    cv2.rectangle(img, (x, y), (x + tw + 8, y + tht + base + 8), (30, 30, 30), -1)
    cv2.putText(img, text, (x + 4, y + tht + 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), th, cv2.LINE_AA)
    return img
