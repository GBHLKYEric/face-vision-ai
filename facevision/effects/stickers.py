"""动态贴纸: 墨镜 / 礼帽 / 猫耳 / 胡子, 跟随头部的平移、缩放、歪头 (roll) 和转头 (yaw)。

整体流程 (每张脸每帧):
  1. 在 "脸坐标" 里规定贴纸放在哪 (common.face_frame: 原点 = 两眼中点, 单位 = 两眼距离 IOD)
     例如墨镜: 中心对准两眼中点, 宽 2.3 个 IOD。
  2. 把贴纸图片的 4 个角映射到脸坐标, 再变换到图像上, 得到 4 个目标点。
  3. 由 "贴纸 4 角 -> 图像 4 点" 求变换矩阵, 把贴纸 warp 到画面上:
       - mode="2d": 只用 2D 关键点, Y 轴取 X 轴旋转 90°, 4 个点之间只差 旋转+等比缩放+平移,
         即 **相似变换** (4 个自由度, 2 对点就能唯一确定):
             [x']   [s·cosθ  -s·sinθ  tx] [x]
             [y'] = [s·sinθ   s·cosθ  ty] [y]
                                          [1]
         s = 图像上两眼距离 / 贴纸上两镜片距离, θ = 两眼连线的倾角 (就是 roll)。
         能跟随平移、缩放、歪头, 但转头时贴纸不会 "变窄"。
       - mode="3d" (默认): 用带深度 z 的 3D 关键点建立坐标系, 投影到图像后, 转头时 X 轴变短
         (≈ cos(yaw)), 贴纸自然被压扁; 再加一点近大远小 (透视) 修正,
         4 对点不再满足相似/仿射关系, 所以用 **透视变换 (单应矩阵 H, 8 个自由度)**:
             [x', y', 1]ᵀ ∝ H · [x, y, 1]ᵀ     (cv2.getPerspectiveTransform 由 4 对点解出 H)
  4. Alpha 合成 ("over" 运算):  out = α·贴纸 + (1-α)·背景。
     这里用 "预乘 alpha" (先把 RGB 乘上 α 再做插值/缩放), 公式变成 out = 贴纸预乘 + (1-α)·背景,
     这样 warp 时透明区域的黑色不会 "渗" 进贴纸边缘, 不会出现黑边。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..mesh import KEY, MeshFace
from ..paths import ASSETS
from .common import face_frame, face_to_image

STICKER_DIR = ASSETS / "stickers"


@dataclass
class StickerSpec:
    file: str
    anchor_px: tuple[float, float]       # 贴纸图片上的锚点 (像素)
    base: str | int                      # 锚点对准的人脸位置: "eyes" = 两眼中点, 或关键点下标
    offset: tuple[float, float, float]   # 在 base 基础上再偏移 (u, v, w), 单位 IOD; v 向下为正, w 向脑后为正
    width: float                         # 贴纸在脸上的宽度 (IOD)


STICKERS: dict[str, StickerSpec] = {
    "sunglasses": StickerSpec("sunglasses.png", (340, 112), "eyes", (0.0, 0.05, -0.25), 2.6),
    "hat": StickerSpec("top_hat.png", (260, 470), KEY["forehead_top"], (0.0, -0.05, 0.5), 3.3),
    "cat_ears": StickerSpec("cat_ears.png", (400, 330), KEY["forehead_top"], (0.0, -0.2, 0.45), 3.3),
    "mustache": StickerSpec("mustache.png", (200, 38), KEY["philtrum"], (0.0, 0.0, -0.05), 1.6),
}


# ---------------------------------------------------------------------------
# 1) 用 PIL 程序化地 "画" 出贴纸素材 (RGBA PNG)
#    先在 4 倍大的画布上画, 再缩小 -> 免费得到抗锯齿的平滑边缘 (超采样)。
# ---------------------------------------------------------------------------
SS = 4


def _canvas(w: int, h: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img)


def _finish(img: Image.Image, w: int, h: int, path: Path):
    img.resize((w, h), Image.LANCZOS).save(path)


def _s(*v):  # 坐标乘超采样倍数
    return [x * SS for x in v]


def draw_sunglasses(path: Path):
    w, h = 680, 230
    img, d = _canvas(w, h)
    lens_w, lens_h = 250, 160
    for cx in (190, 490):
        box = _s(cx - lens_w / 2, 112 - lens_h / 2, cx + lens_w / 2, 112 + lens_h / 2)
        # 镜片: 上深下浅的渐变, 半透明
        grad = Image.new("RGBA", img.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(grad)
        y0, y1 = box[1], box[3]
        for y in range(int(y0), int(y1)):
            t = (y - y0) / (y1 - y0)
            gd.line([(box[0], y), (box[2], y)], fill=(int(15 + 45 * t), int(18 + 40 * t), int(30 + 60 * t), 235))
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle(box, radius=70 * SS, fill=255)
        img.paste(grad, (0, 0), mask)
        # 高光: 一道斜的白色半透明条纹
        glare = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(glare).polygon(_s(cx - 80, 75, cx - 45, 75, cx - 95, 165, cx - 130, 165), fill=(255, 255, 255, 70))
        img.alpha_composite(Image.composite(glare, Image.new("RGBA", img.size, (0, 0, 0, 0)), mask))
        # 镜框
        d.rounded_rectangle(box, radius=70 * SS, outline=(10, 10, 12, 255), width=14 * SS)
    # 鼻梁 + 上横梁 + 两侧镜腿
    d.line(_s(305, 85, 340, 72, 375, 85), fill=(10, 10, 12, 255), width=14 * SS, joint="curve")
    d.rectangle(_s(0, 52, 70, 72), fill=(10, 10, 12, 255))
    d.rectangle(_s(610, 52, 680, 72), fill=(10, 10, 12, 255))
    _finish(img, w, h, path)


def draw_top_hat(path: Path):
    w, h = 520, 520
    img, d = _canvas(w, h)
    black, dark = (22, 22, 26, 255), (38, 38, 46, 255)
    d.ellipse(_s(10, 430, 510, 510), fill=black)                                # 帽檐 (后半)
    d.polygon(_s(105, 70, 415, 70, 400, 470, 120, 470), fill=dark)              # 帽筒, 上宽下窄
    d.polygon(_s(150, 75, 185, 75, 180, 465, 150, 465), fill=(70, 70, 82, 255))  # 竖条高光
    d.polygon(_s(116, 380, 404, 380, 401, 440, 119, 440), fill=(175, 25, 40, 255))  # 红色缎带
    d.ellipse(_s(105, 45, 415, 95), fill=(55, 55, 66, 255))                     # 帽顶
    d.chord(_s(10, 430, 510, 510), 0, 180, fill=black)                          # 帽檐前半, 挡住帽筒底部
    d.arc(_s(10, 430, 510, 510), 10, 170, fill=(80, 80, 90, 255), width=3 * SS)
    _finish(img, w, h, path)


def draw_cat_ears(path: Path):
    w, h = 800, 380
    img, d = _canvas(w, h)
    fur, inner, band = (60, 50, 50, 255), (250, 160, 185, 255), (235, 90, 140, 255)
    # 发箍: 一段椭圆弧
    d.arc(_s(40, 250, 760, 610), 200, 340, fill=band, width=24 * SS)
    for sx in (-1, 1):
        def X(x):  # 左右镜像
            return 400 + sx * (x - 400)
        outer = [(X(165), 330), (X(205), 20), (X(385), 262)]
        inner_tri = [(X(205), 295), (X(225), 95), (X(335), 262)]
        d.polygon([v * SS for p in outer for v in p], fill=fur)
        d.polygon([v * SS for p in inner_tri for v in p], fill=inner)
    img = img.filter(ImageFilter.GaussianBlur(1.5 * SS))  # 让三角形尖角稍微圆润
    _finish(img, w, h, path)


def draw_mustache(path: Path):
    w, h = 400, 150
    img, d = _canvas(w, h)
    color = (55, 35, 25, 255)
    # 沿一条参数曲线画很多半径递减的圆, 叠起来就是一笔 "由粗到细" 的笔触 (八字胡)
    for sx in (-1, 1):
        for t in np.linspace(0, 1, 160):
            x = 200 + sx * 175 * t
            y = 45 + 40 * np.sin(np.pi * t * 0.8) - 45 * t ** 4
            r = 26 * (1 - t) ** 0.9 + 4
            d.ellipse(_s(x - r, y - r, x + r, y + r), fill=color)
    _finish(img, w, h, path)


ASSET_DRAWERS = {
    "sunglasses.png": draw_sunglasses,
    "top_hat.png": draw_top_hat,
    "cat_ears.png": draw_cat_ears,
    "mustache.png": draw_mustache,
}


def make_sticker_assets(out_dir: Path = STICKER_DIR, force: bool = False) -> list[Path]:
    """生成所有贴纸 PNG (已存在则跳过, force=True 重画)。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in ASSET_DRAWERS.items():
        p = out_dir / name
        if force or not p.exists():
            fn(p)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# 2) 锚定 + warp + alpha 合成
# ---------------------------------------------------------------------------
class Sticker:
    """一张贴纸: 预乘 alpha 的 float32 图像金字塔 + 放置规则。"""

    def __init__(self, rgba_bgra: np.ndarray, spec: StickerSpec):
        self.spec = spec
        a = rgba_bgra[..., 3:4].astype(np.float32) / 255.0
        premul = np.dstack([rgba_bgra[..., :3].astype(np.float32) * a, a * 255.0])  # (B·α, G·α, R·α, 255α)
        # 图像金字塔 (mipmap): 贴纸要缩得很小时, 先挑一层接近目标大小的再 warp, 避免锯齿和闪烁
        self.levels = [premul]
        while self.levels[-1].shape[1] > 64:
            p = self.levels[-1]
            self.levels.append(cv2.resize(p, (p.shape[1] // 2, p.shape[0] // 2), interpolation=cv2.INTER_AREA))
        self.h, self.w = premul.shape[:2]

    def corners_uvw(self) -> np.ndarray:
        """贴纸 4 个角在 "相对 base 的脸坐标" 里的位置 (u, v, w)。"""
        s = self.spec.width / self.w                    # 每个贴纸像素 = 多少 IOD
        ax, ay = self.spec.anchor_px
        du, dv, dw = self.spec.offset
        corners = np.array([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], np.float64)
        uv = (corners - [ax, ay]) * s + [du, dv]
        return np.column_stack([uv, np.full(4, dw)])


def _base_point(face: MeshFace, base: str | int, O: np.ndarray) -> np.ndarray:
    if base == "eyes":
        return O
    return face.landmarks[int(base)].astype(np.float64)


def sticker_quad(face: MeshFace, sticker: Sticker, img_shape, mode: str = "3d") -> np.ndarray:
    """计算贴纸 4 个角在图像上的位置 (4, 2)。"""
    O, X, Y, Z = face_frame(face)
    if mode == "2d":
        # 2D 相似变换: 只用 X 的图像投影, Y 取 X 顺时针转 90°(图像 y 向下), 长度相同
        X2 = X[:2]
        Y2 = np.array([-X2[1], X2[0]])
        X, Y, Z = np.r_[X2, 0], np.r_[Y2, 0], np.zeros(3)
    B = _base_point(face, sticker.spec.base, O)
    P = face_to_image(sticker.corners_uvw(), B, X, Y, Z)    # (4, 3)
    xy = P[:, :2]
    if mode == "3d":
        # 近大远小 (针孔相机): x' = c + (x - c)·f/(f + Δz)
        #   c = 画面中心 (主点), f = 焦距 (像素, 这里近似取图像宽, 约 53° 视场角),
        #   Δz = 相对两眼中点的深度差 (Δz=0 的点不动)。
        # 转头时贴纸平面是斜的, 两侧 Δz 一正一负, 一边放大一边缩小 -> 梯形。
        h, w = img_shape[:2]
        f = float(w)
        c = np.array([w / 2, h / 2])
        dz = P[:, 2] - O[2]
        xy = c + (xy - c) * (f / (f + dz))[:, None]
    return xy.astype(np.float32)


def alpha_composite_premul(bg: np.ndarray, fg_premul: np.ndarray) -> np.ndarray:
    """预乘 alpha 的 "over" 合成: out = fg + (1-α)·bg。bg uint8 (h,w,3), fg float32 (h,w,4)。"""
    a = fg_premul[..., 3:4] / 255.0
    out = fg_premul[..., :3] + (1.0 - a) * bg.astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def apply_sticker(img: np.ndarray, face: MeshFace, sticker: Sticker, mode: str = "3d") -> np.ndarray:
    """把贴纸贴到一张脸上 (原地修改 img 并返回)。"""
    dst = sticker_quad(face, sticker, img.shape, mode)
    h, w = img.shape[:2]
    x0, y0 = np.floor(dst.min(0)).astype(int)
    x1, y1 = np.ceil(dst.max(0)).astype(int) + 1
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img                                       # 贴纸完全在画面外
    # 选金字塔层: 目标宽度 (上边长度) 对应的最小但不小于它的那层
    target_w = float(np.linalg.norm(dst[1] - dst[0]))
    lvl = 0
    while lvl + 1 < len(sticker.levels) and sticker.levels[lvl + 1].shape[1] >= target_w:
        lvl += 1
    src_img = sticker.levels[lvl]
    sh, sw = src_img.shape[:2]
    src = np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32)
    dst_roi = dst - np.array([x0, y0], np.float32)      # 只 warp 到 ROI 里, 省时间
    if mode == "2d":
        M = cv2.getAffineTransform(src[:3], dst_roi[:3])   # 相似变换 (仿射的特例), 3 对点即可
        warped = cv2.warpAffine(src_img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    else:
        H = cv2.getPerspectiveTransform(src, dst_roi)      # 4 对点 -> 3x3 单应矩阵
        warped = cv2.warpPerspective(src_img, H, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    img[y0:y1, x0:x1] = alpha_composite_premul(img[y0:y1, x0:x1], warped)
    return img


class StickerLibrary:
    """加载 (必要时先生成) assets/stickers/ 下的所有贴纸。"""

    def __init__(self, sticker_dir: Path = STICKER_DIR):
        make_sticker_assets(sticker_dir)
        self.stickers: dict[str, Sticker] = {}
        for name, spec in STICKERS.items():
            rgba = cv2.imread(str(sticker_dir / spec.file), cv2.IMREAD_UNCHANGED)
            self.stickers[name] = Sticker(rgba, spec)

    def apply(self, img: np.ndarray, faces: list[MeshFace], name: str, mode: str = "3d") -> np.ndarray:
        for face in faces:   # 多张脸: 每张脸各算各的变换
            apply_sticker(img, face, self.stickers[name], mode)
        return img
