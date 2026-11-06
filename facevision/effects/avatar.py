"""表情驱动的卡通头像 (Animoji 的极简版)。

MediaPipe 输出 52 个 blendshape 系数 (ARKit 标准, 每个 0~1), 每个系数描述一个 "局部动作",
例如 eyeBlinkLeft (左眼闭合)、jawOpen (张嘴)、mouthSmileLeft (左嘴角上扬)、browInnerUp (眉心上抬)。
3D 动画里, 角色脸 = 中性脸 + Σ 系数_i × 动作形变_i —— 只要 "系数" 对齐, 就能把真人表情迁移到任何角色上。

这里不用 3D 模型, 直接用 OpenCV 画一个矢量卡通脸, 把系数映射到绘制参数:
    眼睛张开度  = 1 - eyeBlink + 0.3·eyeWide
    瞳孔位置    = eyeLookIn/Out/Up/Down
    眉毛高度    = browInnerUp / browOuterUp - browDown
    嘴巴张开    = jawOpen + mouthUpperUp + mouthLowerDown;  嘴宽 = 1 + smile + stretch - pucker;
    嘴角上扬    = smile - frown
    头部姿态    = yaw/pitch 让五官整体平移 (假 3D), roll 让整张图旋转
注意 "Left/Right" 是被拍者自己的左右; 卡通脸画在图像坐标里, 被拍者的左眼在画面右侧。
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ..mesh import MeshFace

BG = (236, 226, 212)          # BGR
SKIN = (170, 205, 245)
SKIN_DARK = (120, 160, 215)
HAIR = (50, 60, 95)
SHIRT = (160, 110, 60)
INK = (40, 35, 40)
PINK = (150, 120, 245)


def _bez(p0, p1, p2, n: int = 24) -> np.ndarray:
    """二次贝塞尔曲线上的 n 个点 (画嘴巴弧线用)。"""
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * np.asarray(p0) + 2 * (1 - t) * t * np.asarray(p1) + t ** 2 * np.asarray(p2)


class AvatarRenderer:
    """把 (blendshapes, yaw, pitch, roll) 画成卡通头像。先画 2 倍大再缩小 = 抗锯齿。"""

    def __init__(self, size: int = 480, supersample: int = 2):
        self.size = size
        self.ss = supersample

    def render(self, bs: dict[str, float] | None, yaw: float = 0.0, pitch: float = 0.0,
               roll: float = 0.0) -> np.ndarray:
        bs = bs or {}
        g = lambda k: float(bs.get(k, 0.0))  # noqa: E731
        N = self.size * self.ss
        S = self.ss
        img = np.full((N, N, 3), BG, np.uint8)
        aa = cv2.LINE_AA
        P = lambda x, y: (int(round(x)), int(round(y)))  # noqa: E731

        ys = math.sin(math.radians(np.clip(yaw, -50, 50)))      # 转头 -> 五官左右平移
        ps = math.sin(math.radians(np.clip(pitch, -40, 40)))    # 抬头 -> 五官上移
        hx, hy = N * 0.5 + ys * 0.03 * N, N * 0.52              # 头的中心
        head_ax = (int(0.30 * N * (1 - 0.10 * abs(ys))), int(0.35 * N))
        fx, fy = hx + ys * 0.12 * N, hy - ps * 0.10 * N + 0.03 * N  # 五官的中心

        # ---- 身体、耳朵、头发、脸 ----
        cv2.ellipse(img, P(hx - ys * 0.02 * N, N * 1.05), (int(0.36 * N), int(0.20 * N)), 0, 0, 360, SHIRT, -1, aa)
        cv2.rectangle(img, P(hx - 0.07 * N, hy + 0.25 * N), P(hx + 0.07 * N, N * 0.9), SKIN_DARK, -1)
        for sx in (-1, 1):
            ex = hx + sx * head_ax[0] * 0.98 - ys * 0.05 * N
            cv2.ellipse(img, P(ex, hy + 0.03 * N), (int(0.045 * N), int(0.075 * N)), 0, 0, 360, SKIN_DARK, -1, aa)
        cv2.ellipse(img, P(hx - ys * 0.02 * N, hy - 0.05 * N), (head_ax[0] + int(0.03 * N), head_ax[1] + int(0.02 * N)),
                    0, 0, 360, HAIR, -1, aa)
        cv2.ellipse(img, P(hx, hy), head_ax, 0, 0, 360, SKIN, -1, aa)
        cv2.ellipse(img, P(hx - ys * 0.04 * N, hy - 0.17 * N), (head_ax[0] + int(0.01 * N), int(0.20 * N)),
                    0, 180, 360, HAIR, -1, aa)  # 刘海

        # ---- 腮红: 笑得越开心越红 ----
        smile = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2
        blush_col = tuple(int(s + (p - s) * (0.25 + 0.6 * smile)) for s, p in zip(SKIN, PINK))
        for sx in (-1, 1):
            cv2.ellipse(img, P(fx + sx * 0.15 * N, fy + 0.09 * N), (int(0.05 * N), int(0.03 * N)), 0, 0, 360,
                        blush_col, -1, aa)

        # ---- 眼睛 + 眉毛 ----
        # side=-1: 画面左边 = 被拍者的右眼 (...Right 系数); side=+1: 画面右边 = 被拍者左眼 (...Left)
        for side, sfx in ((-1, "Right"), (1, "Left")):
            near = 1 + 0.15 * ys * (-side)                     # 转头时靠近镜头那只眼睛稍大
            ex, ey = fx + side * 0.115 * N * (1 - 0.25 * abs(ys)), fy - 0.04 * N
            rx = 0.055 * N * near
            openness = np.clip(1.0 - 1.15 * g("eyeBlink" + sfx) + 0.4 * g("eyeWide" + sfx), 0.0, 1.3)
            ry = 0.062 * N * near * openness
            if openness < 0.18:                               # 闭眼: 画一道向下弯的弧线
                cv2.ellipse(img, P(ex, ey), (int(rx), int(0.025 * N)), 0, 10, 170, INK, int(0.012 * N), aa)
            else:
                cv2.ellipse(img, P(ex, ey), (int(rx), int(ry)), 0, 0, 360, (255, 255, 255), -1, aa)
                # 瞳孔: 看向哪里 (In = 朝鼻子, Out = 朝外)
                look_x = (g("eyeLookIn" + sfx) - g("eyeLookOut" + sfx)) * (-side)
                look_y = g("eyeLookDown" + sfx) - g("eyeLookUp" + sfx)
                px = ex + (look_x * 0.5 + ys * 0.4) * rx
                py = ey + (look_y * 0.5 - ps * 0.4) * ry
                pr = int(0.55 * rx)
                x0, y0 = int(ex - rx - 2), int(ey - ry - 2)
                x1, y1 = int(ex + rx + 3), int(ey + ry + 3)
                eye_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
                cv2.ellipse(eye_mask, P(ex - x0, ey - y0), (int(rx), int(ry)), 0, 0, 360, 255, -1, aa)
                pupil = img[y0:y1, x0:x1].copy()
                cv2.circle(pupil, P(px - x0, py - y0), pr, (60, 50, 40), -1, aa)
                cv2.circle(pupil, P(px - x0, py - y0), int(pr * 0.55), (20, 15, 15), -1, aa)
                cv2.circle(pupil, P(px - x0 - pr * 0.35, py - y0 - pr * 0.35), max(2, int(pr * 0.22)), (255, 255, 255), -1, aa)
                m = eye_mask[..., None].astype(np.float32) / 255  # 瞳孔不能画出眼眶
                img[y0:y1, x0:x1] = (pupil * m + img[y0:y1, x0:x1] * (1 - m)).astype(np.uint8)
                cv2.ellipse(img, P(ex, ey), (int(rx), int(ry)), 0, 180, 360, INK, int(0.010 * N), aa)  # 上眼睑
            # 眉毛: 整体高度 + 眉心单独上抬/下压
            lift = (0.6 * g("browOuterUp" + sfx) + 0.5 * g("browInnerUp") - 0.8 * g("browDown" + sfx)) * 0.05 * N
            inner_lift = (g("browInnerUp") - g("browDown" + sfx)) * 0.03 * N
            by = ey - 0.11 * N - lift
            b_out = (ex + side * 0.07 * N, by + 0.012 * N)
            b_mid = (ex + side * 0.005 * N, by - 0.012 * N)
            b_in = (ex - side * 0.065 * N, by + 0.008 * N - inner_lift)
            pts = _bez(b_out, b_mid, b_in, 16).astype(np.int32)
            cv2.polylines(img, [pts], False, HAIR, int(0.022 * N), aa)

        # ---- 鼻子 ----
        nx, ny = fx + ys * 0.03 * N, fy + 0.07 * N
        cv2.ellipse(img, P(nx, ny), (int(0.022 * N), int(0.014 * N)), 0, 20, 160, SKIN_DARK, int(0.008 * N), aa)

        # ---- 嘴巴 ----
        mx, my = fx + ys * 0.02 * N, fy + 0.18 * N
        stretch = (g("mouthStretchLeft") + g("mouthStretchRight")) / 2
        half_w = 0.08 * N * (1 + 0.45 * smile + 0.3 * stretch - 0.5 * g("mouthPucker"))
        # 嘴唇张开 = 下巴张开 (jawOpen) + 上唇上提 (mouthUpperUp) + 下唇下拉 (mouthLowerDown, 露牙时)
        jaw = np.clip(1.2 * g("jawOpen") + 0.4 * (g("mouthUpperUpLeft") + g("mouthUpperUpRight")) / 2
                      + 0.3 * (g("mouthLowerDownLeft") + g("mouthLowerDownRight")) / 2, 0, 1)
        open_h = jaw * 0.13 * N
        # 嘴角高度: 画面左边嘴角 = 被拍者右嘴角
        cl = (g("mouthSmileRight") - g("mouthFrownRight")) * 0.045 * N
        cr = (g("mouthSmileLeft") - g("mouthFrownLeft")) * 0.045 * N
        L, R = (mx - half_w, my - cl), (mx + half_w, my - cr)
        curve = (smile - (g("mouthFrownLeft") + g("mouthFrownRight")) / 2) * 0.05 * N
        if open_h < 0.012 * N:                                 # 闭嘴: 一条弧线
            pts = _bez(L, (mx, my + curve), R).astype(np.int32)
            cv2.polylines(img, [pts], False, (60, 50, 150), int(0.012 * N), aa)
        else:                                                  # 张嘴: 上下两条曲线围成的多边形
            upper = _bez(L, (mx, my - 0.25 * open_h + 0.5 * curve), R)
            lower = _bez(R, (mx, my + 1.6 * open_h + curve), L)
            poly = np.vstack([upper, lower]).astype(np.int32)
            x0, y0 = poly.min(0) - 4
            x1, y1 = poly.max(0) + 4
            mouth = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
            mouth[:] = (50, 40, 110)                           # 口腔
            cv2.ellipse(mouth, P(mx - x0, my + 1.25 * open_h - y0), (int(half_w * 0.6), int(0.45 * open_h + 2)),
                        0, 0, 360, (130, 120, 235), -1, aa)    # 舌头
            cv2.polylines(mouth, [(upper - [x0, y0]).astype(np.int32)], False, (255, 255, 255),
                          int(0.035 * N), aa)                  # 上排牙齿
            mmask = np.zeros(mouth.shape[:2], np.uint8)
            cv2.fillPoly(mmask, [poly - [x0, y0]], 255, aa)
            m = mmask[..., None].astype(np.float32) / 255
            roi = img[y0:y1, x0:x1]
            img[y0:y1, x0:x1] = (mouth * m + roi * (1 - m)).astype(np.uint8)
            cv2.polylines(img, [poly], True, (60, 50, 150), int(0.010 * N), aa)

        # ---- 歪头: 整张图绕头部中心旋转 (roll 顺时针为正, OpenCV 角度逆时针为正, 所以取负) ----
        if abs(roll) > 0.5:
            M = cv2.getRotationMatrix2D((hx, hy), -float(np.clip(roll, -45, 45)), 1.0)
            img = cv2.warpAffine(img, M, (N, N), flags=cv2.INTER_LINEAR, borderValue=BG)
        if S != 1:
            img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return img


def draw_blendshape_bars(panel: np.ndarray, bs: dict[str, float],
                         keys=("jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "mouthSmileLeft",
                               "mouthSmileRight", "browInnerUp", "mouthPucker")) -> np.ndarray:
    """在面板左上角画几个 blendshape 的数值条, 方便观察 "系数 -> 卡通" 的对应关系。"""
    h = panel.shape[0]
    s = h / 480
    y = int(14 * s)
    x1, y1 = int(204 * s), int(18 * s + 15 * s * len(keys))
    panel[:y1, :x1] = (panel[:y1, :x1] * 0.35 + 255 * 0.65).astype(np.uint8)   # 半透明白底, 文字更清楚
    for k in keys:
        v = float(bs.get(k, 0.0))
        x0 = int(118 * s)
        cv2.putText(panel, k, (int(6 * s), y + int(8 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.36 * s, INK, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y), (x0 + int(80 * s), y + int(9 * s)), (200, 190, 180), -1)
        cv2.rectangle(panel, (x0, y), (x0 + int(80 * s * v), y + int(9 * s)), (90, 140, 230), -1)
        y += int(15 * s)
    return panel


def avatar_panel(renderer: AvatarRenderer, face: MeshFace | None, height: int) -> np.ndarray:
    """生成和视频帧等高的正方形头像面板; face=None 时画一张中性脸并提示没有人脸。

    绘制分辨率最多 480 (超采样后 960x960), 更高的画面直接放大 —— 卡通图都是色块, 放大几乎看不出,
    但 720p 时能省下一半以上的绘制时间。
    """
    renderer.size = min(height, 480)
    if face is None:
        panel = renderer.render({}, 0, 0, 0)
    else:
        panel = draw_blendshape_bars(renderer.render(face.blendshapes, face.yaw, face.pitch, face.roll),
                                     face.blendshapes)
    if panel.shape[0] != height:
        panel = cv2.resize(panel, (height, height), interpolation=cv2.INTER_LINEAR)
    if face is None:
        cv2.putText(panel, "no face", (int(0.38 * height), int(0.08 * height)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * height / 480, (0, 0, 200), 2, cv2.LINE_AA)
    return panel
