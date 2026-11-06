"""特效流水线: 每帧只跑一次 MediaPipe, 然后按固定顺序叠加选中的特效。

为什么顺序是固定的?
    1. 先做 "逐像素的颜色处理" (磨皮、美白、口红、腮红、眼线) —— 它们用关键点算掩码,
       必须在几何变形之前做, 否则关键点和变形后的画面对不上。
    2. 再做 "几何变形" (瘦脸、大眼) —— 已经画好的妆会跟着一起变形, 不会错位。
    3. 最后贴 "贴纸" 和画 "网格" —— 它们盖在最上层。
    4. "avatar" 不改原图, 而是在右边拼一个卡通头像面板。
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from ..mesh import FaceMesh, MeshFace, draw_mesh
from ..utils import disable_power_throttling  # noqa: F401  (tools 从这里导入)
from . import beauty, makeup
from .avatar import AvatarRenderer, avatar_panel
from .filters import FaceSmoother
from .stickers import STICKERS, StickerLibrary

EFFECT_ORDER = ["smooth", "whiten", "lipstick", "blush", "eyeliner", "slim", "big_eyes",
                "sunglasses", "hat", "cat_ears", "mustache", "mesh", "avatar"]

DEFAULT_PARAMS: dict[str, dict] = {
    "smooth": {"strength": 0.7, "method": "guided"},
    "whiten": {"strength": 0.3},
    "lipstick": {"strength": 0.6, "color": (60, 30, 200)},      # BGR: 正红偏玫瑰
    "blush": {"strength": 0.5, "color": (150, 120, 245)},
    "eyeliner": {"strength": 0.8},
    "slim": {"strength": 0.6},
    "big_eyes": {"strength": 0.5},
    "sticker": {"mode": "3d"},
}


class EffectPipeline:
    """用法:
        pipe = EffectPipeline(["smooth", "lipstick", "sunglasses"], mode="video")
        out = pipe.process(frame_bgr, timestamp_ms)
        pipe.toggle("hat")
    """

    def __init__(self, effects=("sunglasses",), mode: str = "video", num_faces: int = 2,
                 smoothing: bool = True, params: dict | None = None):
        unknown = set(effects) - set(EFFECT_ORDER)
        if unknown:
            raise ValueError(f"未知特效 {sorted(unknown)}, 可选: {EFFECT_ORDER}")
        self.mesh = FaceMesh(mode=mode, num_faces=num_faces)
        self.mode = mode
        self.effects: set[str] = set(effects)
        self.params = {k: dict(v) for k, v in DEFAULT_PARAMS.items()}
        for k, v in (params or {}).items():
            self.params.setdefault(k, {}).update(v)
        self.smoother = FaceSmoother() if (smoothing and mode == "video") else None
        self.stickers = StickerLibrary()
        self.avatar = AvatarRenderer()
        self.faces: list[MeshFace] = []
        self.timings: dict[str, float] = {}     # 上一帧各阶段耗时 (毫秒)

    def toggle(self, name: str) -> bool:
        """开/关某个特效, 返回切换后的状态。"""
        if name in self.effects:
            self.effects.discard(name)
            return False
        self.effects.add(name)
        return True

    def detect(self, frame: np.ndarray, timestamp_ms: int | None = None) -> list[MeshFace]:
        t0 = time.perf_counter()
        faces = self.mesh.detect(frame, timestamp_ms)
        self.timings["landmarks"] = (time.perf_counter() - t0) * 1000
        if self.smoother is not None:
            t1 = time.perf_counter()
            ts = (timestamp_ms if timestamp_ms is not None else self.mesh._last_ts) / 1000.0
            faces = self.smoother(faces, ts)
            self.timings["one_euro"] = (time.perf_counter() - t1) * 1000
        self.faces = faces
        return faces

    def apply(self, frame: np.ndarray, faces: list[MeshFace]) -> np.ndarray:
        """对已检测到的人脸应用特效 (不修改输入 frame)。"""
        out = frame.copy()
        p = self.params
        skins: dict[int, np.ndarray] = {}                       # 皮肤掩码: 每张脸只算一次, 多个特效共用

        def skin(i: int, f: MeshFace) -> np.ndarray:
            if i not in skins:
                t = time.perf_counter()
                skins[i] = beauty.skin_mask(f, out.shape)
                self.timings["skin_mask"] = self.timings.get("skin_mask", 0) + (time.perf_counter() - t) * 1000
            return skins[i]

        self.timings.pop("skin_mask", None)
        for name in EFFECT_ORDER:
            if name not in self.effects or name == "avatar":
                continue
            if name == "big_eyes" and "slim" in self.effects:
                continue                                         # 已和瘦脸合并成一次 remap
            t = time.perf_counter()
            for i, f in enumerate(faces):
                if name == "smooth":
                    beauty.smooth_skin(out, f, mask=skin(i, f), **p["smooth"])
                elif name == "whiten":
                    beauty.whiten(out, f, mask=skin(i, f), **p["whiten"])
                elif name == "lipstick":
                    makeup.lipstick(out, f, **p["lipstick"])
                elif name == "blush":
                    makeup.blush(out, f, skin=skin(i, f), **p["blush"])
                elif name == "eyeliner":
                    makeup.eyeliner(out, f, **p["eyeliner"])
                elif name in ("slim", "big_eyes"):
                    slim = p["slim"]["strength"] if "slim" in self.effects else 0.0
                    eyes = p["big_eyes"]["strength"] if "big_eyes" in self.effects else 0.0
                    beauty.warp_face(out, f, slim=slim, eyes=eyes)
                elif name in STICKERS:
                    self.stickers.apply(out, [f], name, **p["sticker"])
                elif name == "mesh":
                    draw_mesh(out, f)
            self.timings[name] = (time.perf_counter() - t) * 1000

        if "avatar" in self.effects:
            t = time.perf_counter()
            panel = avatar_panel(self.avatar, faces[0] if faces else None, out.shape[0])
            out = np.hstack([out, panel])
            self.timings["avatar"] = (time.perf_counter() - t) * 1000
        return out

    def process(self, frame: np.ndarray, timestamp_ms: int | None = None) -> np.ndarray:
        """检测 + 应用特效, 返回新图像 (开启 avatar 时宽度会增加一个正方形面板)。"""
        t0 = time.perf_counter()
        faces = self.detect(frame, timestamp_ms)
        out = self.apply(frame, faces)
        self.timings["total"] = (time.perf_counter() - t0) * 1000
        return out

    def close(self):
        self.mesh.close()




def put_label(img: np.ndarray, text: str, org=(10, 30), scale: float = 0.7) -> np.ndarray:
    """带黑色描边的白字, 在任何背景上都看得清。
    (OpenCV 的字间距会随线宽变化, 所以描边用 "同样线宽的黑字向 8 个方向各偏移 1 像素" 来画。)"""
    t = 1 if scale < 0.8 else 2
    x, y = org
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx or dy:
                cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), t, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), t, cv2.LINE_AA)
    return img
