"""One Euro 滤波器: 视频里给关键点 "去抖动"。

为什么需要?
    关键点模型逐帧独立预测, 即使人完全不动, 相邻两帧的点也会差 1~2 个像素,
    贴纸/口红就会 "抖"。最简单的办法是指数滑动平均 (EMA):
        x̂_t = α·x_t + (1-α)·x̂_{t-1}
    α 小 -> 很平滑但 "跟不上" (头一动贴纸就拖在后面); α 大 -> 跟得上但抖。
    固定的 α 无法两全。

One Euro 滤波器 (Casiez, Roussel, Vogel, CHI 2012) 的思路: 让 α 随速度自适应。
    1. 先估计速度 dx (对速度本身也做一次 EMA, 截止频率 d_cutoff)
    2. 截止频率  fc = min_cutoff + beta·|dx|
       - 静止时 |dx|≈0, fc 很低 -> 强平滑, 消除抖动
       - 快速运动时 fc 变高 -> 弱平滑, 减少延迟
    3. 把截止频率换成 EMA 系数:  τ = 1/(2π·fc),  α = 1 / (1 + τ/Δt)
    调参口诀: 先令 beta=0, 调 min_cutoff 直到静止时不抖; 再加大 beta 直到运动时不拖。
"""
from __future__ import annotations

import math

import numpy as np

from ..mesh import MeshFace, pose_from_matrix


def smoothing_factor(dt: float, cutoff) -> np.ndarray:
    """截止频率 fc (Hz) -> EMA 系数 α。fc 可以是数组 (每个坐标一个)。"""
    tau = 1.0 / (2 * math.pi * np.asarray(cutoff, dtype=np.float64))
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """对任意形状的 numpy 数组做 One Euro 滤波 (每个元素独立)。"""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self):
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x: np.ndarray, t: float, scale: float = 1.0) -> np.ndarray:
        """x: 当前观测; t: 时间 (秒); scale: 速度的单位 (例如 IOD 像素),
        速度先除以 scale 再乘 beta, 这样参数和分辨率、脸的大小无关。"""
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-3)
        # 1) 速度估计 (本身也低通一下, 否则噪声会让 fc 乱跳)
        dx = (x - self.x_prev) / dt
        a_d = smoothing_factor(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        # 2) 速度越快截止频率越高
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat) / scale
        # 3) 按自适应的 α 做 EMA
        a = smoothing_factor(dt, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()


class FaceSmoother:
    """给视频中的多张脸分别做 One Euro 平滑。

    多人时要先解决 "这一帧的第 i 张脸对应上一帧的哪一张" (数据关联),
    这里用最简单的贪心最近邻: 按脸中心距离匹配, 距离超过 1.5 个 IOD 就当作新出现的脸。
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 2.0, max_missing: int = 5):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.max_missing = max_missing
        self.tracks: list[dict] = []

    def _new_track(self, face: MeshFace) -> dict:
        return {
            "center": face.xy.mean(0),
            "lmk": OneEuroFilter(self.min_cutoff, self.beta),      # 关键点 (单位: IOD)
            "bs": OneEuroFilter(2.0, 5.0),                          # 表情系数 (0~1), 眨眼要跟得上
            "rot": OneEuroFilter(1.0, 2.0),                         # 头部旋转矩阵 R 的 9 个元素
            "missing": 0,
        }

    def __call__(self, faces: list[MeshFace], t: float) -> list[MeshFace]:
        used: set[int] = set()
        out = []
        for face in faces:
            c = face.xy.mean(0)
            iod = face.eye_distance()
            best, best_d = None, 1.5 * iod
            for k, tr in enumerate(self.tracks):
                d = float(np.linalg.norm(tr["center"] - c))
                if k not in used and d < best_d:
                    best, best_d = k, d
            if best is None:
                self.tracks.append(self._new_track(face))
                best = len(self.tracks) - 1
            used.add(best)
            tr = self.tracks[best]
            tr["center"], tr["missing"] = c, 0

            lmk = tr["lmk"](face.landmarks, t, scale=iod).astype(np.float32)
            names = list(face.blendshapes)
            bs_vals = tr["bs"](np.array([face.blendshapes[n] for n in names]), t)
            # 旋转矩阵逐元素滤波后不再是严格的旋转矩阵, 用 SVD 投影回最近的旋转矩阵 (R = U·Vᵀ)
            M = face.matrix.astype(np.float64).copy()
            R = tr["rot"](M[:3, :3].ravel(), t).reshape(3, 3)
            U, _, Vt = np.linalg.svd(R)
            M[:3, :3] = U @ Vt
            yaw, pitch, roll = pose_from_matrix(M)
            out.append(MeshFace(lmk, dict(zip(names, np.clip(bs_vals, 0, 1).tolist())),
                                M.astype(np.float32), yaw, pitch, roll))
        # 没匹配上的轨迹: 计数, 连续丢失太久就删掉 (下次出现时重新初始化, 不会从旧位置 "滑" 过来)
        for k, tr in enumerate(self.tracks):
            if k not in used:
                tr["missing"] += 1
        self.tracks = [tr for tr in self.tracks if tr["missing"] <= self.max_missing]
        return out
