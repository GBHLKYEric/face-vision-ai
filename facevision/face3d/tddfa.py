"""单张图片 3D 人脸重建: 3DDFA_V2 (Guo et al., ECCV 2020) 推理部分的精简移植。

核心思想 —— 不直接预测 3.8 万个顶点, 而是预测 62 个"参数", 再用 3DMM 把参数变成网格:

1. 3DMM (3D 形变模型, 这里用 Basel Face Model):
       S = S̄ + A_shp·α + A_exp·β
   S̄      平均脸, N=38365 个顶点的 xyz 拉成 3N 维向量
   A_shp  40 个"身份"基 (脸宽窄、鼻子高低……), α 是 40 维系数
   A_exp  10 个"表情"基 (张嘴、微笑……),       β 是 10 维系数
   这些基是对几百个 3D 扫描做 PCA 得到的, 所以 α, β 很少几个数就能描述一张脸。

2. 弱透视投影 (weak perspective): 人脸深度变化 (几厘米) 远小于到相机的距离 (几十厘米),
   透视效果可忽略, 投影近似为  V = s·R·S + t  (旋转 R + 统一缩放 s + 平移 t)。
   V 的前两行就是图像上的 (x, y), 第三行是深度。3DDFA 把 s·R (3x3) 和 t (3x1)
   拼成 3x4 的相机矩阵 P, 共 12 个数。

3. MobileNetV1 从 120x120 人脸图直接回归 12 + 40 + 10 = 62 个参数
   (训练时参数做了标准化, 推理时要乘 std 加 mean 还原)。

4. 参数代回公式: 只算 68 个关键点对应的那几行 (很快) 得到稀疏关键点,
   算全部行得到稠密网格; 再从 120x120 裁剪图坐标映射回原图坐标。

网络结构和权重来自 3DDFA_V2 (MIT License, https://github.com/cleardusk/3DDFA_V2),
这里只依赖它的权重/数据文件 (pth, pkl), 不依赖它的代码和需要编译的 Cython 模块。
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import cached_property
from math import asin, atan2, cos, sqrt

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..paths import WEIGHTS

TDDFA_DIR = WEIGHTS / "3ddfa_v2" / "3DDFA_V2-master"
CKPT_MB1 = TDDFA_DIR / "weights" / "mb1_120x120.pth"
BFM_PKL = TDDFA_DIR / "configs" / "bfm_noneck_v3.pkl"        # 去掉脖子的 BFM, 38365 个顶点
TRI_PKL = TDDFA_DIR / "configs" / "tri.pkl"                  # 为 noneck 版重建的三角面片
PARAM_STATS_PKL = TDDFA_DIR / "configs" / "param_mean_std_62d_120x120.pkl"


# ---------------------------------------------------------------------------
# 网络: MobileNetV1
# 改编自 3DDFA_V2 models/mobilenet_v1.py (MIT License;
# Copyright (c) 2017 Yang Lu, modified by cleardusk)。层名保持一致, 这样权重能直接加载。
# ---------------------------------------------------------------------------
class DepthWiseBlock(nn.Module):
    """深度可分离卷积 = 3x3 逐通道卷积 (只看空间) + 1x1 卷积 (只混合通道)。
    计算量约是普通 3x3 卷积的 1/8, 这是 MobileNet 快的原因。"""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.conv_dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn_dw = nn.BatchNorm2d(cin)
        self.conv_sep = nn.Conv2d(cin, cout, 1, 1, 0, bias=False)
        self.bn_sep = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn_dw(self.conv_dw(x)))
        return self.relu(self.bn_sep(self.conv_sep(x)))


class MobileNetV1(nn.Module):
    # (层名, 输入通道, 输出通道, 步长): 120x120 输入经过 5 次步长 2 -> 4x4 特征图
    BLOCKS = [("dw2_1", 32, 64, 1), ("dw2_2", 64, 128, 2),
              ("dw3_1", 128, 128, 1), ("dw3_2", 128, 256, 2),
              ("dw4_1", 256, 256, 1), ("dw4_2", 256, 512, 2),
              ("dw5_1", 512, 512, 1), ("dw5_2", 512, 512, 1), ("dw5_3", 512, 512, 1),
              ("dw5_4", 512, 512, 1), ("dw5_5", 512, 512, 1), ("dw5_6", 512, 1024, 2),
              ("dw6", 1024, 1024, 1)]

    def __init__(self, widen_factor: float = 1.0, num_params: int = 62):
        super().__init__()
        w = lambda c: int(c * widen_factor)  # noqa: E731  (mb05 模型把所有通道数减半)
        self.conv1 = nn.Conv2d(3, w(32), 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(w(32))
        self.relu = nn.ReLU(inplace=True)
        for name, cin, cout, stride in self.BLOCKS:
            setattr(self, name, DepthWiseBlock(w(cin), w(cout), stride))
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w(1024), num_params)  # 全局池化后接全连接, 直接输出 62 个参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        for name, *_ in self.BLOCKS:
            x = getattr(self, name)(x)
        return self.fc(self.avgpool(x).flatten(1))


def load_mobilenet(ckpt_path=CKPT_MB1, widen_factor: float = 1.0) -> MobileNetV1:
    model = MobileNetV1(widen_factor)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)["state_dict"]
    state = {}
    for k, v in ckpt.items():
        k = k.replace("module.", "")          # 多卡训练时保存的权重带 "module." 前缀
        if k.startswith("fc_lm."):            # 训练时的辅助分支 (直接回归 68 点), 推理用不到
            continue
        state[k.replace("fc_param.", "fc.")] = v
    model.load_state_dict(state)               # strict=True: 层名对不上会直接报错
    return model.eval()


# ---------------------------------------------------------------------------
# 裁剪: 检测框 -> 正方形 ROI
# ---------------------------------------------------------------------------
def roi_from_bbox(bbox) -> np.ndarray:
    """3DDFA_V2 的裁剪策略: 以框中心 (略往下移) 为中心, 取 1.58 倍框大小的正方形。

    检测框通常只框到眉毛, 而 3DMM 网格包含额头和下巴, 所以要放大并下移一点。
    """
    left, top, right, bottom = bbox[:4]
    old_size = (right - left + bottom - top) / 2
    center_x = right - (right - left) / 2.0
    center_y = bottom - (bottom - top) / 2.0 + old_size * 0.14
    size = int(old_size * 1.58)
    x1, y1 = center_x - size / 2, center_y - size / 2
    return np.array([x1, y1, x1 + size, y1 + size], dtype=np.float32)


def roi_from_landmarks(pts) -> np.ndarray:
    """由关键点 (2xK 或 3xK) 求 ROI: 外接正方形的对角线长度作为边长。
    用于视频跟踪 (用上一帧的关键点裁这一帧) 和 AFLW2000 评测 (用 GT 关键点裁)。"""
    pts = np.asarray(pts)
    x1, y1, x2, y2 = pts[0].min(), pts[1].min(), pts[0].max(), pts[1].max()
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1)
    llength = sqrt(side ** 2 + side ** 2)  # 外接正方形的对角线 = 边长·√2
    return np.array([cx - llength / 2, cy - llength / 2, cx + llength / 2, cy + llength / 2], dtype=np.float32)


def crop_img(img: np.ndarray, roi_box) -> np.ndarray:
    """裁出 roi_box, 超出图像的部分补黑 (保证输出正好是 roi 大小)。"""
    h, w = img.shape[:2]
    sx, sy, ex, ey = [int(round(float(v))) for v in roi_box]
    out = np.zeros((ey - sy, ex - sx) + img.shape[2:], dtype=img.dtype)
    x1, y1, x2, y2 = max(sx, 0), max(sy, 0), min(ex, w), min(ey, h)
    if x2 > x1 and y2 > y1:
        out[y1 - sy:y2 - sy, x1 - sx:x2 - sx] = img[y1:y2, x1:x2]
    return out


# ---------------------------------------------------------------------------
# 参数解析 / 坐标变换 / 姿态
# ---------------------------------------------------------------------------
def parse_params(params: np.ndarray):
    """62 维 = 12 (相机矩阵 P=[sR|t], 按行展开) + 40 (形状 α) + 10 (表情 β)。"""
    P = params[:12].reshape(3, 4)
    alpha_shp = params[12:52].reshape(-1, 1)
    alpha_exp = params[52:62].reshape(-1, 1)
    return P, alpha_shp, alpha_exp


def similar_transform(pts3d: np.ndarray, roi_box, size: int = 120) -> np.ndarray:
    """把 120x120 裁剪图坐标系里的点映射回原图。

    网络坐标系 y 轴朝上 (数学习惯), 图像 y 轴朝下, 所以先 y -> size - y;
    "-1" 是因为训练数据来自 MATLAB (下标从 1 开始)。
    然后按 roi 大小等比缩放 + 平移; z 按同样比例缩放, 并平移到最小值为 0 (z 越大离相机越近)。
    """
    pts3d = pts3d.copy()
    pts3d[0] -= 1
    pts3d[2] -= 1
    pts3d[1] = size - pts3d[1]
    sx, sy, ex, ey = roi_box
    scale_x, scale_y = (ex - sx) / size, (ey - sy) / size
    pts3d[0] = pts3d[0] * scale_x + sx
    pts3d[1] = pts3d[1] * scale_y + sy
    pts3d[2] *= (scale_x + scale_y) / 2
    pts3d[2] -= pts3d[2].min()
    return pts3d.astype(np.float32)


def P2sRt(P: np.ndarray):
    """把相机矩阵 P = [sR | t] 拆成 缩放 s、旋转 R、平移 t。

    网络输出的 sR 是 9 个自由数, 不保证正交, 所以: 前两行各自的模长 ≈ s,
    归一化后作为 R 的前两行, 第三行用叉乘补上 (保证右手系)。
    """
    t = P[:, 3]
    r1, r2 = P[0, :3], P[1, :3]
    s = (np.linalg.norm(r1) + np.linalg.norm(r2)) / 2.0
    r1 = r1 / np.linalg.norm(r1)
    r2 = r2 / np.linalg.norm(r2)
    R = np.stack([r1, r2, np.cross(r1, r2)])
    return s, R, t


def matrix2angle(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 -> 欧拉角 (yaw, pitch, roll), 弧度。R ≈ Rz(roll)·Ry(-yaw)·Rx(pitch)。

    |R[2,0]| ≈ 1 时 yaw = ±90°, 出现"万向节死锁", 单独处理。
    """
    if R[2, 0] > 0.998:
        yaw, roll = np.pi / 2, 0.0
        pitch = roll + atan2(-R[0, 1], -R[0, 2])
    elif R[2, 0] < -0.998:
        yaw, roll = -np.pi / 2, 0.0
        pitch = -roll + atan2(R[0, 1], R[0, 2])
    else:
        yaw = asin(R[2, 0])
        pitch = atan2(R[2, 1] / cos(yaw), R[2, 2] / cos(yaw))
        roll = atan2(R[1, 0] / cos(yaw), R[0, 0] / cos(yaw))
    return yaw, pitch, roll


@dataclass
class Face3DResult:
    params: np.ndarray        # (62,) 反标准化后的 3DMM 参数
    vertices: np.ndarray      # (3, N) 稠密网格, 原图坐标: x 向右, y 向下, z 越大越靠近相机
    landmarks68: np.ndarray   # (68, 3) 稀疏关键点 (同上坐标系; 注意 z 的零点和稠密网格不同)
    pose_deg: tuple           # (yaw, pitch, roll) 角度制
    R: np.ndarray             # (3, 3) 旋转 (y 轴朝上的模型坐标系 -> 相机坐标系)
    t: np.ndarray             # (3,) 平移 (120x120 裁剪图坐标系)
    s: float                  # 缩放
    roi_box: np.ndarray       # (4,) 送入网络的正方形裁剪区域 x1, y1, x2, y2

    @property
    def yaw(self) -> float:
        return self.pose_deg[0]


class TDDFA3D:
    """3DDFA_V2 (MobileNetV1, 120x120) 推理: 图像 + 人脸框 -> 3DMM 参数 -> 3D 网格。"""

    def __init__(self, ckpt_path=CKPT_MB1, bfm_path=BFM_PKL, tri_path=TRI_PKL,
                 stats_path=PARAM_STATS_PKL, widen_factor: float = 1.0, size: int = 120,
                 shape_dim: int = 40, exp_dim: int = 10, threads: int | None = None):
        if threads:
            # PyTorch 默认用满所有核; 这个小网络 1~4 线程就够了, 机器繁忙时线程太多反而互相等待变慢
            torch.set_num_threads(threads)
        self.size = size
        self.model = load_mobilenet(ckpt_path, widen_factor)

        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.param_mean = stats["mean"].astype(np.float32)
        self.param_std = stats["std"].astype(np.float32)

        # BFM: 所有顶点的 xyz 按 [x0,y0,z0, x1,y1,z1, ...] 拉成一列, 所以后面要 reshape(3, -1, order="F")
        with open(bfm_path, "rb") as f:
            bfm = pickle.load(f)
        self.u = bfm["u"].astype(np.float32)                         # (3N, 1)   平均脸 S̄
        self.w_shp = bfm["w_shp"].astype(np.float32)[:, :shape_dim]  # (3N, 40)  身份基 A_shp
        self.w_exp = bfm["w_exp"].astype(np.float32)[:, :exp_dim]    # (3N, 10)  表情基 A_exp
        with open(tri_path, "rb") as f:
            self.tri = np.ascontiguousarray(pickle.load(f).T).astype(np.int32)  # (T, 3) 三角形顶点下标

        # 68 个关键点在 3N 向量里对应的 204 行; 只取这些行算, 比算整个网格快 500 多倍
        self.keypoints = bfm["keypoints"].astype(np.int64)
        self.u_base = self.u[self.keypoints].reshape(-1, 1)
        self.w_shp_base = self.w_shp[self.keypoints]
        self.w_exp_base = self.w_exp[self.keypoints]

    @property
    def num_vertices(self) -> int:
        return self.u.shape[0] // 3

    @cached_property
    def mirror_index(self) -> np.ndarray:
        """每个顶点左右对称位置的顶点下标: 把平均脸沿 x 镜像, 再找最近的顶点。
        用于纹理补全 (侧脸照片里看不到的那半边脸, 借用另一半边的颜色)。"""
        from scipy.spatial import cKDTree
        u = self.u.reshape(3, -1, order="F").T
        mirrored = u.copy()
        mirrored[:, 0] = 2 * u[:, 0].mean() - u[:, 0]
        return cKDTree(u).query(mirrored)[1].astype(np.int64)

    # ---------- 网络推理 ----------
    @torch.inference_mode()
    def predict_params(self, img: np.ndarray, roi_box) -> np.ndarray:
        """裁剪 -> 缩放到 120x120 -> 归一化 -> CNN -> 62 维参数 (已还原标准化)。"""
        crop = crop_img(img, roi_box)
        crop = cv2.resize(crop, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        # 注意: 3DDFA 训练时输入就是 BGR 顺序 (cv2.imread 的默认), 不需要转 RGB
        x = (crop.astype(np.float32) - 127.5) / 128.0
        x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None]))
        out = self.model(x)[0].numpy().astype(np.float32)
        return out * self.param_std + self.param_mean

    # ---------- 3DMM 重建 ----------
    def reconstruct(self, params: np.ndarray, roi_box, dense: bool = True, frontal: bool = False) -> np.ndarray:
        """V = s·R·(S̄ + A_shp·α + A_exp·β) + t, 再映射回原图坐标。返回 (3, N)。

        dense=False 只算 68 个关键点; frontal=True 把 R 换成单位阵, 得到"摆正"的正脸网格。
        """
        P, alpha_shp, alpha_exp = parse_params(params)
        if dense:
            u, w_shp, w_exp = self.u, self.w_shp, self.w_exp
        else:
            u, w_shp, w_exp = self.u_base, self.w_shp_base, self.w_exp_base
        shape = (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order="F")  # (3, N) 模型坐标系
        sR, t = P[:, :3], P[:, 3:]
        if frontal:
            sR = np.float32(P2sRt(P)[0]) * np.eye(3, dtype=np.float32)
        return similar_transform(sR @ shape + t, roi_box, self.size)

    def fit_roi(self, img: np.ndarray, roi_box) -> Face3DResult:
        roi_box = np.asarray(roi_box, dtype=np.float32)
        params = self.predict_params(img, roi_box)
        vertices = self.reconstruct(params, roi_box, dense=True)
        lmk = self.reconstruct(params, roi_box, dense=False)
        s, R, t = P2sRt(params[:12].reshape(3, 4))
        pose = tuple(float(np.degrees(a)) for a in matrix2angle(R))
        return Face3DResult(params=params, vertices=vertices, landmarks68=lmk.T.copy(), pose_deg=pose,
                            R=R, t=t, s=float(s), roi_box=roi_box)

    def fit(self, img: np.ndarray, bbox) -> Face3DResult:
        """img: BGR 原图; bbox: 检测框 (x1, y1, x2, y2[, score])。"""
        return self.fit_roi(img, roi_from_bbox(bbox))

    def refit(self, img: np.ndarray, result: Face3DResult) -> Face3DResult:
        """用第一次预测的 68 点重新裁剪再预测一次 (3DDFA_V2 视频跟踪里的做法), 裁剪更贴合人脸。"""
        return self.fit_roi(img, roi_from_landmarks(result.landmarks68.T))

    def frontal_vertices(self, result: Face3DResult) -> np.ndarray:
        """去掉头部旋转后的稠密网格 (3, N), 尺度和原图一致, 用于从任意角度渲染。"""
        return self.reconstruct(result.params, result.roi_box, dense=True, frontal=True)
