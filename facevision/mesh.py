"""MediaPipe 人脸网格: 478 个 3D 关键点 + 52 个表情系数 (blendshapes) + 头部姿态矩阵。

和 landmark.py 的 106 点模型比, MediaPipe Face Landmarker 多给了三样东西:
  1. 478 个点 (468 个网格点 + 每只眼 5 个虹膜点), 覆盖整张脸, 适合做美颜/贴纸/上妆;
     每个点还带一个相对深度 z (越小越靠近相机, 尺度和 x 相同), 所以可以近似看成 3D 点。
  2. 52 个 blendshape 系数 (ARKit 定义, 取值 0~1): 例如 jawOpen=张嘴程度,
     eyeBlinkLeft=左眼闭合程度, mouthSmileLeft=嘴角上扬程度。用来驱动卡通头像。
  3. 4x4 "面部变换矩阵": 把标准 3D 人脸模型 (单位 cm) 摆到相机坐标系里的刚体变换,
     左上 3x3 是旋转 R, 最后一列是平移 t。从 R 可以直接读出 yaw/pitch/roll。

两种运行模式:
  - IMAGE: 每张图独立检测, 适合照片;
  - VIDEO: 需要单调递增的时间戳, 内部会利用上一帧的结果做跟踪, 更快也更稳定。

约定: 本项目里 "左/右" 一律指 **图像中的左/右** (与 landmark.py 的 LMK106 一致)。
MediaPipe 官方命名用的是 "被拍者自己的左/右", 正好相反, 读官方文档时注意。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

os.environ.setdefault("GLOG_minloglevel", "2")  # 少打印一些 MediaPipe 的 C++ 日志

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

from .paths import MESH_MODEL

# ---------------------------------------------------------------------------
# 478 点拓扑里的部位下标 (已画图逐组核对, 见 outputs/effects/mesh_groups.jpg)。
# 闭合轮廓都按顺时针/逆时针顺序排列, 可以直接 cv2.fillPoly。
# ---------------------------------------------------------------------------
MESH: dict[str, list[int]] = {
    # 脸部外轮廓 (从额头顶 10 开始绕一圈)
    "face_oval": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
                  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
                  54, 103, 67, 109],
    # 嘴唇: 外轮廓 (从左嘴角 61 开始, 先上唇后下唇) 和内轮廓 (张嘴时的口腔边界)
    "lips_outer": [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84,
                   181, 91, 146],
    "lips_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87,
                   178, 88, 95],
    # 眼睛轮廓 (从外眼角开始, 先上眼睑后下眼睑)
    "left_eye": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "right_eye": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    # 上眼睑 (外眼角 -> 内眼角), 画眼线用
    "left_upper_lid": [33, 246, 161, 160, 159, 158, 157, 173, 133],
    "right_upper_lid": [263, 466, 388, 387, 386, 385, 384, 398, 362],
    # 虹膜: 第一个是瞳孔中心, 后 4 个是虹膜边缘
    "left_iris": [468, 469, 470, 471, 472],
    "right_iris": [473, 474, 475, 476, 477],
    # 眉毛 (上沿从外到内, 再沿下沿回来)
    "left_brow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_brow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    # 鼻子: 鼻梁中线 (两眼之间 168 -> 鼻尖 1 -> 鼻底 2) 和鼻翼外轮廓
    "nose_bridge": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    "nose_outline": [98, 97, 2, 326, 327, 294, 278, 344, 440, 275, 4, 45, 220, 115, 48, 64],
    # 脸颊 (颧骨下方 "苹果肌" 一圈, 腮红画在这里; 中心点分别是 50 / 280)
    "left_cheek": [117, 118, 101, 36, 205, 187, 123],
    "right_cheek": [330, 347, 346, 352, 411, 425, 266],
    # 额头 (眉毛以上到发际线)
    "forehead": [10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104, 103, 67, 109],
    # 下颌线 (瘦脸时往里推的那些点), 从耳下到下巴
    "left_jaw": [132, 58, 172, 136, 150, 149, 176, 148],
    "right_jaw": [361, 288, 397, 365, 379, 378, 400, 377],
}

# 常用的单个关键点
KEY: dict[str, int] = {
    "forehead_top": 10, "forehead_center": 151, "between_eyes": 168,
    "nose_tip": 1, "nose_bottom": 2, "philtrum": 164,
    "upper_lip_top": 0, "upper_lip_inner": 13, "lower_lip_inner": 14, "lower_lip_bottom": 17,
    "mouth_left": 61, "mouth_right": 291, "chin": 152,
    "left_eye_outer": 33, "left_eye_inner": 133, "right_eye_outer": 263, "right_eye_inner": 362,
    "left_eye_top": 159, "left_eye_bottom": 145, "right_eye_top": 386, "right_eye_bottom": 374,
    "left_iris_center": 468, "right_iris_center": 473,
    "left_face": 234, "right_face": 454, "left_cheek": 50, "right_cheek": 280,
}


@dataclass
class MeshFace:
    """一张脸的网格结果。所有坐标都是原图像素坐标。"""
    landmarks: np.ndarray                  # (478, 3) float32: x, y 像素; z 相对深度 (同 x 尺度, 越小越近)
    blendshapes: dict[str, float] = field(default_factory=dict)  # 52 个表情系数, 0~1
    matrix: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))  # 4x4 头部位姿
    yaw: float = 0.0     # 左右转头 (度), 正 = 脸朝向图像右侧
    pitch: float = 0.0   # 抬头低头 (度), 正 = 抬头
    roll: float = 0.0    # 歪头 (度), 正 = 在画面上顺时针转

    @property
    def xy(self) -> np.ndarray:
        return self.landmarks[:, :2]

    @property
    def bbox(self) -> np.ndarray:
        """由关键点算出的外接框 (x1, y1, x2, y2)。"""
        x1, y1 = self.xy.min(0)
        x2, y2 = self.xy.max(0)
        return np.array([x1, y1, x2, y2], np.float32)

    def pts(self, group: str) -> np.ndarray:
        """取某个部位的 2D 点, 例如 face.pts('lips_outer') -> (20, 2)。"""
        return self.xy[MESH[group]]

    def center(self, group: str) -> np.ndarray:
        """某个部位的中心 (多个点取平均, 比单个点更抗抖动)。"""
        return self.xy[MESH[group]].mean(0)

    def eye_distance(self) -> float:
        """两眼中心的距离 (像素)。特效的尺寸都以它为单位, 这样脸大脸小都适用。"""
        return float(np.linalg.norm(self.center("right_eye") - self.center("left_eye")))


def pose_from_matrix(M: np.ndarray) -> tuple[float, float, float]:
    """从 4x4 面部变换矩阵读出 (yaw, pitch, roll), 单位度。

    MediaPipe 的相机坐标系: x 向右, y 向上, z 指向观众 (OpenGL 约定)。
    R 的第 3 列是 "脸的正前方" 在相机坐标系里的方向 f:
        yaw   = atan2(f_x, f_z)   脸往右转, f 向 +x 偏
        pitch = asin(f_y)         抬头, f 向 +y 偏
    R 的第 1 列是 "脸的水平轴" (左眼 -> 右眼) 方向 a; 换成图像坐标 (y 向下) 后
        roll  = atan2(-a_y, a_x)  画面上的顺时针为正
    """
    R = np.asarray(M, dtype=np.float64)[:3, :3]
    f = R[:, 2]
    a = R[:, 0]
    yaw = np.degrees(np.arctan2(f[0], f[2]))
    pitch = np.degrees(np.arcsin(np.clip(f[1], -1.0, 1.0)))
    roll = np.degrees(np.arctan2(-a[1], a[0]))
    return float(yaw), float(pitch), float(roll)


class FaceMesh:
    """MediaPipe FaceLandmarker 的简单封装。

    用法:
        mesh = FaceMesh(mode="image", num_faces=2)
        faces = mesh.detect(img_bgr)                 # list[MeshFace]
        mesh = FaceMesh(mode="video")
        faces = mesh.detect(frame_bgr, timestamp_ms)  # 视频模式要给时间戳 (毫秒, 单调递增)
    """

    def __init__(self, model_path=MESH_MODEL, mode: str = "image", num_faces: int = 1,
                 min_detection_conf: float = 0.5, min_presence_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        assert mode in ("image", "video")
        self.mode = mode
        running_mode = vision.RunningMode.IMAGE if mode == "image" else vision.RunningMode.VIDEO
        # 用 model_asset_buffer 读入字节, 避免 Windows 下中文/长路径的问题
        with open(model_path, "rb") as f:
            buf = f.read()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=buf),
            running_mode=running_mode,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_conf,
            min_face_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_tracking_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, img_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[MeshFace]:
        h, w = img_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.mode == "video":
            if timestamp_ms is None:
                timestamp_ms = self._last_ts + 33           # 没给时间戳就假设 30 FPS
            timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)  # 必须严格递增
            self._last_ts = timestamp_ms
            res = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        else:
            res = self.landmarker.detect(mp_img)

        faces = []
        for i, lms in enumerate(res.face_landmarks):
            # 归一化坐标 [0,1] -> 像素坐标; z 和 x 同尺度, 所以也乘图像宽度
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            bs = {}
            if res.face_blendshapes:
                bs = {c.category_name: float(c.score) for c in res.face_blendshapes[i]}
            M = np.eye(4, dtype=np.float32)
            if res.facial_transformation_matrixes:
                M = np.asarray(res.facial_transformation_matrixes[i], dtype=np.float32)
            yaw, pitch, roll = pose_from_matrix(M)
            faces.append(MeshFace(pts, bs, M, yaw, pitch, roll))
        return faces

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------
TESSELATION = [(c.start, c.end) for c in vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION]

GROUP_COLORS = {  # BGR
    "face_oval": (255, 255, 255), "lips_outer": (60, 60, 255), "lips_inner": (180, 120, 255),
    "left_eye": (255, 200, 0), "right_eye": (255, 200, 0),
    "left_iris": (0, 255, 255), "right_iris": (0, 255, 255),
    "left_brow": (0, 160, 255), "right_brow": (0, 160, 255),
    "nose_bridge": (0, 255, 0), "nose_outline": (0, 200, 120),
    "left_cheek": (200, 80, 255), "right_cheek": (200, 80, 255),
    "forehead": (255, 120, 120), "left_jaw": (120, 255, 255), "right_jaw": (120, 255, 255),
}
OPEN_GROUPS = {"nose_bridge", "left_upper_lid", "right_upper_lid", "left_jaw", "right_jaw"}


def draw_mesh(img: np.ndarray, face: MeshFace, tessellation: bool = True, groups: bool = True,
              thickness: int = 1) -> np.ndarray:
    """在图上画网格三角面 (细灰线) 和各部位轮廓 (彩色), 用来核对下标是否正确。"""
    out = img
    xy = np.round(face.xy).astype(np.int32)
    if tessellation:
        lines = np.array([[xy[a], xy[b]] for a, b in TESSELATION], np.int32)
        cv2.polylines(out, lines, False, (170, 170, 170), 1, cv2.LINE_AA)
    if groups:
        for name, color in GROUP_COLORS.items():
            p = xy[MESH[name]]
            if name.endswith("iris"):
                r = max(1, int(np.linalg.norm(p[1:] - p[0], axis=1).mean()))
                cv2.circle(out, tuple(int(v) for v in p[0]), r, color, thickness, cv2.LINE_AA)
                cv2.circle(out, tuple(int(v) for v in p[0]), 1, color, -1)
                continue
            cv2.polylines(out, [p], name not in OPEN_GROUPS, color, thickness, cv2.LINE_AA)
    return out
