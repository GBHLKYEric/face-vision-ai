"""项目里常用的路径, 所有脚本都从这里取, 避免到处写死字符串。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
ASSETS = ROOT / "assets"

BUFFALO = WEIGHTS / "buffalo_l"
DET_MODEL = BUFFALO / "det_10g.onnx"          # SCRFD-10G (RetinaFace 的改进版)
LMK106_MODEL = BUFFALO / "2d106det.onnx"       # 106 点 2D 关键点
LMK3D68_MODEL = BUFFALO / "1k3d68.onnx"        # 68 点 3D 关键点
REC_MODEL = BUFFALO / "w600k_r50.onnx"         # ArcFace ResNet50 (WebFace600K)
GENDERAGE_MODEL = BUFFALO / "genderage.onnx"   # 性别 + 年龄
MESH_MODEL = WEIGHTS / "face_landmarker.task"  # MediaPipe 478 点 3D 网格
