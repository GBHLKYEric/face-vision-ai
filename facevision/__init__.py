"""FaceVision: 人脸检测、识别与特效 (教学项目)。"""
from .analyzer import Face, FaceAnalyzer
from .align import norm_crop, umeyama
from .detector import SCRFD
from .recognizer import ArcFaceONNX, cosine_similarity

__all__ = ["Face", "FaceAnalyzer", "SCRFD", "ArcFaceONNX", "norm_crop", "umeyama", "cosine_similarity"]
__version__ = "1.0.0"
