"""实时人脸特效: AR 贴纸、美颜、美妆、表情驱动卡通头像。

所有特效都建立在 MediaPipe 478 点网格 (facevision.mesh) 之上:
    filters   One Euro 滤波, 视频关键点去抖
    stickers  贴纸素材生成 + 相似/透视变换锚定 + alpha 合成
    beauty    皮肤掩码、导向/双边滤波磨皮、LAB 美白、Gustafsson 局部变形 (瘦脸/大眼)
    makeup    口红 (LAB 换色保纹理)、腮红 (高斯椭圆)、眼线
    avatar    blendshape 驱动的矢量卡通脸
    pipeline  EffectPipeline: 一次检测, 按顺序叠加多个特效
"""
from .avatar import AvatarRenderer, avatar_panel
from .beauty import big_eyes, skin_mask, slim_face, smooth_skin, warp_face, whiten
from .filters import FaceSmoother, OneEuroFilter
from .makeup import blush, eyeliner, lip_mask, lipstick
from .pipeline import EFFECT_ORDER, EffectPipeline, put_label
from .stickers import STICKERS, StickerLibrary, apply_sticker, make_sticker_assets

__all__ = [
    "AvatarRenderer", "avatar_panel", "big_eyes", "skin_mask", "slim_face", "smooth_skin", "warp_face",
    "whiten", "FaceSmoother", "OneEuroFilter", "blush", "eyeliner", "lip_mask", "lipstick",
    "EFFECT_ORDER", "EffectPipeline", "put_label", "STICKERS", "StickerLibrary", "apply_sticker",
    "make_sticker_assets",
]
