"""StarGAN: 用一个生成器完成多种人脸属性编辑 (发色 / 性别 / 年龄)。

论文: Choi et al., "StarGAN: Unified Generative Adversarial Networks for Multi-Domain
Image-to-Image Translation", CVPR 2018。
Generator / Discriminator 结构和 CelebA 预处理改编自官方实现 https://github.com/yunjey/stargan
(MIT License, Copyright (c) 2017 Yunjey Choi)。层的顺序和参数名与官方完全一致,
所以官方 CelebA 128x128 权重 (200000-G.ckpt / 200000-D.ckpt) 可以 strict=True 直接加载。

核心思想 (GAN = 两个网络的博弈):
  - 生成器 G(x, c): 输入一张脸 x 和"目标属性"向量 c (5 维 0/1), 输出改了属性的脸。
    条件生成的做法很朴素: 把 c 的每一维复制成一张 128x128 的常数图, 和 RGB 拼起来 (3+5=8 通道)。
  - 判别器 D(x) 有两个输出头:
      out_src: 这张图像不像真实照片 (PatchGAN: 输出 2x2 个分数, 每个分数只看图像的一块)
      out_cls: 这张图有哪些属性 (辅助分类器, AC-GAN 的思路), 逼着 G 真的把属性改到位
  - 循环一致 (cycle consistency): G(G(x, c_trg), c_org) ≈ x, 保证只改属性、不乱改身份和背景。
训练细节见 tools/train_stargan.py。
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from ..align import umeyama
from ..paths import WEIGHTS

# 官方 CelebA 模型使用的 5 个属性, 顺序不能变 (标签向量的第 i 维就是第 i 个属性)
ATTRS: tuple[str, ...] = ("Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young")
HAIR_ATTRS: tuple[str, ...] = ("Black_Hair", "Blond_Hair", "Brown_Hair")

ALIGNED_W, ALIGNED_H = 178, 218   # CelebA 官方对齐图 (img_align_celeba) 的尺寸
CROP_SIZE = 178                   # 官方预处理: CenterCrop(178) -> 去掉上下各 20 像素
CROP_TOP = (ALIGNED_H - CROP_SIZE) // 2
IMAGE_SIZE = 128                  # 再 Resize 到 128x128 送进网络

# CelebA 对齐图 (178x218) 中 5 个关键点的平均位置: 左眼、右眼、鼻尖、左嘴角、右嘴角 (x, y)。
# 获得方法: 用本项目的 SCRFD 检测 CelebA valid 划分前 400 张对齐图 (全部检出), 取均值
# (脚本见 _scratch/sg_kps.py; 各点标准差 1~6 像素)。结果与 CelebA 自带
# list_landmarks_align_celeba.txt 的平均值 ((69,111) (108,111) (88,135) (72,152) (106,152)) 基本一致。
# 任意照片只要用相似变换把 5 点对到这里, 就得到和训练数据"一样摆放"的人脸。
CELEBA_KPS = np.array(
    [[69.58, 111.59],
     [108.26, 111.66],
     [88.72, 135.73],
     [72.56, 152.72],
     [105.80, 152.82]], dtype=np.float32)

STARGAN_DIR = WEIGHTS / "stargan"
G_CKPT = STARGAN_DIR / "200000-G.ckpt"
D_CKPT = STARGAN_DIR / "200000-D.ckpt"

AttrValue = Union[float, int, bool, str]   # 0/1/小数, 或 "flip" 表示取反

# 常用编辑预设。发色三选一: 设置某个发色为 1 时其它发色会自动置 0 (与官方测试代码一致)。
PRESETS: dict[str, dict[str, AttrValue]] = {
    "black_hair": {"Black_Hair": 1},
    "blond_hair": {"Blond_Hair": 1},
    "brown_hair": {"Brown_Hair": 1},
    "flip_gender": {"Male": "flip"},
    "flip_age": {"Young": "flip"},
    "flip_male": {"Male": "flip"},       # 别名 (Gradio 界面用这个名字)
    "flip_young": {"Young": "flip"},     # 别名
    "to_male": {"Male": 1},
    "to_female": {"Male": 0},
    "older": {"Young": 0},
    "younger": {"Young": 1},
    "blond_flip_gender": {"Blond_Hair": 1, "Male": "flip"},
    "black_hair_older": {"Black_Hair": 1, "Young": 0},
    "flip_gender_age": {"Male": "flip", "Young": "flip"},
}
# 评测和展示用的 5 个单属性编辑 (= 官方 solver.create_labels 生成的 5 个目标)
SINGLE_EDITS: tuple[str, ...] = ("black_hair", "blond_hair", "brown_hair", "flip_gender", "flip_age")


# ======================================================================
# 网络结构 (adapted from yunjey/stargan model.py, MIT License)
# ======================================================================
class ResidualBlock(nn.Module):
    """残差块: 输出 = x + F(x)。G 的"瓶颈"由 6 个残差块组成, 在 32x32 的特征图上改属性。"""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.main(x)


class Generator(nn.Module):
    """编码器 (2 次下采样) -> 6 个残差块 -> 解码器 (2 次上采样), 输出 tanh 到 [-1, 1]。

    用 InstanceNorm (每张图单独归一化) 而不是 BatchNorm: 风格/外观迁移类任务里,
    一张图的颜色统计量不应该受同一 batch 里其它图片的影响。
    """

    def __init__(self, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(3 + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(conv_dim, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)]
        curr = conv_dim
        for _ in range(2):   # 下采样 128 -> 64 -> 32, 通道 64 -> 128 -> 256
            layers += [nn.Conv2d(curr, curr * 2, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(curr * 2, affine=True, track_running_stats=True),
                       nn.ReLU(inplace=True)]
            curr *= 2
        for _ in range(repeat_num):
            layers.append(ResidualBlock(curr, curr))
        for _ in range(2):   # 转置卷积上采样 32 -> 64 -> 128
            layers += [nn.ConvTranspose2d(curr, curr // 2, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.InstanceNorm2d(curr // 2, affine=True, track_running_stats=True),
                       nn.ReLU(inplace=True)]
            curr //= 2
        layers += [nn.Conv2d(curr, 3, kernel_size=7, stride=1, padding=3, bias=False), nn.Tanh()]
        self.main = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: (B,3,H,W) in [-1,1]; c: (B,c_dim) 目标属性。

        条件输入: 把 c 在空间上复制成 (B,c_dim,H,W) 的常数图, 与图像在通道维拼接。
        (官方注释: 这种做法配合 zero padding 才有效; 如果用 reflection padding, 常数通道经过
        卷积后也是常数, 会被 InstanceNorm 的减均值完全抹掉, 网络就"看不到"标签了。)
        """
        c = c.view(c.size(0), c.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
        return self.main(torch.cat([x, c], dim=1))


class Discriminator(nn.Module):
    """PatchGAN 判别器 + 辅助属性分类器, 两个头共享 6 层卷积特征 (128 -> 2x2, 2048 通道)。"""

    def __init__(self, image_size: int = 128, conv_dim: int = 64, c_dim: int = 5, repeat_num: int = 6):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1),
                                   nn.LeakyReLU(0.01)]
        curr = conv_dim
        for _ in range(1, repeat_num):
            layers += [nn.Conv2d(curr, curr * 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.01)]
            curr *= 2
        kernel_size = image_size // 2 ** repeat_num   # 128 / 64 = 2
        self.main = nn.Sequential(*layers)
        # 判别器里不用归一化层: WGAN-GP 的梯度惩罚是对"单个样本"求梯度, BatchNorm 会让样本互相影响
        self.conv1 = nn.Conv2d(curr, 1, kernel_size=3, stride=1, padding=1, bias=False)   # 真/假 (2x2 patch)
        self.conv2 = nn.Conv2d(curr, c_dim, kernel_size=kernel_size, bias=False)          # 属性 logits

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


def load_generator(path: Path | str = G_CKPT, device: str | torch.device = "cpu") -> Generator:
    G = Generator()
    G.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return G.to(device)


def load_discriminator(path: Path | str = D_CKPT, device: str | torch.device = "cpu") -> Discriminator:
    D = Discriminator()
    D.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return D.to(device)


def use_instance_stats(model: nn.Module) -> nn.Module:
    """推理时让 InstanceNorm 始终使用"每张图自己的均值/方差"。

    官方代码训练时 G 一直是 train 模式 (测试脚本也从不调用 G.eval()), 而 train 模式下
    InstanceNorm 用的就是单张图自身的统计量。checkpoint 里的 running_mean/var 只是顺带累计的
    平均值, 训练中从未参与计算。如果直接 G.eval(), 网络会改用这些平均值, 输出明显变差
    (eval_stargan.py 里有定量对比)。这里关掉 running stats, 之后 eval() 也安全。
    """
    for m in model.modules():
        if isinstance(m, nn.InstanceNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


# ======================================================================
# 预处理 / 后处理
# ======================================================================
def celeba_transform(train: bool = False):
    """官方 CelebA 预处理: (训练时随机水平翻转) -> CenterCrop(178) -> Resize(128) -> [-1, 1]。"""
    import torchvision.transforms as T
    t = [T.RandomHorizontalFlip()] if train else []
    t += [T.CenterCrop(CROP_SIZE), T.Resize(IMAGE_SIZE), T.ToTensor(),
          T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    return T.Compose(t)


def resize_pil(img_rgb: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """与 torchvision.Resize 作用在 PIL 图像上完全相同的缩放 (PIL 双线性, 带抗锯齿)。"""
    return np.array(Image.fromarray(img_rgb).resize((size, size), Image.BILINEAR))


def preprocess_aligned(img_rgb: np.ndarray) -> np.ndarray:
    """CelebA 对齐图 (218x178) -> 官方的 128x128 输入 (uint8 RGB)。"""
    assert img_rgb.shape[:2] == (ALIGNED_H, ALIGNED_W), img_rgb.shape
    return resize_pil(np.ascontiguousarray(img_rgb[CROP_TOP:CROP_TOP + CROP_SIZE]))


def to_tensor(img_rgb: np.ndarray | Sequence[np.ndarray]) -> torch.Tensor:
    """uint8 RGB (H,W,3) 或其列表 -> (B,3,H,W) float in [-1, 1]。"""
    arr = np.asarray(img_rgb)
    if arr.ndim == 3:
        arr = arr[None]
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float()
    return x / 127.5 - 1.0


def to_image(x: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) 或 (3,H,W) in [-1, 1] -> uint8 RGB (B,H,W,3) 或 (H,W,3)。"""
    x = ((x.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).numpy() if x.dim() == 4 else x.permute(1, 2, 0).numpy()


# ======================================================================
# 标签向量
# ======================================================================
def attrs_to_vector(attrs: Mapping[str, float] | Sequence[float] | torch.Tensor) -> torch.Tensor:
    """{"Male": 1, ...} 或长度 5 的序列 -> (5,) float 张量, 缺省属性记 0。"""
    if isinstance(attrs, Mapping):
        unknown = set(attrs) - set(ATTRS)
        if unknown:
            raise KeyError(f"未知属性 {sorted(unknown)}, 可选: {ATTRS}")
        return torch.tensor([float(attrs.get(a, 0)) for a in ATTRS])
    v = torch.as_tensor(attrs, dtype=torch.float32)
    assert v.shape[-1] == len(ATTRS), v.shape
    return v


def make_target(c_org: torch.Tensor | Sequence[float], edits: Mapping[str, AttrValue]) -> torch.Tensor:
    """根据原属性 c_org ((5,) 或 (B,5)) 和编辑指令构造目标标签 c_trg。

    edits 的值: 1 / 0 (也可以是 0~1 之间的小数做"插值", 或略大于 1 做夸张), "flip" 表示取反。
    没提到的属性保持原样 —— 这一点很重要: StarGAN 会把标签里的每一维都"执行"一遍,
    原属性填错了, 不想改的属性也会被改掉。
    发色是互斥的: 把某个发色设为正值时, 其余发色自动置 0 (与官方 create_labels 一致)。
    """
    c = torch.as_tensor(c_org, dtype=torch.float32).clone()
    for name, v in edits.items():
        if name not in ATTRS:
            raise KeyError(f"未知属性 {name!r}, 可选: {ATTRS}")
        i = ATTRS.index(name)
        if isinstance(v, str):
            if v != "flip":
                raise ValueError(f"属性值只能是数字或 'flip', 收到 {v!r}")
            c[..., i] = 1.0 - c[..., i]
        else:
            c[..., i] = float(v)
        if name in HAIR_ATTRS and (isinstance(v, str) or float(v) > 0):
            for h in HAIR_ATTRS:
                if h != name:
                    hj = ATTRS.index(h)
                    # flip 时只有"翻成 1"的那些样本才需要清掉其它发色
                    c[..., hj] = torch.where(c[..., i] > 0, torch.zeros_like(c[..., hj]), c[..., hj])
    return c


def resolve_edit(edit: str | Mapping[str, AttrValue]) -> dict[str, AttrValue]:
    if isinstance(edit, str):
        if edit not in PRESETS:
            raise KeyError(f"未知预设 {edit!r}, 可选: {sorted(PRESETS)}")
        return dict(PRESETS[edit])
    return dict(edit)


def official_test_labels(c_org: torch.Tensor) -> list[torch.Tensor]:
    """官方 solver.create_labels (CelebA) 的 5 个目标: 黑/金/棕发, 性别取反, 年龄取反。"""
    return [make_target(c_org, PRESETS[name]) for name in SINGLE_EDITS]


# ======================================================================
# 任意照片 <-> CelebA 对齐空间
# ======================================================================
def crop128_matrix(kps: np.ndarray) -> np.ndarray:
    """返回 2x3 仿射矩阵 A: 原图坐标 -> 128x128 网络输入坐标。

    A = 缩放(128/178) · 平移(去掉上方 20 行) · 相似变换(5 点 -> CELEBA_KPS)。
    缩放那一步加了半像素修正, 使它与 PIL 的 178->128 缩放在像素中心约定上一致。
    """
    M = umeyama(kps, CELEBA_KPS)                       # 3x3, 原图 -> 178x218
    r = IMAGE_SIZE / CROP_SIZE
    S = np.array([[r, 0, 0.5 * r - 0.5],
                  [0, r, -CROP_TOP * r + 0.5 * r - 0.5],
                  [0, 0, 1]])
    return (S @ M)[:2].astype(np.float32)


def warp_to_crop(img_bgr: np.ndarray, kps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把任意照片里的人脸摆成 CelebA 的样子: 先用 5 点相似变换得到 178x178 区域, 再按官方方式缩到 128。

    返回 (face128 BGR uint8, A 原图->128 的 2x3 矩阵)。
    """
    M = umeyama(kps, CELEBA_KPS)[:2].copy()
    M[1, 2] -= CROP_TOP                                # 178x218 -> 中间 178x178
    src = img_bgr
    s = float(np.sqrt(abs(np.linalg.det(M[:, :2]))))
    if s < 0.75:                                       # 人脸很大: 先用 INTER_AREA 缩小原图, 避免双线性缩小时的锯齿
        f = s / 0.75                                   # 缩小后剩余的缩放比约为 0.75
        src = cv2.resize(img_bgr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        M[:, :2] /= f
    crop = cv2.warpAffine(src, M, (CROP_SIZE, CROP_SIZE), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT101)   # 超出照片的部分用镜像填充, 比黑边更"像照片"
    face = resize_pil(crop[..., ::-1].copy())[..., ::-1]   # PIL 是 RGB, 缩放和官方一致
    return np.ascontiguousarray(face), crop128_matrix(kps)


def ellipse_mask(size: int = IMAGE_SIZE, center=(64.0, 62.0), axes=(50.0, 58.0), feather: float = 7.0) -> np.ndarray:
    """128x128 网络坐标系里的羽化椭圆 (头发 + 脸), 值域 [0,1]。

    边缘用高斯模糊做羽化, 贴回原图时就不会出现生硬的接缝。默认参数按 CelebA 对齐后
    人头的位置手调: 眼睛约在 y=65, 嘴约在 y=95, 头发占上方和两侧。
    """
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (int(round(center[0])), int(round(center[1]))), (int(axes[0]), int(axes[1])),
                0, 0, 360, 1.0, -1, cv2.LINE_AA)
    k = int(feather * 3) * 2 + 1
    return cv2.GaussianBlur(m, (k, k), feather)


# ======================================================================
# 编辑器
# ======================================================================
class StarGANEditor:
    """人脸属性编辑: 128x128 CelebA 风格人脸 -> edit(); 任意照片 -> edit_photo()。

    原属性 (c_org) 怎么来? 任意照片没有标注, 这里用判别器 D 的辅助分类头来估计 ——
    D 在训练中一直被要求"认出真实照片的属性", 正好可以复用。
    """

    def __init__(self, g_path: Path | str = G_CKPT, d_path: Path | str = D_CKPT,
                 device: str | torch.device = "cpu", detector=None):
        self.device = torch.device(device)
        self.G = use_instance_stats(load_generator(g_path)).to(self.device).eval()
        self._d_path = d_path
        self._D: Discriminator | None = None
        self._detector = detector

    @property
    def D(self) -> Discriminator:
        if self._D is None:                       # 179MB, 用到时才加载
            self._D = load_discriminator(self._d_path, self.device).eval()
        return self._D

    @property
    def detector(self):
        if self._detector is None:
            from ..detector import SCRFD
            self._detector = SCRFD()
        return self._detector

    # ---------- 张量接口 (批量, 评测用) ----------
    @torch.no_grad()
    def generate(self, x: torch.Tensor, c_trg: torch.Tensor) -> torch.Tensor:
        """x: (B,3,128,128) in [-1,1]; c_trg: (B,5) -> (B,3,128,128) in [-1,1] (在 CPU 上返回)。"""
        return self.G(x.to(self.device), c_trg.to(self.device).float()).cpu()

    @torch.no_grad()
    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """D 的辅助分类器: 返回每个属性的概率 (B,5)。"""
        return torch.sigmoid(self.D(x.to(self.device))[1]).cpu()

    def estimate_attrs(self, x: torch.Tensor) -> torch.Tensor:
        return (self.classify(x) > 0.5).float()

    # ---------- numpy 接口 ----------
    def _src_vector(self, x: torch.Tensor, src_attrs) -> torch.Tensor:
        if src_attrs is None:
            return self.estimate_attrs(x)[0]
        return attrs_to_vector(src_attrs)

    def edit_many(self, face128: np.ndarray, edits: Sequence[str | Mapping[str, AttrValue]],
                  src_attrs=None, bgr: bool = True) -> list[np.ndarray]:
        """一张 128x128 人脸, 一次做多个编辑 (拼成一个 batch 跑 G)。"""
        if face128.shape[:2] == (ALIGNED_H, ALIGNED_W):          # 直接给了 CelebA 对齐原图
            face128 = preprocess_aligned(face128[..., ::-1] if bgr else face128)
            face128 = face128[..., ::-1] if bgr else face128
        assert face128.shape[:2] == (IMAGE_SIZE, IMAGE_SIZE), face128.shape
        rgb = face128[..., ::-1] if bgr else face128
        x = to_tensor(rgb)
        c_org = self._src_vector(x, src_attrs)
        c_trg = torch.stack([make_target(c_org, resolve_edit(e)) for e in edits])
        out = to_image(self.generate(x.repeat(len(edits), 1, 1, 1), c_trg))
        return [np.ascontiguousarray(o[..., ::-1] if bgr else o) for o in out]

    def edit(self, face128: np.ndarray, target_attrs: str | Mapping[str, AttrValue],
             src_attrs=None, bgr: bool = True) -> np.ndarray:
        """编辑一张 128x128 人脸 (BGR 或 RGB, 由 bgr 指定)。

        target_attrs: 预设名 (如 "blond_hair") 或 {"Blond_Hair": 1, "Male": "flip"};
        src_attrs: 原属性 (dict 或 5 维向量), None 时用 D 的分类器估计。
        """
        return self.edit_many(face128, [target_attrs], src_attrs, bgr)[0]

    # ---------- 任意照片 ----------
    def edit_photo_many(self, img_bgr: np.ndarray, edits: Sequence[str | Mapping[str, AttrValue]],
                        blend: str = "residual", kps: np.ndarray | None = None,
                        src_attrs=None, return_info: bool = False):
        """任意照片: 检测 -> 5 点对齐到 CelebA 空间 -> G -> 逆变换贴回原图 (羽化椭圆蒙版)。

        blend:
          "replace":  蒙版内直接换成 G 的输出 (放大后的 128x128, 人脸大时会偏糊)
          "residual": 只把"G 改了什么" (输出 - 输入) 放大后加回原图, 原图的高频细节得以保留
        """
        if kps is None:
            dets, kpss = self.detector.detect(img_bgr, max_num=1)
            if len(dets) == 0:
                raise ValueError("照片里没有检测到人脸")
            kps = kpss[0]
        face, A = warp_to_crop(img_bgr, kps)
        outs = self.edit_many(face, edits, src_attrs=src_attrs, bgr=True)

        h, w = img_bgr.shape[:2]
        inv = dict(dsize=(w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        mask = cv2.warpAffine(ellipse_mask(), A, **inv)[..., None]
        base = img_bgr.astype(np.float32)
        results = []
        for o in outs:
            if blend == "residual":
                delta = o.astype(np.float32) - face.astype(np.float32)
                res = base + mask * cv2.warpAffine(delta, A, **inv)
            elif blend == "replace":
                res = base * (1 - mask) + mask * cv2.warpAffine(o.astype(np.float32), A, **inv)
            else:
                raise ValueError(f"blend 只能是 'residual' 或 'replace', 收到 {blend!r}")
            results.append(np.clip(res + 0.5, 0, 255).astype(np.uint8))
        if return_info:
            x = to_tensor(face[..., ::-1].copy())
            info = {"kps": kps, "A": A, "face128": face, "edited128": outs,
                    "src_attrs": dict(zip(ATTRS, self._src_vector(x, src_attrs).tolist()))}
            return results, info
        return results

    def edit_photo(self, img_bgr: np.ndarray, edits: str | Mapping[str, AttrValue], **kw) -> np.ndarray:
        """edit_photo_many 的单编辑版本, 返回编辑后的整张照片 (BGR)。"""
        return self.edit_photo_many(img_bgr, [edits], **kw)[0]
