"""人脸识别骨干网络: IResNet (ArcFace 论文用的改进版 ResNet) 和 MobileFaceNet (移动端)。

IResNet 相对普通 ResNet 的改动 (都是为人脸 112x112 小图和特征质量服务的):
  1. 第一层不用 7x7 卷积 + 最大池化, 而是 3x3 卷积、不下采样 -> 保留更多细节
  2. 残差块是 BN-Conv-BN-PReLU-Conv-BN 的"预激活"结构, 训练更稳
  3. 结尾不用全局平均池化, 而是 BN -> Dropout -> 全连接 -> BN 得到 512 维特征
     (人脸各位置含义不同, 平均池化会丢掉"眼睛在哪"这种空间信息)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride, 1, bias=False)


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.bn1(x)
        out = self.prelu(self.bn2(self.conv1(out)))
        out = self.bn3(self.conv2(out))
        return out + identity          # 残差连接: 学"变化量", 梯度可以直接流回浅层


class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, layers, dropout=0.0, num_features=512, width=64):
        super().__init__()
        self.inplanes = width
        self.conv1 = conv3x3(3, width)
        self.bn1 = nn.BatchNorm2d(width, eps=1e-05)
        self.prelu = nn.PReLU(width)
        # 4 个 stage, 每个 stage 开头 stride=2 下采样: 112 -> 56 -> 28 -> 14 -> 7
        self.layer1 = self._make_layer(width, layers[0], stride=2)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(width * 8, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(width * 8 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # 每个残差块最后一个 BN 初始化为 0: 训练初期每块近似恒等映射, 深网络也好训
        for m in self.modules():
            if isinstance(m, IBasicBlock):
                nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, planes, blocks, stride):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes, eps=1e-05))
        layers = [IBasicBlock(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return self.features(x)


# ---------------- MobileFaceNet ----------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=0, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.PReLU(cout) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthWise(nn.Module):
    """倒残差块 (MobileNetV2): 1x1 升维 -> 3x3 逐通道卷积 -> 1x1 降维。

    逐通道卷积 (depthwise) 每个通道单独卷积, 计算量只有普通卷积的 1/通道数,
    是移动端网络省算力的关键。
    """

    def __init__(self, cin, cout, stride, groups, residual):
        super().__init__()
        self.residual = residual
        self.block = nn.Sequential(
            ConvBlock(cin, groups, 1),
            ConvBlock(groups, groups, 3, stride, 1, groups=groups),
            ConvBlock(groups, cout, 1, act=False))

    def forward(self, x):
        out = self.block(x)
        return x + out if self.residual else out


class MobileFaceNet(nn.Module):
    """MobileFaceNet (Chen et al. 2018): 约 1.2M 参数, 0.22 G 次乘加, 专为手机端人脸识别设计。

    关键设计: 结尾用 7x7 的 "全局逐通道卷积" (GDConv) 代替全局平均池化,
    让网络自己学每个空间位置的权重 (人脸中心比边角更重要)。
    """

    def __init__(self, num_features=512, width=2):
        super().__init__()
        w = 64 * width // 2
        self.stem = nn.Sequential(
            ConvBlock(3, w, 3, 2, 1),                 # 112 -> 56
            ConvBlock(w, w, 3, 1, 1, groups=w))
        cfg = [  # (输出通道, 重复次数, 第一个的步长, 扩展通道)
            (w, 1, 2, 128 * width // 2),              # 56 -> 28
            (w, 4, 1, 128 * width // 2),
            (128 * width // 2, 1, 2, 256 * width // 2),   # 28 -> 14
            (128 * width // 2, 6, 1, 256 * width // 2),
            (128 * width // 2, 1, 2, 512 * width // 2),   # 14 -> 7
            (128 * width // 2, 2, 1, 256 * width // 2),
        ]
        layers, cin = [], w
        for cout, n, s, g in cfg:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(DepthWise(cin, cout, stride, g, residual=(stride == 1 and cin == cout)))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.conv_sep = ConvBlock(cin, 512, 1)
        self.gdc = ConvBlock(512, 512, 7, 1, 0, groups=512, act=False)  # 全局逐通道卷积 7x7 -> 1x1
        self.linear = nn.Conv2d(512, num_features, 1, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        x = self.conv_sep(self.body(self.stem(x)))
        x = self.bn(self.linear(self.gdc(x)))
        return torch.flatten(x, 1)


def get_backbone(name: str, num_features: int = 512, dropout: float = 0.0) -> nn.Module:
    name = name.lower()
    depth = {"r18": [2, 2, 2, 2], "r34": [3, 4, 6, 3], "r50": [3, 4, 14, 3], "r100": [3, 13, 30, 3]}
    if name in depth:
        return IResNet(depth[name], dropout=dropout, num_features=num_features)
    if name == "mbf":
        return MobileFaceNet(num_features)
    if name == "mbf_large":
        return MobileFaceNet(num_features, width=4)
    raise ValueError(f"unknown backbone {name}")
