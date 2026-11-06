"""人脸识别损失函数: 从 Softmax 到 ArcFace 的演进, 以及困难样本挖掘。

统一视角: 把最后一层分类器的权重 W_j 看成"第 j 个人的中心方向"。
特征 x 和 W_j 都归一化成单位向量后, logit_j = cos(θ_j), θ_j 是两者夹角。
  Softmax   : logit = W·x + b                      只要求"分对", 类内可以很松散
  SphereFace: 目标类 logit = cos(m1·θ)             角度乘 m1 (乘性角度间隔)
  ArcFace   : 目标类 logit = cos(θ + m2)           角度加 m2 (加性角度间隔, 几何意义最清晰)
  CosFace   : 目标类 logit = cos(θ) - m3           余弦减 m3 (加性余弦间隔)
最后都乘一个尺度 s (如 64), 再送进普通的交叉熵。
间隔 m 的作用: 训练时故意给目标类"使绊子", 逼网络把同一个人的特征压得更紧、
不同人推得更远, 测试时不加间隔, 就有了安全余量。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginSoftmax(nn.Module):
    """统一的间隔 softmax: cos(m1·θ + m2) - m3。

    m1=1, m2=0,   m3=0    -> 归一化 Softmax (NormFace)
    m1=1, m2=0.5, m3=0    -> ArcFace
    m1=1, m2=0,   m3=0.35 -> CosFace
    m1=1.35 ...           -> SphereFace 风格 (这里用 cos(m1·θ) 的单调化近似)
    """

    def __init__(self, embedding_size: int, num_classes: int, s: float = 64.0,
                 m1: float = 1.0, m2: float = 0.5, m3: float = 0.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_size))
        nn.init.normal_(self.weight, std=0.01)
        self.s, self.m1, self.m2, self.m3 = s, m1, m2, m3

    def cosine(self, emb: torch.Tensor) -> torch.Tensor:
        return F.linear(F.normalize(emb), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)

    def margin_logits(self, cos: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        target = cos.gather(1, labels[:, None])                 # 目标类的 cosθ
        theta = torch.acos(target)
        if self.m1 != 1.0:
            # SphereFace: ψ(θ) = (-1)^k·cos(m1·θ) - 2k, k = ⌊m1·θ/π⌋, 保证 ψ 在 [0, π] 上单调递减
            k = torch.floor(self.m1 * theta / math.pi)
            sign = 1.0 - 2.0 * torch.remainder(k, 2)
            new_target = sign * torch.cos(self.m1 * theta) - 2 * k
        else:
            new_target = torch.cos(theta + self.m2) - self.m3
            if self.m2 > 0:
                # θ + m 超过 π 后 cos 不再单调递减, 会"奖励"错得更离谱的样本。
                # 这时改用线性惩罚 cosθ - m·sin(m), 保证 logit 随 θ 单调下降
                new_target = torch.where(target > math.cos(math.pi - self.m2), new_target,
                                         target - self.m2 * math.sin(self.m2) - self.m3)
        logits = cos.scatter(1, labels[:, None], new_target)
        return logits * self.s

    def forward(self, emb, labels):
        cos = self.cosine(emb)
        logits = self.margin_logits(cos, labels)
        return F.cross_entropy(logits, labels), cos


class CurricularFace(MarginSoftmax):
    """CurricularFace (CVPR 2020): 把"困难样本挖掘"写进损失函数里。

    困难负类: 某个非目标类 j 的 cosθ_j 比目标类加了间隔后的值还大 (网络把它认成别人了)。
    对这些困难负类的 logit 做放大: cosθ_j * (t + cosθ_j)。
    t 是全局"课程进度", 用目标类 cosθ 的滑动平均估计:
      训练初期 t 小 -> 困难样本权重低, 先学简单的;
      后期 t 变大 -> 困难样本权重越来越高, 专攻难例 (像人学习一样由易到难)。
    """

    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5, momentum=0.01):
        super().__init__(embedding_size, num_classes, s=s, m2=m)
        self.momentum = momentum
        self.register_buffer("t", torch.zeros(1))

    def forward(self, emb, labels):
        cos = self.cosine(emb)
        target = cos.gather(1, labels[:, None])
        theta = torch.acos(target)
        cos_tm = torch.cos(theta + self.m2)
        cos_tm = torch.where(target > math.cos(math.pi - self.m2), cos_tm,
                             target - self.m2 * math.sin(self.m2))
        with torch.no_grad():
            self.t.mul_(1 - self.momentum).add_(self.momentum * target.mean())
        hard = cos > cos_tm
        cos = torch.where(hard, cos * (self.t + cos), cos)
        logits = cos.scatter(1, labels[:, None], cos_tm) * self.s
        return F.cross_entropy(logits, labels), cos


def batch_hard_triplet_loss(emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """三元组损失 + 批内困难样本挖掘 (FaceNet / "In Defense of the Triplet Loss")。

    对每个锚点 a: 在批内找 最远的同类样本 p (最难的正样本) 和 最近的异类样本 n (最难的负样本),
    要求 d(a,p) + margin < d(a,n)。只用最难的样本, 收敛比随机三元组快很多。
    需要配合 PK 采样器 (每批 P 个人 x 每人 K 张) 使用, 保证批内一定有同类。
    """
    emb = F.normalize(emb)
    dist = torch.cdist(emb, emb)
    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=emb.device)
    hardest_pos = (dist * (same & ~eye)).max(1).values
    hardest_neg = dist.masked_fill(same, float("inf")).min(1).values
    return F.relu(hardest_pos - hardest_neg + margin).mean()


def contrastive_loss(emb1, emb2, same: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """对比损失 (DeepID2): 同一人距离越小越好; 不同人距离至少拉开 margin。"""
    d = F.pairwise_distance(F.normalize(emb1), F.normalize(emb2))
    return (same * d.pow(2) + (~same) * F.relu(margin - d).pow(2)).mean()


def distillation_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """特征蒸馏: 让学生的特征方向对齐老师的特征方向 (1 - 余弦相似度)。

    老师 (大模型) 已经学会了"什么样的脸是同一个人", 学生直接模仿老师的输出,
    比只靠身份标签学得快得多, 也能学到标签里没有的"人和人之间有多像"的信息。
    """
    return (1 - F.cosine_similarity(student_emb, teacher_emb, dim=1)).mean()


def build_head(name: str, embedding_size: int, num_classes: int) -> MarginSoftmax:
    name = name.lower()
    if name == "softmax":
        return MarginSoftmax(embedding_size, num_classes, s=30.0, m2=0.0)
    if name == "sphereface":
        return MarginSoftmax(embedding_size, num_classes, s=64.0, m1=1.35, m2=0.0)
    if name == "cosface":
        return MarginSoftmax(embedding_size, num_classes, s=64.0, m2=0.0, m3=0.35)
    if name == "arcface":
        return MarginSoftmax(embedding_size, num_classes, s=64.0, m2=0.5)
    if name == "curricularface":
        return CurricularFace(embedding_size, num_classes, s=64.0, m=0.5)
    raise ValueError(name)
