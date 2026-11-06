"""核心算法的单元测试: 对齐、NMS、验证指标、损失函数、骨干网络、BN 融合。

运行: python -m pytest tests -q
"""
import math

import numpy as np
import pytest
import torch

from facevision.align import ARCFACE_TEMPLATE, estimate_norm, transform_points, umeyama
from facevision.deploy import fuse_model
from facevision.detector import distance2bbox, nms
from facevision.training.backbones import get_backbone
from facevision.training.losses import (MarginSoftmax, batch_hard_triplet_loss, build_head,
                                        distillation_loss)
from facevision.verification import best_threshold, evaluate_verification


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(5, 2)) * 30 + 100
    ang, s, t = 0.4, 1.7, np.array([12.0, -7.0])
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    dst = s * src @ R.T + t
    T = umeyama(src, dst)
    assert np.allclose(T[:2, :2], s * R, atol=1e-8)
    assert np.allclose(T[:2, 2], t, atol=1e-6)


def test_umeyama_two_points_is_rotation_not_reflection():
    T = umeyama(np.array([[100, 120], [140, 100]], float), ARCFACE_TEMPLATE[:2])
    assert np.linalg.det(T[:2, :2]) > 0
    assert np.allclose(transform_points(np.array([[100, 120], [140, 100]], np.float32), T[:2]),
                       ARCFACE_TEMPLATE[:2], atol=1e-3)


def test_estimate_norm_maps_template_to_itself():
    M = estimate_norm(ARCFACE_TEMPLATE * 2 + 10)
    assert np.allclose(transform_points(ARCFACE_TEMPLATE * 2 + 10, M), ARCFACE_TEMPLATE, atol=1e-3)


def test_nms_and_distance2bbox():
    boxes = distance2bbox(np.array([[50.0, 50.0], [52.0, 50.0], [200.0, 200.0]]),
                          np.array([[10, 10, 10, 10], [10, 10, 10, 10], [5, 5, 5, 5]], dtype=float))
    dets = np.hstack([boxes, np.array([[0.9], [0.8], [0.7]])])
    assert nms(dets, 0.4) == [0, 2]


def test_best_threshold_and_10fold():
    rng = np.random.default_rng(0)
    issame = np.tile(np.r_[np.ones(300, bool), np.zeros(300, bool)], 10)
    scores = np.where(issame, rng.normal(0.7, 0.1, 6000), rng.normal(0.0, 0.1, 6000))
    thr, acc = best_threshold(scores, issame)
    assert 0.2 < thr < 0.5 and acc > 0.99
    res = evaluate_verification(scores, issame)
    assert res.accuracy_mean > 0.99 and 0.99 < res.auc <= 1.0


def test_arcface_margin_lowers_target_logit():
    torch.manual_seed(0)
    head = build_head("arcface", 8, 4)
    emb, y = torch.randn(6, 8), torch.randint(0, 4, (6,))
    cos = head.cosine(emb)
    logits = head.margin_logits(cos, y) / head.s
    target_before = cos.gather(1, y[:, None])
    target_after = logits.gather(1, y[:, None])
    assert torch.all(target_after < target_before)            # 目标类被"惩罚"
    other = torch.ones_like(cos, dtype=torch.bool).scatter(1, y[:, None], False)
    assert torch.allclose(logits[other], cos[other])           # 非目标类不变


def test_sphereface_psi_monotonic():
    head = MarginSoftmax(4, 2, m1=1.35, m2=0.0)
    theta = torch.linspace(0, math.pi - 1e-3, 500)
    cos = torch.stack([torch.cos(theta), torch.zeros_like(theta)], 1)
    psi = head.margin_logits(cos, torch.zeros(500, dtype=torch.long))[:, 0]
    assert torch.all(psi[1:] <= psi[:-1] + 1e-5)


def test_triplet_and_kd_losses():
    emb = torch.tensor([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.1, 0.9]])
    labels = torch.tensor([0, 0, 1, 1])
    assert batch_hard_triplet_loss(emb, labels, margin=0.3) == 0
    assert distillation_loss(emb, emb).abs() < 1e-6


@pytest.mark.parametrize("name", ["r18", "mbf"])
def test_backbone_shapes_and_fusion(name):
    torch.manual_seed(0)
    model = get_backbone(name).eval()
    # 随便给 BN 一些统计量, 否则融合前后都是恒等, 测不出问题
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2)
            m.weight.data.uniform_(0.5, 1.5)
    x = torch.randn(2, 3, 112, 112)
    with torch.no_grad():
        y = model(x)
        y2 = fuse_model(model)(x)
    assert y.shape == (2, 512)
    # 随机初始化的网络输出量级很大, 用相对误差比较
    assert ((y - y2).abs().max() / y.abs().max()) < 1e-4
