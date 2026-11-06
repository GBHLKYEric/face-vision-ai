"""StarGAN 模块的快速测试 (CPU, 几秒内跑完): python -m pytest tests/test_stargan.py -q"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from facevision.gan import stargan as sg

pytestmark = pytest.mark.skipif(not sg.G_CKPT.exists(), reason="缺少 weights/stargan/200000-G.ckpt")


@pytest.fixture(scope="module")
def G() -> sg.Generator:
    torch.manual_seed(0)
    return sg.use_instance_stats(sg.load_generator()).eval()


def test_checkpoints_load_strict():
    # strict=True: 官方权重里的每个参数名/形状都必须和我们的网络一一对应
    G = sg.Generator()
    missing, unexpected = G.load_state_dict(torch.load(sg.G_CKPT, map_location="cpu", weights_only=True), strict=True)
    assert not missing and not unexpected
    D = sg.Discriminator()
    D.load_state_dict(torch.load(sg.D_CKPT, map_location="cpu", weights_only=True), strict=True)
    assert sum(p.numel() for p in G.parameters()) == 8_430_528    # 约 843 万参数
    assert sum(p.numel() for p in D.parameters()) == 44_762_048   # 约 4476 万参数 (大头是最后 2048 通道的卷积)


def test_generator_output_shape_and_range(G):
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    c = torch.tensor([[1, 0, 0, 1, 1], [0, 1, 0, 0, 0]], dtype=torch.float32)
    with torch.no_grad():
        y = G(x, c)
        y_other = G(x, c.flip(0))
    assert y.shape == (2, 3, 128, 128)
    assert y.min() >= -1 and y.max() <= 1          # tanh 输出
    assert not torch.allclose(y[0], y_other[0])    # 标签不同 -> 输出不同


def test_discriminator_heads():
    D = sg.Discriminator()
    out_src, out_cls = D(torch.zeros(3, 3, 128, 128))
    assert out_src.shape == (3, 1, 2, 2)     # PatchGAN: 2x2 个真假分数
    assert out_cls.shape == (3, 5)           # 5 个属性 logits


def test_instance_stats_eval_equals_train_mode(G):
    # use_instance_stats 之后 eval() 与官方测试时的 train 模式输出完全一致 (全卷积网络, 用 64x64 省时间)
    G_train = sg.load_generator().train()
    x, c = torch.rand(1, 3, 64, 64) * 2 - 1, torch.tensor([[0, 1, 0, 1, 0.]])
    with torch.no_grad():
        assert torch.allclose(G_train(x, c), G(x, c), atol=1e-5)


def test_make_target_rules():
    c = torch.tensor([1, 0, 0, 1, 1.])      # 黑发、男、年轻
    assert sg.make_target(c, {"Blond_Hair": 1}).tolist() == [0, 1, 0, 1, 1]   # 发色互斥
    assert sg.make_target(c, {"Male": "flip"}).tolist() == [1, 0, 0, 0, 1]
    assert sg.make_target(c, {"Young": 0, "Brown_Hair": 1}).tolist() == [0, 0, 1, 1, 0]
    assert sg.make_target(c, {"Male": 0.5}).tolist() == [1, 0, 0, 0.5, 1]     # 允许插值
    assert c.tolist() == [1, 0, 0, 1, 1]                                       # 不修改输入
    # 批量 + 对发色 flip: 只有翻成 1 的样本才清掉其它发色
    cb = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0.]])
    assert sg.make_target(cb, {"Blond_Hair": "flip"}).tolist() == [[0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]
    with pytest.raises(KeyError):
        sg.make_target(c, {"Bald": 1})


def test_official_test_labels_match_solver_create_labels():
    c = torch.tensor([[0, 0, 1, 0, 1.]])    # 棕发、女、年轻
    got = [t[0].tolist() for t in sg.official_test_labels(c)]
    assert got == [[1, 0, 0, 0, 1], [0, 1, 0, 0, 1], [0, 0, 1, 0, 1], [0, 0, 1, 1, 1], [0, 0, 1, 0, 0]]


def test_attrs_to_vector_and_presets():
    assert sg.attrs_to_vector({"Male": 1, "Young": 1}).tolist() == [0, 0, 0, 1, 1]
    for name, edit in sg.PRESETS.items():
        sg.make_target(torch.zeros(5), edit)   # 每个预设都能构造标签
    assert sg.resolve_edit("blond_hair") == {"Blond_Hair": 1}
    assert sg.resolve_edit("flip_male") == {"Male": "flip"} and sg.resolve_edit("flip_young") == {"Young": "flip"}
    assert sg.resolve_edit({"Blond_Hair": 1}) == {"Blond_Hair": 1}


def test_preprocess_matches_official_transform():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (218, 178, 3), dtype=np.uint8)
    ours = sg.to_tensor(sg.preprocess_aligned(img))[0]
    official = sg.celeba_transform(train=False)(Image.fromarray(img))
    assert torch.allclose(ours, official, atol=1e-6)


def test_crop_matrix_maps_canonical_points():
    # 已经是 CelebA 对齐图时, 原图 -> 128 的变换就是"去掉上方 20 行再缩放 128/178"
    A = sg.crop128_matrix(sg.CELEBA_KPS)
    pts = sg.CELEBA_KPS @ A[:, :2].T + A[:, 2]
    r = 128 / 178
    expected = (sg.CELEBA_KPS - [0, 20] + 0.5) * r - 0.5
    assert np.abs(pts - expected).max() < 1e-3


def test_edit_photo_keeps_background(G):
    ed = sg.StarGANEditor.__new__(sg.StarGANEditor)   # 复用已加载的 G, 不加载 D/检测器
    ed.device, ed.G, ed._D, ed._detector = torch.device("cpu"), G, None, None
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (400, 300, 3), dtype=np.uint8)
    kps = sg.CELEBA_KPS * 1.2 + [40, 60]        # 假装检测到的 5 点
    out, info = ed.edit_photo_many(img, ["blond_hair"], kps=kps, src_attrs=[1, 0, 0, 1, 1], return_info=True)
    out = out[0]
    assert out.shape == img.shape and out.dtype == np.uint8
    assert info["face128"].shape == (128, 128, 3)
    assert np.array_equal(out[:10], img[:10]) and np.array_equal(out[-10:], img[-10:])   # 蒙版外不变
    assert np.abs(out.astype(int) - img).mean() > 0                                     # 蒙版内有变化
