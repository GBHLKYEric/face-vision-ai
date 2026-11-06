"""用老师模型 (ArcFace-R50, WebFace600K 训练, LFW 99.8%) 给 CelebA 每张脸提取特征, 供知识蒸馏使用。

ONNX 模型先用 onnx2torch 转成 PyTorch 模块, 这样能跑在 Intel 核显 (xpu) 上, 比 CPU 快几倍。
转换后先和 ONNX Runtime 的输出逐元素比对, 确认转换无误再批量提取。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx2torch import convert
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision.paths import DATA, REC_MODEL  # noqa: E402
from facevision.recognizer import ArcFaceONNX  # noqa: E402
from facevision.training.data import to_tensor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="xpu" if torch.xpu.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    from facevision.utils import disable_power_throttling
    disable_power_throttling()

    images = np.load(DATA / "celeba" / "aligned112.npy", mmap_mode="r")
    out_path = DATA / "celeba" / "teacher_r50.npy"
    model = convert(onnx.load(str(REC_MODEL))).eval().to(args.device)

    # 1) 校验: PyTorch 版和 ONNX Runtime 版输出应几乎一致
    sample = [images[i] for i in range(8)]
    ref = ArcFaceONNX().embed(sample, flip_test=False)
    with torch.no_grad():
        mine = torch.nn.functional.normalize(model(to_tensor(np.stack(sample), args.device)), dim=1).cpu().numpy()
    cos = (ref * mine).sum(1)
    print("onnx2torch 校验, 与 ORT 的余弦相似度:", cos.round(6))
    assert cos.min() > 0.9999

    # 2) 批量提取 (原图 + 水平翻转 两次相加, 与评测时一致)
    feats = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float16, shape=(len(images), 512))
    t = time.time()
    with torch.no_grad(), torch.autocast(device_type=args.device, dtype=torch.bfloat16):
        for i in tqdm(range(0, len(images), args.batch)):
            x = to_tensor(images[i:i + args.batch], args.device)
            f = model(x).float() + model(x.flip(-1)).float()
            feats[i:i + args.batch] = torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float16)
    feats.flush()
    print(f"完成 {len(images)} 张, 用时 {time.time() - t:.0f}s -> {out_path}")

    # 3) bf16 推理的特征和 fp32 ORT 的差异 (确认半精度不影响老师质量)
    ref = ArcFaceONNX().embed([images[i] for i in range(64)], flip_test=True)
    print("bf16 老师特征 vs fp32 ORT 余弦相似度: min=%.5f" % (ref * feats[:64].astype(np.float32)).sum(1).min())


if __name__ == "__main__":
    main()
