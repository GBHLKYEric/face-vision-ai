"""模型优化与部署对比: FP32 / BN融合 / 动态量化 / 剪枝 / ONNX / ONNX-INT8 的 大小、速度、LFW 精度。

用法:
    python tools/deploy_benchmark.py --ckpt runs/r18_arcface_kd/final.pt
结果: outputs/deploy/report.json, report.md, 以及导出的 .onnx 文件
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision.datasets import read_lfw_pairs  # noqa: E402
from facevision.deploy import (bench, export_onnx, file_size_mb, fuse_model, magnitude_prune,  # noqa: E402
                               quantize_dynamic_torch, quantize_static_onnx, torch_size_mb)
from facevision.paths import DATA, OUTPUTS, REC_MODEL  # noqa: E402
from facevision.recognizer import ArcFaceONNX  # noqa: E402
from facevision.training.data import to_tensor  # noqa: E402
from facevision.training.embedder import load_backbone  # noqa: E402
from facevision.verification import evaluate_verification  # noqa: E402

OUT = OUTPUTS / "deploy"


class LFW:
    def __init__(self):
        z = np.load(OUTPUTS / "lfw" / "aligned.npz", allow_pickle=False)
        self.aligned = z["aligned"]
        index = {Path(p): i for i, p in enumerate(z["paths"])}
        pairs = read_lfw_pairs(DATA / "lfw")
        self.i1 = np.array([index[a] for a, _, _ in pairs])
        self.i2 = np.array([index[b] for _, b, _ in pairs])
        self.issame = np.array([s for _, _, s in pairs])

    def accuracy(self, embed_fn, batch=128) -> float:
        feats = np.concatenate([embed_fn(self.aligned[i:i + batch]) for i in range(0, len(self.aligned), batch)])
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return evaluate_verification((feats[self.i1] * feats[self.i2]).sum(1), self.issame).accuracy_mean * 100


def write_report(rows, threads: int):
    (OUT / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"| 方案 | 大小(MB) | 单张延迟(ms, CPU {threads}线程) | LFW 准确率(%) | 说明 |", "|---|---|---|---|---|"]
    md += [f"| {r['variant']} | {r['size_mb']} | {r['latency_ms']} | {r['lfw_acc']} | {r['note']} |" for r in rows]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")


def torch_embed(model):
    @torch.no_grad()
    def fn(faces):
        x = to_tensor(faces, "cpu").contiguous()
        return (model(x) + model(x.flip(-1))).numpy()
    return fn


def onnx_embed(path):
    m = ArcFaceONNX(path)
    return lambda faces: m.embed(list(faces), flip_test=True, normalize=False), m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()
    from facevision.utils import disable_power_throttling
    disable_power_throttling()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    import os
    os.environ["FV_ORT_THREADS"] = str(args.threads)

    lfw = LFW()
    calib = np.load(DATA / "celeba" / "aligned112.npy", mmap_mode="r")
    calib = calib[np.random.default_rng(0).choice(len(calib), 512, replace=False)]
    one = torch.from_numpy(np.zeros((1, 3, 112, 112), np.float32))
    rows = []

    def add(name, size, ms, acc, note=""):
        rows.append({"variant": name, "size_mb": round(size, 2), "latency_ms": round(ms, 2),
                     "lfw_acc": round(acc, 3), "note": note})
        print(rows[-1], flush=True)
        write_report(rows, args.threads)   # 每测完一项就落盘, 中途失败也不丢前面的结果

    model, ck = load_backbone(args.ckpt, "cpu")
    tag = ck["args"]["backbone"]
    # PyTorch FP32
    add(f"{tag} PyTorch FP32", torch_size_mb(model), bench(lambda: model(one)), lfw.accuracy(torch_embed(model)))
    # BN 融合: 输出应完全一致, 速度更快
    fused = fuse_model(model)
    with torch.no_grad():
        x = to_tensor(lfw.aligned[:16], "cpu").contiguous()
        diff = (model(x) - fused(x)).abs().max().item()
    add(f"{tag} PyTorch BN融合", torch_size_mb(fused), bench(lambda: fused(one)), lfw.accuracy(torch_embed(fused)),
        f"与原模型最大输出差 {diff:.2e}")
    # PyTorch 动态量化 (只量化 Linear)
    dq = quantize_dynamic_torch(model)
    add(f"{tag} PyTorch 动态INT8", torch_size_mb(dq), bench(lambda: dq(one)), lfw.accuracy(torch_embed(dq)),
        "只量化全连接层")
    # 剪枝 (不微调)
    for amount in (0.3, 0.5, 0.7):
        pm = magnitude_prune(model, amount)
        add(f"{tag} 剪枝{int(amount * 100)}%", torch_size_mb(pm), bench(lambda pm=pm: pm(one)),
            lfw.accuracy(torch_embed(pm)), "非结构化剪枝, 未微调; 稠密存储所以大小/速度不变")
    # ONNX FP32 / INT8
    onnx_fp32 = export_onnx(fused, OUT / f"{tag}_fp32.onnx")
    fn, m = onnx_embed(onnx_fp32)
    blob = np.zeros((1, 3, 112, 112), np.float32)
    add(f"{tag} ONNX FP32", file_size_mb(onnx_fp32), bench(lambda: m.session.run(None, {m.input_name: blob})),
        lfw.accuracy(fn))
    onnx_int8 = quantize_static_onnx(onnx_fp32, OUT / f"{tag}_int8.onnx", calib)
    fn, m = onnx_embed(onnx_int8)
    add(f"{tag} ONNX INT8(静态)", file_size_mb(onnx_int8), bench(lambda: m.session.run(None, {m.input_name: blob})),
        lfw.accuracy(fn), "逐通道权重 + 512 张 CelebA 校准")

    if not args.skip_teacher:
        fn, m = onnx_embed(REC_MODEL)
        add("R50 老师 ONNX FP32", file_size_mb(REC_MODEL), bench(lambda: m.session.run(None, {m.input_name: blob})),
            lfw.accuracy(fn))
        r50_int8 = quantize_static_onnx(REC_MODEL, OUT / "r50_teacher_int8.onnx", calib)
        fn, m = onnx_embed(r50_int8)
        add("R50 老师 ONNX INT8(静态)", file_size_mb(r50_int8),
            bench(lambda: m.session.run(None, {m.input_name: blob})), lfw.accuracy(fn))

    print((OUT / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
