"""训练人脸识别模型: IResNet / MobileFaceNet + ArcFace 损失 (+ 知识蒸馏)。

数据: CelebA 20 万张对齐人脸, 10177 个身份 (与 LFW 人物不重叠)。
损失: L = L_margin(ArcFace/CosFace/...) + λ · (1 - cos(学生特征, 老师特征))
每个 epoch 结束在 LFW 6000 对上按官方 10 折协议测一次准确率, 画训练曲线。

例:
    python tools/train_recognition.py --backbone r18 --head arcface --kd-weight 50 --epochs 10 --lr 0.05 --name r18_arcface_kd
中断后续跑 (学习率曲线会接上):
    python tools/train_recognition.py ...相同参数... --resume runs/r18_arcface_kd/last.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision.datasets import read_lfw_pairs  # noqa: E402
from facevision.paths import DATA, OUTPUTS  # noqa: E402
from facevision.training.backbones import get_backbone  # noqa: E402
from facevision.training.data import MemmapBatches, to_tensor  # noqa: E402
from facevision.training.losses import build_head, distillation_loss  # noqa: E402
from facevision.utils import disable_power_throttling  # noqa: E402
from facevision.verification import evaluate_verification  # noqa: E402


class LFWEvaluator:
    """复用 eval_lfw.py 缓存的对齐人脸, 每个 epoch 快速测一次 LFW。"""

    def __init__(self, device):
        z = np.load(OUTPUTS / "lfw" / "aligned.npz", allow_pickle=False)
        self.aligned = z["aligned"]
        index = {Path(p): i for i, p in enumerate(z["paths"])}
        pairs = read_lfw_pairs(DATA / "lfw")
        self.i1 = np.array([index[a] for a, _, _ in pairs])
        self.i2 = np.array([index[b] for _, b, _ in pairs])
        self.issame = np.array([s for _, _, s in pairs])
        self.device = device

    @torch.no_grad()
    def __call__(self, model) -> dict:
        model.eval()
        feats = []
        for i in range(0, len(self.aligned), 256):
            x = to_tensor(self.aligned[i:i + 256], self.device)
            f = model(x).float() + model(x.flip(-1)).float()
            feats.append(F.normalize(f, dim=1).cpu().numpy())
        model.train()
        feats = np.concatenate(feats)
        scores = (feats[self.i1] * feats[self.i2]).sum(1)
        return evaluate_verification(scores, self.issame).summary()


def lr_at(step, total, warmup, base_lr):
    """先线性预热 (warmup), 再余弦退火到 0。预热避免一开始大学习率把随机初始化的网络带飞。"""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1 + math.cos(math.pi * p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="r18")
    ap.add_argument("--head", default="arcface", choices=["softmax", "sphereface", "cosface", "arcface", "curricularface"])
    ap.add_argument("--kd-weight", type=float, default=50.0, help="蒸馏损失权重 λ, 0 表示不蒸馏")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--subset", type=int, default=0, help="只用前 N 个身份 (做快速对比实验)")
    ap.add_argument("--device", default="xpu" if torch.xpu.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    disable_power_throttling()   # 后台进程不被 Windows 降到能效核 (喂数据的 CPU 端也要快)

    name = args.name or f"{args.backbone}_{args.head}_kd{args.kd_weight:g}"
    run_dir = ROOT / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    dev = args.device
    torch.manual_seed(0)

    # ---------- 数据 ----------
    images = np.load(DATA / "celeba" / "aligned112.npy", mmap_mode="r")
    meta = np.load(DATA / "celeba" / "meta.npz", allow_pickle=False)
    labels = meta["labels"].astype(np.int64)
    teacher = np.load(DATA / "celeba" / "teacher_r50.npy", mmap_mode="r") if args.kd_weight > 0 else None
    indices = np.arange(len(labels))
    if args.subset:
        indices = indices[labels < args.subset]
    num_classes = int(labels[indices].max()) + 1
    batches = MemmapBatches(images, indices, labels, args.batch_size, teacher)
    print(f"训练样本 {len(indices)}, 身份数 {num_classes}, 每轮 {len(batches)} 个 batch")

    # ---------- 模型 ----------
    model = get_backbone(args.backbone).to(dev).to(memory_format=torch.channels_last)
    head = build_head(args.head, 512, num_classes).to(dev)
    # BN 和 PReLU 的参数不做权重衰减 (衰减它们没有正则化意义, 反而有害)
    decay, no_decay = [], []
    for p in model.parameters():
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": head.parameters(), "weight_decay": args.weight_decay},
    ], lr=args.lr, momentum=0.9, nesterov=True)
    total_steps = args.epochs * len(batches)
    warmup = int(args.warmup_epochs * len(batches))
    start_epoch, step = 0, 0
    history = []
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["backbone"])
        head.load_state_dict(ck["head"])
        opt.load_state_dict(ck["opt"])
        start_epoch, step, history = ck["epoch"] + 1, ck["step"], ck.get("history", [])
        # 中断时最后一轮没跑完, 检查点之后写进日志的行要删掉, 否则续训后曲线会重复
        log_path = run_dir / "train_log.csv"
        rows = log_path.read_text(encoding="utf-8").splitlines()
        kept = rows[:1] + [r for r in rows[1:] if int(r.split(",")[0]) <= step]
        log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    evaluator = LFWEvaluator(dev)
    log_f = open(run_dir / "train_log.csv", "a", newline="")
    log = csv.writer(log_f)
    if step == 0:
        log.writerow(["step", "epoch", "lr", "loss", "loss_margin", "loss_kd", "train_acc", "img_per_s"])
    amp = not args.no_amp

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        meter = {"loss": 0.0, "lm": 0.0, "kd": 0.0, "acc": 0.0, "n": 0}
        for imgs, y, t_feat in batches:
            lr = lr_at(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = to_tensor(imgs, dev)
            y = torch.from_numpy(y).to(dev)
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=amp):
                emb = model(x)
            emb = emb.float()
            loss_margin, cos = head(emb, y)
            loss = loss_margin
            loss_kd = torch.zeros((), device=dev)
            if t_feat is not None:
                loss_kd = distillation_loss(emb, torch.from_numpy(t_feat).to(dev))
                loss = loss + args.kd_weight * loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # 梯度裁剪: 防止偶发的大梯度把训练搞崩
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
            opt.step()
            step += 1

            meter["loss"] += loss.item()
            meter["lm"] += loss_margin.item()
            meter["kd"] += loss_kd.item()
            meter["acc"] += (cos.argmax(1) == y).float().mean().item()
            meter["n"] += 1
            if step % 100 == 0:
                n = meter["n"]
                speed = n * args.batch_size / (time.time() - t0)
                row = [step, epoch, f"{lr:.5f}", f"{meter['loss'] / n:.4f}", f"{meter['lm'] / n:.4f}",
                       f"{meter['kd'] / n:.5f}", f"{meter['acc'] / n:.4f}", f"{speed:.0f}"]
                log.writerow(row)
                log_f.flush()
                print(f"[ep {epoch} step {step}/{total_steps}] lr={lr:.4f} loss={meter['loss'] / n:.3f} "
                      f"margin={meter['lm'] / n:.3f} kd={meter['kd'] / n:.4f} acc={meter['acc'] / n:.3f} {speed:.0f} img/s",
                      flush=True)

        lfw = evaluator(model)
        n = max(meter["n"], 1)
        record = {"epoch": epoch, "lfw_acc": lfw["accuracy"], "lfw_std": lfw["accuracy_std"],
                  "tar@far1e-3": lfw["TAR@FAR=0.001"], "loss": meter["loss"] / n,
                  "train_acc": meter["acc"] / n, "minutes": (time.time() - t0) / 60}
        history.append(record)
        print("EPOCH", json.dumps(record), flush=True)
        ck = {"backbone": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
              "epoch": epoch, "step": step, "history": history, "args": vars(args)}
        torch.save(ck, run_dir / "last.pt")
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))

    # 最终只保存骨干网络 (部署时不需要分类头)
    torch.save({"backbone": model.state_dict(), "args": vars(args), "history": history}, run_dir / "final.pt")
    print("训练完成:", run_dir)


if __name__ == "__main__":
    main()
