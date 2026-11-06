"""损失函数消融实验: 同样的网络、数据、训练轮数, 只换损失函数, 比较 LFW 准确率。

为了在合理时间内跑完, 用 MobileFaceNet + CelebA 前 2000 个身份 (约 4 万张) + 不蒸馏。
绝对准确率会比完整训练低很多, 但"谁比谁好"的相对关系是有意义的。
用法: python tools/ablation_losses.py --device cpu --epochs 6
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADS = ["softmax", "sphereface", "cosface", "arcface", "curricularface"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--subset", type=int, default=2000)
    ap.add_argument("--backbone", default="mbf")
    ap.add_argument("--heads", nargs="*", default=HEADS)
    args = ap.parse_args()
    results = {}
    out = ROOT / "outputs" / "train" / "ablation_losses.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    for head in args.heads:
        name = f"abl_{args.backbone}_{head}_s{args.subset}"
        hist = ROOT / "runs" / name / "history.json"
        if not (hist.exists() and len(json.loads(hist.read_text())) >= args.epochs):
            cmd = [sys.executable, str(ROOT / "tools" / "train_recognition.py"), "--backbone", args.backbone,
                   "--head", head, "--kd-weight", "0", "--epochs", str(args.epochs), "--subset", str(args.subset),
                   "--device", args.device, "--name", name]
            if args.device == "cpu":
                cmd.append("--no-amp")
            print(">>", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        h = json.loads(hist.read_text())
        results[head] = {"lfw_acc_per_epoch": [e["lfw_acc"] for e in h], "final_lfw_acc": h[-1]["lfw_acc"],
                         "final_tar@far1e-3": h[-1]["tar@far1e-3"], "final_train_acc": h[-1]["train_acc"]}
        out.write_text(json.dumps({"setting": vars(args), "results": results}, indent=2), encoding="utf-8")
        print(head, results[head], flush=True)


if __name__ == "__main__":
    main()
