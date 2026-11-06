"""StarGAN 训练脚本 (CelebA 128x128, 5 个属性: 黑/金/棕发、男性、年轻)。

训练循环改编自官方 solver.py (https://github.com/yunjey/stargan, MIT License), 超参数默认与官方一致。

一次迭代 = 1 步判别器 D 更新, 每 n_critic=5 步再做 1 步生成器 G 更新:

  D 的目标 (WGAN-GP + 属性分类):
      L_D = -E[D_src(x)] + E[D_src(G(x,c))] + λ_cls·L_cls(D_cls(x), c_org) + λ_gp·E[(‖∇_x̂ D_src(x̂)‖₂ - 1)²]
      * 前两项: Wasserstein 距离的估计, D 给真图打高分、给假图打低分 (分数无上下界, 不是概率)
      * 梯度惩罚: x̂ 是真假图之间的随机插值, 约束 D 对输入的梯度范数≈1 (1-Lipschitz),
        这是 Wasserstein 距离成立的前提, 比原始 GAN 的 log 损失训练稳定得多
      * 属性分类只用真图训练: D 学会"认属性"
  G 的目标:
      L_G = -E[D_src(G(x,c))] + λ_cls·L_cls(D_cls(G(x,c)), c_trg) + λ_rec·‖x - G(G(x,c_trg), c_org)‖₁
      * 对抗项: 骗 D 给假图打高分 (看起来像真照片)
      * 分类项: 假图要被 D 认成目标属性 (属性真的改了)
      * 循环重建项: 改完再改回来要还原 (只改属性, 不改身份/背景)
  目标标签 c_trg = 把本 batch 的真实标签随机打乱, 保证属性组合都是"真实存在"的组合。

用法:
    python tools/train_stargan.py --device xpu                                    # 完整训练 (官方: 20 万步)
    python tools/train_stargan.py --device xpu --num-iters 300 --out outputs/stargan/train_sanity \
        --batch-size 16 --log-step 5 --sample-step 100 --save-step 300            # 短暂的"健全性"检查
    python tools/train_stargan.py --plot-only --out outputs/stargan/train_sanity  # 只重画损失曲线
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facevision.gan.data import CelebAParquetStream, decode_png, iter_rows  # noqa: E402
from facevision.gan.stargan import (ATTRS, Discriminator, Generator, SINGLE_EDITS,  # noqa: E402
                                    celeba_transform, official_test_labels, to_image)
from facevision.gan.viz import save_grid  # noqa: E402
from facevision.paths import OUTPUTS  # noqa: E402

LOSS_KEYS = ["D/loss_real", "D/loss_fake", "D/loss_cls", "D/loss_gp", "G/loss_fake", "G/loss_rec", "G/loss_cls"]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    if name == "xpu" and not torch.xpu.is_available():
        raise RuntimeError("torch.xpu 不可用, 请改用 --device cpu")
    return torch.device(name)


def classification_loss(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """多标签分类: 每个属性一个独立的二分类 (sigmoid + BCE), 对属性求和、对 batch 求平均 (同官方)。"""
    return F.binary_cross_entropy_with_logits(logit, target, reduction="sum") / logit.size(0)


def gradient_penalty(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """WGAN-GP: E[(‖∂D(x̂)/∂x̂‖₂ - 1)²]。create_graph=True 是因为这个惩罚本身也要对 D 的参数求导。"""
    grad = torch.autograd.grad(outputs=out, inputs=x, grad_outputs=torch.ones_like(out),
                               retain_graph=True, create_graph=True)[0]
    grad = grad.view(grad.size(0), -1)
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def infinite(loader: DataLoader, dataset: CelebAParquetStream):
    """数据流读完一遍 (一个 epoch) 后换个随机种子从头再来。"""
    epoch = 0
    while True:
        dataset.set_epoch(epoch)
        yield from loader
        epoch += 1


def fixed_samples(n: int, stride: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """从 valid 划分每隔 stride 行取 1 张, 共 n 张固定图, 训练中定期看 G 把它们改成什么样。

    (parquet 里的行按身份聚在一起, 连续取的话 8 张往往是同一个人。)
    """
    tf = celeba_transform(train=False)
    xs, cs = [], []
    for k, (_, b, lab) in enumerate(iter_rows("valid", ATTRS, start=0)):
        if k % stride:
            continue
        xs.append(tf(decode_png(b)))
        cs.append(torch.from_numpy(lab))
        if len(xs) == n:
            break
    return torch.stack(xs), torch.stack(cs)


@torch.no_grad()
def save_samples(G: Generator, x_fixed: torch.Tensor, c_fixed: torch.Tensor, device, path: Path) -> None:
    was_training = G.training
    G.train()   # 与官方一致: InstanceNorm 用单张图自身的统计量 (见 stargan.use_instance_stats)
    outs = [x_fixed] + [G(x_fixed.to(device), c.to(device)).cpu() for c in official_test_labels(c_fixed)]
    G.train(was_training)
    imgs = [to_image(o)[..., ::-1] for o in outs]           # RGB -> BGR (cv2 写图)
    rows = [[im[i] for im in imgs] for i in range(len(x_fixed))]
    save_grid(rows, path, ["input", *SINGLE_EDITS])


def plot_losses(csv_path: Path, png_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return
    it = np.array([int(r["iter"]) for r in rows])
    v = {k: np.array([float(r[k]) for r in rows]) for k in LOSS_KEYS}
    v["W"] = -(v["D/loss_real"] + v["D/loss_fake"])

    def ema(y: np.ndarray, beta: float = 0.8) -> np.ndarray:
        """指数滑动平均: GAN 的损失抖动很大, 看趋势要平滑 (原始值画成浅色)。"""
        out, s = np.empty_like(y), np.nan
        for j, val in enumerate(y):
            if not np.isnan(val):      # log_step < n_critic 时某些窗口没有 G 的更新, 记为 nan, 跳过
                s = val if np.isnan(s) else beta * s + (1 - beta) * val
            out[j] = s
        return out

    def line(a, key, label, color, **kw):
        a.plot(it, v[key], color=color, alpha=0.25, lw=1)
        a.plot(it, ema(v[key]), color=color, label=label, **kw)

    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    a = ax[0, 0]
    line(a, "D/loss_real", "D/loss_real = -E[D(x)]", "tab:blue")
    line(a, "D/loss_fake", "D/loss_fake = E[D(G(x,c))]", "tab:orange")
    line(a, "W", "W estimate = E[D(x)] - E[D(fake)]", "k", lw=2)
    a.set_title("WGAN critic terms"), a.legend(fontsize=8)
    a = ax[0, 1]
    line(a, "D/loss_gp", "D/loss_gp", "tab:blue")
    a.set_yscale("log"), a.set_title("gradient penalty (log scale)"), a.legend(fontsize=8)
    a = ax[1, 0]
    line(a, "D/loss_cls", "D/loss_cls (real imgs)", "tab:blue")
    line(a, "G/loss_cls", "G/loss_cls (fake imgs vs target)", "tab:orange")
    a.axhline(5 * np.log(2), color="gray", ls="--", lw=1, label="5 ln2 (random guess)")
    a.set_title("attribute classification (BCE, summed over 5 attrs)"), a.legend(fontsize=8)
    a = ax[1, 1]
    line(a, "G/loss_rec", "G/loss_rec (cycle L1)", "tab:green")
    a2 = a.twinx()
    a2.plot(it, ema(v["G/loss_fake"]), color="tab:red", alpha=0.6, label="G/loss_fake = -E[D(fake)] (EMA)")
    a.set_title("generator"), a.legend(loc="upper left", fontsize=8), a2.legend(loc="upper right", fontsize=8)
    for a in ax.flat:
        a.set_xlabel("iteration"), a.grid(alpha=0.3)
    fig.suptitle(f"StarGAN training losses ({csv_path.parent.name})")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto", help="xpu / cpu / auto")
    ap.add_argument("--out", default=str(OUTPUTS / "stargan" / "train"))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-iters", type=int, default=200000)
    ap.add_argument("--num-iters-decay", type=int, default=100000, help="最后这么多步里学习率线性降到 0")
    ap.add_argument("--g-lr", type=float, default=1e-4)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--beta1", type=float, default=0.5)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--n-critic", type=int, default=5, help="每更新 D 这么多次才更新 1 次 G")
    ap.add_argument("--lambda-cls", type=float, default=1.0)
    ap.add_argument("--lambda-rec", type=float, default=10.0)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--lr-update-step", type=int, default=1000)
    ap.add_argument("--log-step", type=int, default=10)
    ap.add_argument("--sample-step", type=int, default=1000)
    ap.add_argument("--save-step", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4, help="CPU 线程数 (device=cpu 时)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-iters", type=int, default=0, help="从 out/models/{N}-G.ckpt 继续训练")
    ap.add_argument("--init-from", default=None, help="用已有权重初始化, 例如 weights/stargan/200000 (微调)")
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    model_dir, sample_dir = out / "models", out / "samples"
    csv_path, png_path = out / "loss_log.csv", out / "loss_curve.png"
    if args.plot_only:
        plot_losses(csv_path, png_path)
        return
    model_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print("device:", device)

    G, D = Generator().to(device), Discriminator().to(device)
    start = 0
    if args.init_from:
        G.load_state_dict(torch.load(f"{args.init_from}-G.ckpt", map_location="cpu", weights_only=True))
        D.load_state_dict(torch.load(f"{args.init_from}-D.ckpt", map_location="cpu", weights_only=True))
    if args.resume_iters:
        start = args.resume_iters
        G.load_state_dict(torch.load(model_dir / f"{start}-G.ckpt", map_location="cpu", weights_only=True))
        D.load_state_dict(torch.load(model_dir / f"{start}-D.ckpt", map_location="cpu", weights_only=True))
    g_opt = torch.optim.Adam(G.parameters(), args.g_lr, betas=(args.beta1, args.beta2))
    d_opt = torch.optim.Adam(D.parameters(), args.d_lr, betas=(args.beta1, args.beta2))
    if args.resume_iters and (model_dir / f"{start}-opt.pt").exists():
        st = torch.load(model_dir / f"{start}-opt.pt", map_location="cpu", weights_only=True)
        g_opt.load_state_dict(st["g"]), d_opt.load_state_dict(st["d"])

    dataset = CelebAParquetStream("train", ATTRS, celeba_transform(train=True), seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    print(f"train images: {dataset.num_rows}")
    data_iter = infinite(loader, dataset)
    x_fixed, c_fixed = fixed_samples(8)

    new_log = not (args.resume_iters and csv_path.exists())
    log_f = open(csv_path, "w" if new_log else "a", newline="", encoding="utf-8")
    writer = csv.writer(log_f)
    if new_log:
        writer.writerow(["iter", "sec", "sec_per_iter", "g_lr", "d_lr", *LOSS_KEYS])

    def lr_at(i: int, base: float) -> float:
        """论文做法: 最后 num_iters_decay 步内学习率线性降到 0 (每 lr_update_step 步更新一次)。

        注: 官方 solver.py 每次只减 base/num_iters_decay, 按默认参数 20 万步结束时学习率只降了 0.1%,
        与论文描述不符; 这里按论文实现线性衰减。
        """
        decay_start = max(args.num_iters - args.num_iters_decay, 0)
        if i < decay_start:
            return base
        steps = (i - decay_start) // args.lr_update_step * args.lr_update_step
        return base * max(1.0 - steps / (args.num_iters - decay_start), 0.0)

    G.train(), D.train()
    acc = {k: [] for k in LOSS_KEYS}
    t0 = t_last = time.time()
    for i in range(start, args.num_iters):
        for opt, base in ((g_opt, args.g_lr), (d_opt, args.d_lr)):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(i, base)

        x_real, c_org = next(data_iter)
        c_trg = c_org[torch.randperm(c_org.size(0))]        # 随机目标 = 打乱后的真实标签
        # (在 CPU 上打乱再搬到设备: 实测 torch-xpu 2.15 上用设备端 randperm 做索引会导致 DEVICE_LOST)
        x_real, c_org, c_trg = x_real.to(device), c_org.to(device), c_trg.to(device)

        # ------------------ 1) 训练判别器 D ------------------
        out_src, out_cls = D(x_real)
        d_loss_real = -out_src.mean()                       # 真图分数越高越好
        d_loss_cls = classification_loss(out_cls, c_org)    # 学会识别真图属性
        with torch.no_grad():                               # 这一步不更新 G, 不必为 G 建计算图
            x_fake = G(x_real, c_trg)
        out_src, _ = D(x_fake)
        d_loss_fake = out_src.mean()                        # 假图分数越低越好
        alpha = torch.rand(x_real.size(0), 1, 1, 1, device=device)
        x_hat = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
        d_loss_gp = gradient_penalty(D(x_hat)[0], x_hat)
        d_loss = d_loss_real + d_loss_fake + args.lambda_cls * d_loss_cls + args.lambda_gp * d_loss_gp
        d_opt.zero_grad(set_to_none=True)
        d_loss.backward()
        d_opt.step()
        for k, v in (("D/loss_real", d_loss_real), ("D/loss_fake", d_loss_fake),
                     ("D/loss_cls", d_loss_cls), ("D/loss_gp", d_loss_gp)):
            acc[k].append(v.item())

        # ------------------ 2) 每 n_critic 步训练一次生成器 G ------------------
        if (i + 1) % args.n_critic == 0:
            x_fake = G(x_real, c_trg)
            out_src, out_cls = D(x_fake)
            g_loss_fake = -out_src.mean()                       # 骗过 D
            g_loss_cls = classification_loss(out_cls, c_trg)    # 属性改到位
            x_rec = G(x_fake, c_org)                            # 改回原属性
            g_loss_rec = (x_real - x_rec).abs().mean()          # 循环一致 (L1)
            g_loss = g_loss_fake + args.lambda_rec * g_loss_rec + args.lambda_cls * g_loss_cls
            g_opt.zero_grad(set_to_none=True)
            d_opt.zero_grad(set_to_none=True)   # G 的反传也会给 D 的参数累积梯度, 清掉
            g_loss.backward()
            g_opt.step()
            for k, v in (("G/loss_fake", g_loss_fake), ("G/loss_rec", g_loss_rec), ("G/loss_cls", g_loss_cls)):
                acc[k].append(v.item())

        if (i + 1) % args.log_step == 0:
            now = time.time()
            means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in acc.items()}
            spi = (now - t_last) / args.log_step
            writer.writerow([i + 1, round(now - t0, 1), round(spi, 3), lr_at(i, args.g_lr), lr_at(i, args.d_lr),
                             *[round(means[k], 5) for k in LOSS_KEYS]])
            log_f.flush()
            print(f"[{i + 1}/{args.num_iters}] {spi:.2f}s/it  " +
                  "  ".join(f"{k}={means[k]:.3f}" for k in LOSS_KEYS), flush=True)
            acc = {k: [] for k in LOSS_KEYS}
            t_last = now
        if (i + 1) % args.sample_step == 0:
            save_samples(G, x_fixed, c_fixed, device, sample_dir / f"{i + 1:06d}.jpg")
        if (i + 1) % args.save_step == 0:
            torch.save(G.state_dict(), model_dir / f"{i + 1}-G.ckpt")
            torch.save(D.state_dict(), model_dir / f"{i + 1}-D.ckpt")
            torch.save({"g": g_opt.state_dict(), "d": d_opt.state_dict()}, model_dir / f"{i + 1}-opt.pt")

    log_f.close()
    plot_losses(csv_path, png_path)
    print("done:", out)


if __name__ == "__main__":
    main()
