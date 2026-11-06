"""数据集分析: LFW / CelebA / WIDER FACE 的规模、类别分布、样本示例。

输出到 outputs/datasets/: 统计 JSON + 分布图 + 样例拼图。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from facevision.datasets import read_lfw_pairs  # noqa: E402
from facevision.paths import DATA, OUTPUTS  # noqa: E402

OUT = OUTPUTS / "datasets"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def grid(images, cols, size=112):
    images = [cv2.resize(im, (size, size)) for im in images]
    while len(images) % cols:
        images.append(np.full((size, size, 3), 255, np.uint8))
    return np.vstack([np.hstack(images[i:i + cols]) for i in range(0, len(images), cols)])


def lfw_stats():
    root = DATA / "lfw"
    people = sorted(p for p in root.iterdir() if p.is_dir())
    counts = np.array([len(list(p.glob("*.jpg"))) for p in people])
    pairs = read_lfw_pairs(root)
    stats = {"人数": len(people), "图片数": int(counts.sum()), "只有1张图的人": int((counts == 1).sum()),
             "图片最多的人": f"{people[counts.argmax()].name} ({counts.max()})",
             "评测对数": len(pairs), "正样本对": sum(s for *_, s in pairs),
             "图片尺寸": "250x250", "每人图片数中位数": float(np.median(counts))}
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    ax.hist(np.clip(counts, 1, 30), bins=30, color="#4C72B0")
    ax.set_yscale("log")
    ax.set_xlabel("每人图片数 (>30 记为 30)")
    ax.set_ylabel("人数 (对数)")
    ax.set_title("LFW: 长尾分布 —— 大多数人只有 1 张图")
    fig.tight_layout()
    fig.savefig(OUT / "lfw_images_per_person.png")
    rng = np.random.default_rng(0)
    sample = [cv2.imread(str(p)) for p in rng.choice(sorted(root.glob("*/*.jpg")), 24, replace=False)]
    cv2.imwrite(str(OUT / "lfw_samples.jpg"), grid(sample, 8, 125))
    return stats


def celeba_stats():
    meta = np.load(DATA / "celeba" / "meta.npz", allow_pickle=False)
    attrs, names = meta["attrs"], [str(a) for a in meta["attr_names"]]
    ids = meta["labels"]
    per_id = np.bincount(ids)
    split = Counter(meta["split"].tolist())
    stats = {"图片数": int(len(ids)), "身份数": int(len(per_id)), "每人图片数(均值)": round(float(per_id.mean()), 1),
             "每人图片数(最少/最多)": f"{per_id.min()}/{per_id.max()}", "官方划分": dict(split),
             "属性数": len(names), "SCRFD 漏检数": int((meta["scores"] == 0).sum()),
             "属性正例比例": {n: round(float(attrs[:, i].mean()), 3) for i, n in enumerate(names)}}
    order = np.argsort(attrs.mean(0))
    fig, ax = plt.subplots(figsize=(6, 8), dpi=120)
    ax.barh([names[i] for i in order], attrs.mean(0)[order], color="#DD8452")
    ax.set_xlabel("正例比例")
    ax.set_title("CelebA 40 个属性的分布 (严重不均衡)")
    fig.tight_layout()
    fig.savefig(OUT / "celeba_attributes.png")
    images = np.load(DATA / "celeba" / "aligned112.npy", mmap_mode="r")
    rng = np.random.default_rng(1)
    idx = np.sort(rng.choice(len(ids), 32, replace=False))
    cv2.imwrite(str(OUT / "celeba_aligned_samples.jpg"), grid([images[i] for i in idx], 8))
    return stats


def wider_stats():
    gt = (DATA / "widerface" / "wider_face_split" / "wider_face_val_bbx_gt.txt").read_text().splitlines()
    sizes, faces_per_img, i = [], [], 0
    while i < len(gt):
        n = int(gt[i + 1])
        rows = gt[i + 2:i + 2 + max(n, 1)]
        boxes = [list(map(int, r.split()[:4])) for r in rows] if n else []
        faces_per_img.append(n)
        sizes += [np.sqrt(max(w, 1) * max(h, 1)) for _, _, w, h in boxes]
        i += 2 + max(n, 1)
    sizes = np.array(sizes)
    stats = {"验证集图片": len(faces_per_img), "人脸总数": int(sum(faces_per_img)),
             "单图最多人脸": int(max(faces_per_img)),
             "人脸尺寸<16像素占比": round(float((sizes < 16).mean()), 3),
             "人脸尺寸中位数(像素)": float(np.median(sizes))}
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=120)
    ax.hist(np.log2(np.clip(sizes, 2, 1024)), bins=40, color="#55A868")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([str(2 ** k) for k in range(1, 11)])
    ax.set_xlabel("人脸边长 (像素, 对数刻度)")
    ax.set_ylabel("数量")
    ax.set_title("WIDER FACE val: 大量极小人脸 —— Hard 档的难点")
    fig.tight_layout()
    fig.savefig(OUT / "wider_face_sizes.png")
    return stats


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"LFW": lfw_stats(), "WIDER FACE": wider_stats()}
    if (DATA / "celeba" / "meta.npz").exists():
        stats["CelebA"] = celeba_stats()
    text = json.dumps(stats, ensure_ascii=False, indent=2)
    (OUT / "dataset_stats.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
