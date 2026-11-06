"""在 LFW 上按官方 10 折协议评测人脸识别准确率。

完整流程从原始 250x250 图片开始: SCRFD 检测 -> 5 点对齐 -> ArcFace 特征 -> 余弦相似度 -> 10 折。
对齐后的人脸会缓存到 outputs/lfw/aligned.npz, 之后评测别的识别模型 (自己训练的、量化后的)
可以直接复用, 保证"只换识别模型, 其它完全一致"的公平对比。

用法:
    python tools/eval_lfw.py                                     # 预训练 ArcFace-R50 (ONNX)
    python tools/eval_lfw.py --onnx outputs/deploy/r50_int8.onnx --tag r50_int8
    python tools/eval_lfw.py --torch-ckpt runs/xxx/best.pt --tag mine
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facevision.align import ARCFACE_TEMPLATE, norm_crop  # noqa: E402
from facevision.datasets import read_lfw_pairs  # noqa: E402
from facevision.detector import SCRFD  # noqa: E402
from facevision.paths import DATA, OUTPUTS, REC_MODEL  # noqa: E402
from facevision.utils import disable_power_throttling  # noqa: E402
from facevision.verification import evaluate_verification  # noqa: E402

OUT = OUTPUTS / "lfw"


def detect_and_align(paths: list[Path], pad_ratio: float = 0.25):
    """对每张图检测 1 张脸并对齐。LFW 每张图的主角都在正中间, 所以选"又大又居中"的那张脸。

    pad_ratio: 检测前给图片四周补黑边。LFW 是从大图裁出来的, 主角的脸常被画面边缘切掉一块,
    检测器对"不完整的脸"信心不足; 补边后脸完整地落在画面内, 就能稳定检出 (见报告中的错误分析)。
    """
    det = SCRFD(input_size=(640, 640), conf_thresh=0.5)
    aligned = np.zeros((len(paths), 112, 112, 3), np.uint8)
    status = []
    # 实在检测不到时的兜底: 用 LFW 的平均五点位置 (250x250 图里人脸大致固定) 直接对齐
    fallback_kps = ARCFACE_TEMPLATE * (250 / 112 * 0.42) + np.array([125 - 56 * 250 / 112 * 0.42] * 2)
    for i, p in enumerate(tqdm(paths, desc="detect+align")):
        img = cv2.imread(str(p))
        pad = int(pad_ratio * max(img.shape[:2]))
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        dets, kps = det.detect(padded, max_num=1)
        how = "det"
        if len(dets) == 0:
            dets, kps = det.detect(padded, max_num=1, conf_thresh=0.1)
            how = "det_low_thresh"
        if len(dets) == 0:
            k, how = fallback_kps, "fallback"
        else:
            k = kps[0] - pad
        aligned[i], _ = norm_crop(img, k)
        status.append(how)
    return aligned, status


def build_embedder(args):
    if args.torch_ckpt:
        from facevision.training.embedder import TorchEmbedder
        return TorchEmbedder(args.torch_ckpt, device=args.device)
    from facevision.recognizer import ArcFaceONNX
    model = ArcFaceONNX(args.onnx)
    return lambda faces, flip: model.embed(faces, flip_test=flip)


def save_error_grid(errors, pairs, aligned, index, path: Path):
    """把判错的样本对拼成一张大图, 方便肉眼检查 (很多是 LFW 本身的标注错误)。"""
    tiles = []
    for k, (s, j) in enumerate(errors[:24]):
        p1, p2, same = pairs[j]
        a = aligned[index[p1]].copy()
        b = aligned[index[p2]].copy()
        tile = np.hstack([a, b])
        tile = cv2.copyMakeBorder(tile, 18, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        label = f"{'same' if same else 'diff'} cos={s:.2f}"
        cv2.putText(tile, label, (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
        tiles.append(tile)
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(REC_MODEL))
    ap.add_argument("--torch-ckpt", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tag", default="arcface_r50")
    ap.add_argument("--no-flip", action="store_true", help="关闭翻转测试增强")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pad-ratio", type=float, default=0.25, help="检测前四周补边比例, 0 = 不补")
    args = ap.parse_args()
    disable_power_throttling()
    OUT.mkdir(parents=True, exist_ok=True)

    pairs = read_lfw_pairs(DATA / "lfw")
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    index = {p: i for i, p in enumerate(paths)}
    print(f"{len(pairs)} 对, 涉及 {len(paths)} 张图片")

    cache = OUT / ("aligned.npz" if args.pad_ratio > 0 else "aligned_nopad.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        aligned, status = z["aligned"], [str(s) for s in z["status"]]
    else:
        aligned, status = detect_and_align(paths, args.pad_ratio)
        np.savez_compressed(cache, aligned=aligned, status=np.array(status), paths=np.array([str(p) for p in paths]))
    det_stats = {k: status.count(k) for k in sorted(set(status))}
    print("检测情况:", det_stats)

    embed = build_embedder(args)
    flip = not args.no_flip
    feats = []
    t0 = time.time()
    for i in tqdm(range(0, len(aligned), args.batch), desc=f"embed[{args.tag}]"):
        feats.append(embed(list(aligned[i:i + args.batch]), flip))
    feats = np.concatenate(feats)
    embed_ms = (time.time() - t0) / len(aligned) * 1000

    i1 = np.array([index[a] for a, _, _ in pairs])
    i2 = np.array([index[b] for _, b, _ in pairs])
    issame = np.array([s for _, _, s in pairs])
    scores = np.sum(feats[i1] * feats[i2], axis=1)
    res = evaluate_verification(scores, issame)

    summary = {"model": args.tag, "flip_test": flip, "detection": det_stats,
               "embed_ms_per_image": round(embed_ms, 2), **res.summary()}
    thr = res.best_threshold
    wrong = [(float(scores[j]), int(j)) for j in range(len(pairs)) if (scores[j] > thr) != issame[j]]
    summary["num_errors_at_mean_threshold"] = len(wrong)
    summary["errors"] = [{"pair": j, "img1": str(pairs[j][0].relative_to(DATA)),
                          "img2": str(pairs[j][1].relative_to(DATA)), "same": bool(pairs[j][2]),
                          "cos": round(s, 4)} for s, j in wrong]
    print(json.dumps({k: v for k, v in summary.items() if k != "errors"}, ensure_ascii=False, indent=2))

    (OUT / f"result_{args.tag}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(OUT / f"scores_{args.tag}.npz", scores=scores, issame=issame, fpr=res.roc_fpr, tpr=res.roc_tpr)
    save_error_grid(wrong, pairs, aligned, index, OUT / f"errors_{args.tag}.jpg")


if __name__ == "__main__":
    main()
