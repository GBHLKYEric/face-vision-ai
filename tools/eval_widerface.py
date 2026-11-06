"""在 WIDER FACE 验证集上评测人脸检测器 (Easy / Medium / Hard 三档 AP)。

用法:
    python tools/eval_widerface.py                    # 标准设置: 640x640 输入
    python tools/eval_widerface.py --size 1280 --tag scrfd_1280   # 大输入, 小脸更多
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision.detector import SCRFD  # noqa: E402
from facevision.paths import DATA, OUTPUTS  # noqa: E402
from facevision.utils import disable_power_throttling  # noqa: E402
from facevision.widerface_eval import evaluate  # noqa: E402

WIDER = DATA / "widerface"
GT_DIR = WIDER / "_retinaface_repo" / "Pytorch_Retinaface-master" / "widerface_evaluate" / "ground_truth"
OUT = OUTPUTS / "widerface"


def run_detector(size: int, conf: float):
    det = SCRFD(input_size=(size, size), conf_thresh=conf, nms_thresh=0.4)
    img_root = WIDER / "WIDER_val" / "images"
    files = sorted(img_root.glob("*/*.jpg"))
    preds: dict[str, dict[str, np.ndarray]] = {}
    t = time.time()
    for f in tqdm(files, desc=f"detect@{size}"):
        img = cv2.imread(str(f))
        dets, _ = det.detect(img)
        boxes = np.zeros((len(dets), 5), np.float32)
        if len(dets):
            boxes[:, :2] = dets[:, :2]
            boxes[:, 2:4] = dets[:, 2:4] - dets[:, :2]   # 转成 (x, y, w, h)
            boxes[:, 4] = dets[:, 4]
        preds.setdefault(f.parent.name, {})[f.stem] = boxes
    ms = (time.time() - t) / len(files) * 1000
    return preds, ms


def merge_scales(tags: list[str], nms_thresh: float = 0.4):
    """多尺度测试: 把几个输入尺度的检测框合在一起再做一次 NMS。
    小尺度擅长大脸 (Easy), 大尺度擅长小脸 (Hard), 合并后两头兼顾。"""
    from facevision.detector import nms
    loaded = [pickle.loads((OUT / f"preds_{t}.pkl").read_bytes()) for t in tags]
    merged: dict[str, dict[str, np.ndarray]] = {}
    for event in loaded[0][0]:
        merged[event] = {}
        for img in loaded[0][0][event]:
            boxes = np.concatenate([p[event][img] for p, _ in loaded])
            if len(boxes) == 0:
                merged[event][img] = boxes
                continue
            xyxy = boxes.copy()
            xyxy[:, 2:4] += xyxy[:, :2]
            order = np.argsort(-xyxy[:, 4])
            keep = nms(xyxy[order], nms_thresh)
            merged[event][img] = boxes[order][keep]
    return merged, float(sum(ms for _, ms in loaded))


def plot_pr(results, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=120)
    for name, r in results.items():
        ax.plot(r["recall"], r["precision"], label=f"{name}  AP={r['ap'] * 100:.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.02, help="评测时用很低的阈值, 让 PR 曲线完整")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--merge", nargs="*", default=None, help="合并已缓存的多个尺度, 如 --merge scrfd10g_640 scrfd10g_1280")
    args = ap.parse_args()
    disable_power_throttling()
    tag = args.tag or (f"scrfd10g_{'+'.join(t.split('_')[-1] for t in args.merge)}" if args.merge
                       else f"scrfd10g_{args.size}")
    OUT.mkdir(parents=True, exist_ok=True)

    cache = OUT / f"preds_{tag}.pkl"
    if args.merge:
        preds, ms = merge_scales(args.merge)
    elif cache.exists():
        preds, ms = pickle.loads(cache.read_bytes())
    else:
        preds, ms = run_detector(args.size, args.conf)
        cache.write_bytes(pickle.dumps((preds, ms)))
    results = evaluate(preds, GT_DIR)
    summary = {"model": tag, "input_size": "+".join(t.split("_")[-1] for t in args.merge) if args.merge else args.size,
               "ms_per_image": round(ms, 1),
               **{f"AP_{k}": round(v["ap"] * 100, 2) for k, v in results.items()},
               "num_faces": {k: v["num_faces"] for k, v in results.items()}}
    print(json.dumps(summary, indent=2))
    (OUT / f"result_{tag}.json").write_text(json.dumps(summary, indent=2))
    plot_pr(results, OUT / f"pr_{tag}.png", f"WIDER FACE val - SCRFD-10G @{args.size}")


if __name__ == "__main__":
    main()
