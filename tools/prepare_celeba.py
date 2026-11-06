"""把 CelebA 预处理成识别训练用的 112x112 对齐人脸。

每张图: SCRFD 检测 5 个关键点 -> 相似变换对齐 (与 LFW 评测完全相同的流程) -> 112x112。
结果存成一个 uint8 的 .npy 大数组 (内存映射读取, 训练时不用反复解码 PNG):
    data/celeba/aligned112.npy  (N, 112, 112, 3)  BGR
    data/celeba/meta.npz        身份 ID、40 个属性、官方划分、原文件名、检测置信度
多进程并行, 每个进程处理一个 parquet 分片。
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARQUET_DIR = ROOT / "data" / "celeba" / "hf" / "img_align+identity+attr"
OUT_DIR = ROOT / "data" / "celeba"
PART_DIR = OUT_DIR / "aligned_parts"
ATTRS = None  # 由第一个分片的表头决定


def process_shard(path: str):
    os.environ["FV_ORT_THREADS"] = "2"
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image

    from facevision.align import norm_crop
    from facevision.detector import SCRFD

    out = PART_DIR / (Path(path).stem + ".npz")
    if out.exists():
        return str(out)
    det = SCRFD(input_size=(256, 256), conf_thresh=0.3, threads=2)
    table = pq.read_table(path)
    attr_names = [c for c in table.column_names if c not in ("image", "celeb_id")]
    images = table.column("image").to_pylist()
    n = len(images)
    aligned = np.zeros((n, 112, 112, 3), np.uint8)
    kps_all = np.zeros((n, 5, 2), np.float32)
    scores = np.zeros(n, np.float32)
    names = []
    for i, item in enumerate(images):
        img = cv2.cvtColor(np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
        names.append(item["path"])
        dets, kps = det.detect(img, max_num=1)
        if len(dets):
            scores[i] = dets[0, 4]
            kps_all[i] = kps[0]
            aligned[i], _ = norm_crop(img, kps[0])
    ids = np.array(table.column("celeb_id").to_pylist(), np.int64)
    attrs = np.stack([np.array(table.column(a).to_pylist(), bool) for a in attr_names], 1)
    split = Path(path).stem.split("-")[0]
    np.savez(out, aligned=aligned, kps=kps_all, scores=scores, ids=ids, attrs=attrs,
             names=np.array(names), split=np.array([split] * n), attr_names=np.array(attr_names))
    return str(out)


def process_shard_gpu(path: str, det, batch: int = 64):
    """GPU/核显版本: 多线程解码 PNG, 批量检测, 比 CPU 多进程快一个数量级。"""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    import pyarrow.parquet as pq
    from PIL import Image

    from facevision.align import norm_crop

    out = PART_DIR / (Path(path).stem + ".npz")
    if out.exists():
        return str(out)
    table = pq.read_table(path)
    attr_names = [c for c in table.column_names if c not in ("image", "celeb_id")]
    images = table.column("image").to_pylist()
    n = len(images)

    def decode(item):
        return cv2.cvtColor(np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)

    aligned = np.zeros((n, 112, 112, 3), np.uint8)
    kps_all = np.zeros((n, 5, 2), np.float32)
    scores = np.zeros(n, np.float32)
    with ThreadPoolExecutor(8) as pool:
        for s in range(0, n, batch):
            imgs = list(pool.map(decode, images[s:s + batch]))
            for j, (dets, kps) in enumerate(det.detect_batch(imgs, max_num=1)):
                if len(dets):
                    scores[s + j] = dets[0, 4]
                    kps_all[s + j] = kps[0]
                    aligned[s + j], _ = norm_crop(imgs[j], kps[0])
    ids = np.array(table.column("celeb_id").to_pylist(), np.int64)
    attrs = np.stack([np.array(table.column(a).to_pylist(), bool) for a in attr_names], 1)
    split = Path(path).stem.split("-")[0]
    np.savez(out, aligned=aligned, kps=kps_all, scores=scores, ids=ids, attrs=attrs,
             names=np.array([it["path"] for it in images]), split=np.array([split] * n),
             attr_names=np.array(attr_names))
    return str(out)


def merge():
    parts = sorted(PART_DIR.glob("*.npz"))
    metas = [np.load(p, allow_pickle=False) for p in parts]
    n = sum(len(m["ids"]) for m in metas)
    arr = np.lib.format.open_memmap(OUT_DIR / "aligned112.npy", mode="w+", dtype=np.uint8, shape=(n, 112, 112, 3))
    cat = {k: [] for k in ("kps", "scores", "ids", "attrs", "names", "split")}
    i = 0
    for m in metas:
        k = len(m["ids"])
        arr[i:i + k] = m["aligned"]
        for key in cat:
            cat[key].append(m[key])
        i += k
    arr.flush()
    meta = {k: np.concatenate(v) for k, v in cat.items()}
    # 漏检的图: 用全体检测结果的平均关键点位置兜底 (CelebA 本身已粗对齐, 平均位置很靠谱)
    miss = meta["scores"] == 0
    if miss.any():
        print(f"{miss.sum()} 张未检测到人脸, 用平均关键点对齐")
        meta["fallback_kps"] = meta["kps"][~miss].mean(0)
    # 身份 ID 重新编号为 0..C-1
    uniq, labels = np.unique(meta["ids"], return_inverse=True)
    meta["labels"] = labels
    np.savez(OUT_DIR / "meta.npz", attr_names=metas[0]["attr_names"], **meta)
    print(f"合并完成: {n} 张, {len(uniq)} 个身份, 漏检 {int(miss.sum())}")
    return miss


def fill_missing(miss):
    """对漏检样本用平均关键点重新对齐。"""
    import cv2
    import pyarrow.parquet as pq
    from PIL import Image

    from facevision.align import norm_crop
    meta = dict(np.load(OUT_DIR / "meta.npz", allow_pickle=False))
    arr = np.load(OUT_DIR / "aligned112.npy", mmap_mode="r+")
    want = {str(nm): idx for idx, nm in zip(np.where(miss)[0], meta["names"][miss])}
    for f in sorted(glob.glob(str(PARQUET_DIR / "*.parquet"))):
        col = pq.read_table(f, columns=["image"]).column("image").to_pylist()
        for item in col:
            if item["path"] in want:
                img = cv2.cvtColor(np.array(Image.open(io.BytesIO(item["bytes"])).convert("RGB")), cv2.COLOR_RGB2BGR)
                arr[want[item["path"]]], _ = norm_crop(img, meta["fallback_kps"])
    arr.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="cpu", help="cpu = 多进程 ONNX Runtime; xpu/cuda = 批量 GPU 检测")
    args = ap.parse_args()
    from facevision.utils import disable_power_throttling
    disable_power_throttling()
    PART_DIR.mkdir(parents=True, exist_ok=True)
    shards = sorted(glob.glob(str(PARQUET_DIR / "*.parquet")))
    if args.device == "cpu":
        print(f"{len(shards)} 个分片, {args.workers} 个进程")
        with Pool(args.workers) as pool:
            for i, p in enumerate(pool.imap_unordered(process_shard, shards)):
                print(f"[{i + 1}/{len(shards)}] {Path(p).name}", flush=True)
    else:
        from facevision.detector import SCRFDBatch
        det = SCRFDBatch(device=args.device, input_size=(256, 256), conf_thresh=0.3)
        for i, s in enumerate(shards):
            print(f"[{i + 1}/{len(shards)}] {Path(process_shard_gpu(s, det)).name}", flush=True)
    miss = merge()
    if miss.any():
        fill_missing(miss)


if __name__ == "__main__":
    main()
