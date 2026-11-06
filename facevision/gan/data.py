"""直接从 HuggingFace 的 parquet 文件读取 CelebA (对齐 178x218 PNG + 40 个属性 + 官方划分)。

parquet 是"列式"存储: 每个文件分成若干 row group (这里每组 100 行),
一次读一整组最快, 随机读单行很慢。所以训练时用"两级打乱":
    1) 先把所有 (文件, row group) 打乱顺序;
    2) 每次读入若干组放进缓冲区, 在缓冲区内再打乱后逐张输出。
这和 WebDataset / TFRecord 的 shard 级打乱是同一个思路, 内存占用小, 随机性也够用。
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from ..paths import DATA

CELEBA_PARQUET = DATA / "celeba" / "hf" / "img_align+identity+attr"


def parquet_files(split: str, root: Path = CELEBA_PARQUET) -> list[Path]:
    """split: train / valid / test (CelebA 官方划分)。"""
    files = sorted(Path(root).glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"找不到 {root}/{split}-*.parquet")
    return files


def decode_png(b: bytes) -> Image.Image:
    return Image.open(io.BytesIO(b)).convert("RGB")


def iter_rows(split: str, attrs: Sequence[str], start: int = 0,
              root: Path = CELEBA_PARQUET) -> Iterator[tuple[str, bytes, np.ndarray]]:
    """按文件顺序逐行读取, 产出 (原文件名, PNG 字节, 属性 0/1 向量)。用于评测 (顺序固定、可复现)。"""
    k = 0
    for f in parquet_files(split, root):
        pf = pq.ParquetFile(f)
        for g in range(pf.metadata.num_row_groups):
            n = pf.metadata.row_group(g).num_rows
            if k + n <= start:          # 整组都在 start 之前, 直接跳过不解码
                k += n
                continue
            t = pf.read_row_group(g, columns=["image", *attrs])
            images = t.column("image").to_pylist()
            labels = np.stack([np.asarray(t.column(a).to_numpy(zero_copy_only=False), np.float32)
                               for a in attrs], 1)
            for img, lab in zip(images, labels):
                if k >= start:
                    yield img["path"], img["bytes"], lab
                k += 1


class CelebAParquetStream(IterableDataset):
    """训练用的流式数据集: 产出 (图像张量, 属性标签 float 向量)。

    transform: 作用在 PIL 图像上 (例如 stargan.celeba_transform(train=True))。
    每个 epoch 用 seed + epoch 重新打乱; 多个 DataLoader worker 各自负责一部分 row group。
    """

    def __init__(self, split: str, attrs: Sequence[str], transform: Callable,
                 shuffle: bool = True, buffer_groups: int = 16, seed: int = 0,
                 root: Path = CELEBA_PARQUET):
        super().__init__()
        self.attrs = list(attrs)
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_groups = buffer_groups
        self.seed = seed
        self.epoch = 0
        self.shards: list[tuple[str, int]] = []
        self.num_rows = 0
        for f in parquet_files(split, root):
            md = pq.ParquetFile(f).metadata
            self.shards += [(str(f), g) for g in range(md.num_row_groups)]
            self.num_rows += md.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _read_group(self, files: dict, path: str, g: int):
        if path not in files:
            files[path] = pq.ParquetFile(path)
        t = files[path].read_row_group(g, columns=["image", *self.attrs])
        images = [d["bytes"] for d in t.column("image").to_pylist()]
        labels = np.stack([np.asarray(t.column(a).to_numpy(zero_copy_only=False), np.float32)
                           for a in self.attrs], 1)
        return list(zip(images, labels))

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        shards = list(self.shards)
        if self.shuffle:
            rng.shuffle(shards)               # 第 1 级: 打乱 row group 的顺序
        info = get_worker_info()
        if info is not None:                  # 多进程加载: 每个 worker 取不同的 row group
            shards = shards[info.id::info.num_workers]
            rng = random.Random(self.seed + self.epoch * 1000 + info.id)
        files: dict = {}
        for i in range(0, len(shards), self.buffer_groups):
            buf = []
            for path, g in shards[i:i + self.buffer_groups]:
                buf += self._read_group(files, path, g)
            if self.shuffle:
                rng.shuffle(buf)              # 第 2 级: 缓冲区内打乱
            for b, lab in buf:
                yield self.transform(decode_png(b)), torch.from_numpy(lab)
