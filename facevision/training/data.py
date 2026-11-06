"""训练数据读取。

对齐好的人脸都存在一个 (N,112,112,3) 的 uint8 .npy 里, 用内存映射 (mmap) 打开:
文件不会一次性读进内存, 访问到哪部分操作系统才读哪部分, 而且会被系统缓存,
第二轮起几乎就是读内存的速度。一个 batch 直接用花式索引取出来, 搬到 GPU 上再做归一化和增强,
比 PyTorch DataLoader 多进程解码 JPEG 简单得多, 对这种小图非常够用。
"""
from __future__ import annotations

import numpy as np
import torch


def to_tensor(batch_bgr: np.ndarray, device) -> torch.Tensor:
    """uint8 BGR (B,112,112,3) -> float RGB (B,3,112,112), 归一化到 [-1, 1]。"""
    x = torch.from_numpy(np.ascontiguousarray(batch_bgr)).to(device, non_blocking=True)
    x = x.flip(-1).permute(0, 3, 1, 2).float()      # BGR->RGB, NHWC->NCHW
    return ((x - 127.5) / 127.5).contiguous(memory_format=torch.channels_last)


class MemmapBatches:
    """按 batch 随机迭代内存映射的人脸数组, 同时返回标签和老师特征 (若有)。"""

    def __init__(self, images: np.ndarray, indices: np.ndarray, labels: np.ndarray,
                 batch_size: int, teacher: np.ndarray | None = None, seed: int = 0):
        self.images, self.indices, self.labels = images, indices, labels
        self.teacher = teacher
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __iter__(self):
        order = self.rng.permutation(self.indices)
        for b in range(len(self)):
            idx = np.sort(order[b * self.batch_size:(b + 1) * self.batch_size])  # 排序后读磁盘更连续
            imgs = self.images[idx]
            # 数据增强: 随机水平翻转 (人脸左右对称, 翻转后还是同一个人)
            flip = self.rng.random(len(idx)) < 0.5
            imgs[flip] = imgs[flip, :, ::-1]
            teacher = None if self.teacher is None else self.teacher[idx].astype(np.float32)
            yield imgs, self.labels[idx], teacher
