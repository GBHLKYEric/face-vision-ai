"""数据集读取工具。"""
from __future__ import annotations

from pathlib import Path

from .paths import DATA


def lfw_image_path(root: Path, name: str, idx: int) -> Path:
    return root / name / f"{name}_{idx:04d}.jpg"


def read_lfw_pairs(root: Path = DATA / "lfw", pairs_file: str = "pairs.txt"):
    """解析 LFW 官方 pairs.txt。

    第一行 "10 300": 10 折, 每折 300 对正样本 + 300 对负样本。
    正样本行: 姓名 编号1 编号2
    负样本行: 姓名1 编号1 姓名2 编号2
    返回 [(路径1, 路径2, 是否同一人), ...] 共 6000 项, 顺序即折的顺序。
    """
    root = Path(root)
    lines = (root / pairs_file).read_text().strip().splitlines()
    pairs = []
    for line in lines[1:]:
        p = line.split()
        if len(p) == 3:
            pairs.append((lfw_image_path(root, p[0], int(p[1])), lfw_image_path(root, p[0], int(p[2])), True))
        elif len(p) == 4:
            pairs.append((lfw_image_path(root, p[0], int(p[1])), lfw_image_path(root, p[2], int(p[3])), False))
    return pairs
