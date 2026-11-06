"""人脸库 (1:N 识别): 注册若干人的特征, 查询时找最相似的那个人。

1:1 验证 (verification): "这两张是不是同一个人?"  —— 比一次相似度
1:N 识别 (identification): "这个人是库里的谁?"    —— 和库里每个人比, 取最高分, 再看是否超过阈值
库很大时 (百万级) 会用向量检索库 (Faiss 等) 做近似最近邻, 这里用矩阵乘法暴力检索就够了。
"""
from __future__ import annotations

import numpy as np


class FaceGallery:
    def __init__(self, threshold: float = 0.3):
        self.names: list[str] = []
        self.feats = np.zeros((0, 512), np.float32)
        self.threshold = threshold

    def add(self, name: str, embedding: np.ndarray):
        """同一个人注册多张时, 把特征平均后再归一化 ("类中心"), 比单张更稳。"""
        emb = embedding / np.linalg.norm(embedding)
        if name in self.names:
            i = self.names.index(name)
            merged = self.feats[i] + emb
            self.feats[i] = merged / np.linalg.norm(merged)
        else:
            self.names.append(name)
            self.feats = np.vstack([self.feats, emb[None].astype(np.float32)])

    def search(self, embedding: np.ndarray, topk: int = 3):
        if not self.names:
            return []
        emb = embedding / np.linalg.norm(embedding)
        sims = self.feats @ emb                      # 一次矩阵乘法算出和所有人的余弦相似度
        order = np.argsort(-sims)[:topk]
        return [(self.names[i], float(sims[i])) for i in order]

    def identify(self, embedding: np.ndarray) -> tuple[str, float]:
        res = self.search(embedding, 1)
        if not res or res[0][1] < self.threshold:
            return "unknown", res[0][1] if res else 0.0
        return res[0]
