"""人脸验证 (1:1) 的评测指标。

人脸验证 = 给两张脸, 判断是不是同一个人。做法是算两个特征的余弦相似度,
大于阈值 -> "同一人", 否则 "不同人"。所以评测的核心是: 阈值怎么选、选完准不准。

LFW 标准协议 (10 折交叉验证):
  6000 对图片分成 10 份, 每份 300 对正样本 (同一人) + 300 对负样本 (不同人)。
  轮流拿 9 份来"挑阈值" (找让这 9 份准确率最高的阈值), 在剩下 1 份上测准确率,
  10 次结果取平均。这样阈值从没"见过"测试数据, 结果更可信。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VerificationResult:
    accuracy_mean: float
    accuracy_std: float
    fold_accuracies: list
    fold_thresholds: list
    auc: float
    eer: float
    tar_at_far: dict
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    best_threshold: float

    def summary(self) -> dict:
        return {
            "accuracy": round(self.accuracy_mean * 100, 3),
            "accuracy_std": round(self.accuracy_std * 100, 3),
            "auc": round(self.auc, 6),
            "eer": round(self.eer * 100, 3),
            **{f"TAR@FAR={k}": round(v * 100, 3) for k, v in self.tar_at_far.items()},
            "threshold_cos": round(self.best_threshold, 4),
            "fold_accuracies": [round(a * 100, 2) for a in self.fold_accuracies],
        }


def accuracy_at(scores: np.ndarray, issame: np.ndarray, thr: float) -> float:
    pred = scores > thr
    return float((pred == issame).mean())


def best_threshold(scores: np.ndarray, issame: np.ndarray) -> tuple[float, float]:
    """在给定数据上找让准确率最高的阈值。

    只需要尝试"相邻两个分数的中点"作为候选阈值, 因为在两个相邻分数之间移动阈值,
    预测结果不会变。排序后用累加和一次性算出所有候选阈值的准确率, O(n log n)。
    """
    order = np.argsort(scores)
    s, y = scores[order], issame[order].astype(np.int64)
    n_pos = y.sum()
    # 阈值放在第 i 个分数之后: 前 i+1 个判为负, 其余判为正
    neg_correct = np.cumsum(1 - y)               # 前缀里真负样本数 = 判对的负样本
    pos_correct = n_pos - np.cumsum(y)           # 后缀里真正样本数 = 判对的正样本
    acc = (neg_correct + pos_correct) / len(s)
    # 加上"全部判为正"的情况
    all_pos_acc = n_pos / len(s)
    i = int(np.argmax(acc))
    if all_pos_acc > acc[i]:
        return float(s[0] - 1e-6), float(all_pos_acc)
    thr = (s[i] + s[i + 1]) / 2 if i + 1 < len(s) else s[i] + 1e-6
    return float(thr), float(acc[i])


def roc_curve(scores: np.ndarray, issame: np.ndarray):
    """返回 (FPR, TPR, 阈值)。FPR=误识率 (把不同人认成同一人), TPR=通过率。"""
    order = np.argsort(-scores)
    s, y = scores[order], issame[order].astype(np.int64)
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0], tp / max(y.sum(), 1)])
    fpr = np.concatenate([[0], fp / max((1 - y).sum(), 1)])
    return fpr, tpr, np.concatenate([[np.inf], s])


def tar_at_far(fpr, tpr, far: float) -> float:
    """在误识率不超过 far 的前提下, 能达到的最高通过率。"""
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def evaluate_verification(scores: np.ndarray, issame: np.ndarray, nfolds: int = 10,
                          fars=(1e-3, 1e-2)) -> VerificationResult:
    scores = np.asarray(scores, dtype=np.float64)
    issame = np.asarray(issame, dtype=bool)
    n = len(scores)
    fold_ids = np.arange(n) * nfolds // n   # pairs.txt 本身就是按折顺序排列的
    accs, thrs = [], []
    for k in range(nfolds):
        test = fold_ids == k
        thr, _ = best_threshold(scores[~test], issame[~test])
        thrs.append(thr)
        accs.append(accuracy_at(scores[test], issame[test], thr))

    fpr, tpr, _ = roc_curve(scores, issame)
    auc = float(np.trapezoid(tpr, fpr))
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[i] + fpr[i]) / 2)
    return VerificationResult(
        accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
        fold_accuracies=accs, fold_thresholds=thrs, auc=auc, eer=eer,
        tar_at_far={far: tar_at_far(fpr, tpr, far) for far in fars},
        roc_fpr=fpr, roc_tpr=tpr, best_threshold=float(np.mean(thrs)))
