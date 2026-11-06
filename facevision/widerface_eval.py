"""WIDER FACE 官方评测协议的 numpy 实现 (移植自官方 MATLAB / Python 评测脚本)。

WIDER FACE 验证集 3226 张图, 按人脸大小、遮挡、姿态把每张脸分成 Easy / Medium / Hard 三档:
  Easy  只统计大而清晰的脸, Hard 包含全部脸 (大量 10 像素级的小脸、严重遮挡)。
不在当前档位里的脸算"忽略": 检测到它不算对也不算错。

AP (平均精度) 的计算:
  1. 所有检测框的分数先归一化到 [0,1]
  2. 取 1000 个分数阈值, 对每个阈值统计: 保留下来的框里有多少是对的 (precision),
     所有真实人脸里有多少被找到了 (recall)。框和真值 IoU >= 0.5 才算找到
  3. 画出 precision-recall 曲线, 曲线下面积就是 AP
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def bbox_overlaps(boxes: np.ndarray, query: np.ndarray) -> np.ndarray:
    """IoU 矩阵 (与官方 Cython 实现一致, 宽高按像素计数 +1)。"""
    b_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    q_area = (query[:, 2] - query[:, 0] + 1) * (query[:, 3] - query[:, 1] + 1)
    iw = np.minimum(boxes[:, None, 2], query[None, :, 2]) - np.maximum(boxes[:, None, 0], query[None, :, 0]) + 1
    ih = np.minimum(boxes[:, None, 3], query[None, :, 3]) - np.maximum(boxes[:, None, 1], query[None, :, 1]) + 1
    inter = np.clip(iw, 0, None) * np.clip(ih, 0, None)
    union = b_area[:, None] + q_area[None, :] - inter
    return np.where((iw > 0) & (ih > 0), inter / union, 0.0)


def load_ground_truth(gt_dir: Path):
    gt = loadmat(str(gt_dir / "wider_face_val.mat"))
    settings = {s: loadmat(str(gt_dir / f"wider_{s}_val.mat"))["gt_list"] for s in ("easy", "medium", "hard")}
    return gt["face_bbx_list"], gt["event_list"], gt["file_list"], settings


def image_eval(pred, gt, keep, iou_thresh):
    """单张图: 按分数从高到低逐个检查检测框。

    返回 pred_recall[i] = 前 i+1 个框一共找到了多少张有效人脸,
         proposal[i]   = 1 正常计数, -1 表示这个框落在"被忽略的脸"上 (不计入 precision 分母)。
    """
    p = pred.copy()
    g = gt.copy()
    p[:, 2] += p[:, 0]
    p[:, 3] += p[:, 1]
    g[:, 2] += g[:, 0]
    g[:, 3] += g[:, 1]
    overlaps = bbox_overlaps(p[:, :4], g)
    recall = np.zeros(len(g))
    proposal = np.ones(len(p))
    pred_recall = np.zeros(len(p))
    for h in range(len(p)):
        j = overlaps[h].argmax()
        if overlaps[h, j] >= iou_thresh:
            if keep[j] == 0:            # 匹配到被忽略的脸
                recall[j] = -1
                proposal[h] = -1
            elif recall[j] == 0:        # 第一次找到这张有效人脸
                recall[j] = 1
        pred_recall[h] = (recall == 1).sum()
    return pred_recall, proposal


def img_pr_info(thresh_num, pred, proposal, pred_recall):
    info = np.zeros((thresh_num, 2))
    for t in range(thresh_num):
        thresh = 1 - (t + 1) / thresh_num
        idx = np.where(pred[:, 4] >= thresh)[0]
        if len(idx):
            r = idx[-1]
            info[t, 0] = (proposal[:r + 1] == 1).sum()   # 保留的有效框数
            info[t, 1] = pred_recall[r]                   # 找到的人脸数
    return info


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):        # precision 包络: 右边最大值
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(preds: dict, gt_dir: Path, iou_thresh: float = 0.5, thresh_num: int = 1000):
    """preds: {event_name: {image_name(无扩展名): ndarray[N,5] (x, y, w, h, score) 按分数降序}}"""
    # 分数全局归一化到 [0, 1]
    all_scores = np.concatenate([v[:, 4] for ev in preds.values() for v in ev.values() if len(v)])
    lo, hi = all_scores.min(), all_scores.max()
    preds = {e: {k: (np.hstack([v[:, :4], (v[:, 4:5] - lo) / (hi - lo)]) if len(v) else v)
                 for k, v in ev.items()} for e, ev in preds.items()}

    facebox_list, event_list, file_list, settings = load_ground_truth(gt_dir)
    results = {}
    for name, gt_list in settings.items():
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2))
        for i in range(len(event_list)):
            event = str(event_list[i][0][0])
            img_list = file_list[i][0]
            sub_gt = gt_list[i][0]
            gt_boxes_list = facebox_list[i][0]
            for j in range(len(img_list)):
                pred = preds[event][str(img_list[j][0][0])]
                gt_boxes = gt_boxes_list[j][0].astype(float)
                keep_index = sub_gt[j][0]
                count_face += len(keep_index)
                if len(gt_boxes) == 0 or len(pred) == 0:
                    continue
                keep = np.zeros(len(gt_boxes))
                if len(keep_index):
                    keep[keep_index.ravel() - 1] = 1
                pred_recall, proposal = image_eval(pred, gt_boxes, keep, iou_thresh)
                pr_curve += img_pr_info(thresh_num, pred, proposal, pred_recall)
        precision = pr_curve[:, 1] / np.maximum(pr_curve[:, 0], 1e-12)
        recall = pr_curve[:, 1] / count_face
        results[name] = {"ap": voc_ap(recall, precision), "precision": precision, "recall": recall,
                         "num_faces": count_face}
    return results
