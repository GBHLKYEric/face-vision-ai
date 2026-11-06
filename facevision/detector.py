"""人脸检测: SCRFD (RetinaFace 的高效改进版)。

网络结构 = 骨干网络 + FPN 多尺度特征金字塔 + 3 个检测头 (stride 8/16/32)。
每个特征图上的每个位置放 2 个"锚点", 网络对每个锚点预测:
    - score: 这里是人脸的概率
    - bbox : 锚点中心到框四条边的距离 (left, top, right, bottom), 以 stride 为单位
    - kps  : 5 个关键点相对锚点中心的偏移 (x1,y1,...,x5,y5), 以 stride 为单位
解码 + 非极大值抑制 (NMS) 之后得到最终的人脸框和 5 点关键点。
这里的后处理全部用 numpy 手写, 方便理解 RetinaFace 系检测器的工作原理。
"""
from __future__ import annotations

import cv2
import numpy as np

from .onnx_utils import create_session
from .paths import DET_MODEL


def distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(cx, cy) + (l, t, r, b) -> (x1, y1, x2, y2)"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """(cx, cy) + 10 个偏移 -> 5 个关键点坐标, 形状 (N, 5, 2)"""
    kps = distance.reshape(len(distance), 5, 2) + points[:, None, :]
    return kps


def nms(dets: np.ndarray, iou_thresh: float) -> list[int]:
    """非极大值抑制: 同一张脸常被多个锚点框中, 只保留分数最高的那个。

    dets: (N, 5) = x1, y1, x2, y2, score
    思路: 按分数从高到低, 取出当前最高分框, 删掉和它 IoU > 阈值的其它框, 重复。
    """
    x1, y1, x2, y2, scores = dets.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


class SCRFD:
    """SCRFD-10G 人脸检测器 (InsightFace buffalo_l 中的 det_10g.onnx)。"""

    strides = (8, 16, 32)
    num_anchors = 2

    def __init__(self, model_path=DET_MODEL, input_size=(640, 640), conf_thresh=0.5,
                 nms_thresh=0.4, providers=None, threads=None):
        self.session = create_session(model_path, providers, threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.input_size = tuple(input_size)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self._center_cache: dict[tuple, np.ndarray] = {}

    # ---------- 预处理 ----------
    def _letterbox(self, img: np.ndarray, input_size):
        """等比例缩放后贴到左上角, 右边和下边补黑边, 保证人脸不被拉伸变形。"""
        in_w, in_h = input_size
        im_ratio = img.shape[0] / img.shape[1]
        model_ratio = in_h / in_w
        if im_ratio > model_ratio:
            new_h = in_h
            new_w = int(new_h / im_ratio)
        else:
            new_w = in_w
            new_h = int(new_w * im_ratio)
        scale = new_h / img.shape[0]
        resized = cv2.resize(img, (new_w, new_h))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
        return canvas, scale

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._center_cache:
            # 特征图上每个格子的左上角坐标映射回原图: (x*stride, y*stride)
            ys, xs = np.mgrid[:h, :w]
            centers = (np.stack([xs, ys], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            # 每个位置有 2 个锚点, 中心相同
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self._center_cache[key] = centers
        return self._center_cache[key]

    # ---------- 推理 + 解码 ----------
    def detect(self, img: np.ndarray, input_size=None, max_num: int = 0, conf_thresh=None):
        """img: BGR 图像。返回 (boxes[N,5], kps[N,5,2]), boxes 最后一列是置信度。"""
        input_size = tuple(input_size or self.input_size)
        det_img, scale = self._letterbox(img, input_size)
        # 归一化: (像素 - 127.5) / 128, BGR -> RGB, HWC -> NCHW
        blob = cv2.dnn.blobFromImage(det_img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        return self.postprocess(outs, blob.shape[2:], scale, img.shape, max_num, conf_thresh)

    def postprocess(self, outs, input_hw, scale, img_shape, max_num=0, conf_thresh=None):
        """把网络的 9 个输出解码成人脸框和关键点 (CPU 版和批量 GPU 版共用)。"""
        conf_thresh = self.conf_thresh if conf_thresh is None else conf_thresh
        in_h, in_w = input_hw
        fmc = len(self.strides)
        all_scores, all_boxes, all_kps = [], [], []
        for idx, stride in enumerate(self.strides):
            # 输出顺序: 3 个 score, 3 个 bbox, 3 个 kps
            scores = outs[idx].reshape(-1)
            bbox_pred = outs[idx + fmc].reshape(-1, 4) * stride
            kps_pred = outs[idx + fmc * 2].reshape(-1, 10) * stride
            centers = self._anchor_centers(in_h // stride, in_w // stride, stride)
            pos = np.where(scores >= conf_thresh)[0]
            all_scores.append(scores[pos])
            all_boxes.append(distance2bbox(centers[pos], bbox_pred[pos]))
            all_kps.append(distance2kps(centers[pos], kps_pred[pos]))

        scores = np.concatenate(all_scores)
        if scores.size == 0:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        boxes = np.concatenate(all_boxes) / scale   # 映射回原图坐标
        kps = np.concatenate(all_kps) / scale
        order = scores.argsort()[::-1]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)[order]
        kps = kps[order]
        keep = nms(dets, self.nms_thresh)
        dets, kps = dets[keep], kps[keep]

        if max_num > 0 and len(dets) > max_num:
            # 优先保留"又大又靠近图像中心"的脸
            area = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
            cx, cy = img_shape[1] / 2, img_shape[0] / 2
            offset = ((dets[:, 0] + dets[:, 2]) / 2 - cx) ** 2 + ((dets[:, 1] + dets[:, 3]) / 2 - cy) ** 2
            idx = np.argsort(area - offset * 2.0)[::-1][:max_num]
            dets, kps = dets[idx], kps[idx]
        return dets, kps


class SCRFDBatch(SCRFD):
    """批量版检测器: 把 ONNX 模型转成 PyTorch 跑在 GPU/核显上, 一次处理一批同尺寸的图。

    ONNX 文件把 batch 固定成 1, 转成 PyTorch 后卷积天然支持任意 batch。
    注意 SCRFD 导出时用的是 permute(2,3,0,1): 输出按 (H, W, B, 锚点, k) 的顺序展平,
    batch 维夹在中间, 所以要先 reshape 成 (H*W, B, 2, k) 再取第 i 张图。解码逻辑与单张版完全共用。
    """

    def __init__(self, device="xpu", **kw):
        super().__init__(**kw)
        import onnx
        import torch
        from onnx2torch import convert
        self.torch = torch
        self.device = device
        self.model = convert(onnx.load(str(kw.get("model_path", DET_MODEL)))).eval().to(device)

    def detect_batch(self, imgs: list[np.ndarray], max_num: int = 0, conf_thresh=None):
        torch = self.torch
        letter = [self._letterbox(im, self.input_size) for im in imgs]
        blob = cv2.dnn.blobFromImages([d for d, _ in letter], 1.0 / 128, self.input_size,
                                      (127.5, 127.5, 127.5), swapRB=True)
        with torch.no_grad():
            outs = [o.float().cpu().numpy() for o in self.model(torch.from_numpy(blob).to(self.device))]
        b = len(imgs)
        results = []
        for i in range(b):
            per = [o.reshape(-1, b, self.num_anchors, o.shape[-1])[:, i].reshape(-1, o.shape[-1]) for o in outs]
            results.append(self.postprocess(per, blob.shape[2:], letter[i][1], imgs[i].shape, max_num, conf_thresh))
        return results
