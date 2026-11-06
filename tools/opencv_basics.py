"""阶段一: OpenCV 图像处理基础 —— 读图、灰度化、模糊、边缘、直方图均衡、画框写字。

用法: python tools/opencv_basics.py [图片路径]   结果保存在 outputs/opencv_basics/
(加 --show 会弹窗显示, 服务器/Docker 里不要加)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def label(img: np.ndarray, text: str) -> np.ndarray:
    img = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=str(ROOT / "data/lfw/Colin_Powell/Colin_Powell_0001.jpg"))
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    out = ROOT / "outputs" / "opencv_basics"
    out.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)            # OpenCV 读出来是 BGR 顺序, 形状 (高, 宽, 3), uint8
    if img is None:
        sys.exit(f"读不到图片: {args.image}")
    print("形状", img.shape, "类型", img.dtype, "左上角像素(BGR)", img[0, 0])

    # 灰度化: Y = 0.299R + 0.587G + 0.114B (人眼对绿色最敏感, 权重最大)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊: 用高斯核做加权平均, 去噪
    blur = cv2.GaussianBlur(img, (9, 9), 0)
    # Canny 边缘: 梯度 + 非极大值抑制 + 双阈值
    edges = cv2.Canny(gray, 80, 160)
    # 直方图均衡: 把亮度分布拉平, 增强对比度 (暗光人脸预处理常用)
    eq = cv2.equalizeHist(gray)
    # Haar 级联人脸检测: 2001 年的经典方法 (Viola-Jones), 用来和深度学习检测器对比
    # (OpenCV 5 不再自带级联文件, 由 scripts/download_assets.py 下载到 weights/)
    xml = ROOT / "weights" / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(xml))
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) if not cascade.empty() else []
    haar = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(haar, (x, y), (x + w, y + h), (0, 255, 0), 2)

    tiles = [label(img, "original"), label(gray, "gray"), label(blur, "gaussian blur"),
             label(edges, "canny edges"), label(eq, "hist equalize"), label(haar, f"haar faces={len(boxes)}")]
    grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])])
    cv2.imwrite(str(out / "gray.jpg"), gray)
    cv2.imwrite(str(out / "basics_grid.jpg"), grid)
    print("已保存到", out)
    if args.show:
        cv2.imshow("OpenCV basics", grid)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
