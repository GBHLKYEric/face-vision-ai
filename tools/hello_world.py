"""阶段一: 第一个程序。在 Docker 里运行: docker build -t face-vision-ai . && docker run --rm face-vision-ai"""
import platform

import cv2
import numpy as np

if __name__ == "__main__":
    print("Hello, Face Vision AI!")
    print(f"Python {platform.python_version()} | OpenCV {cv2.__version__} | NumPy {np.__version__}")
