"""环境验证: 打印软件版本、硬件信息, 并对每个模型做一次推理冒烟测试, 生成环境报告。

用法: python tools/env_check.py   ->  docs/ENV_REPORT.md
"""
from __future__ import annotations

import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision import paths  # noqa: E402


def version(mod: str) -> str:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "ok")
    except Exception as e:  # noqa: BLE001
        return f"未安装 ({type(e).__name__})"


def main():
    import cv2
    import onnxruntime as ort
    import torch

    from facevision.utils import disable_power_throttling
    disable_power_throttling()

    info = {
        "系统": platform.platform(),
        "CPU": platform.processor(),
        "Python": sys.version.split()[0],
        "PyTorch": torch.__version__,
        "PyTorch 加速设备": ("Intel XPU: " + torch.xpu.get_device_name(0)) if torch.xpu.is_available()
        else ("CUDA: " + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "仅 CPU",
        "torchvision": version("torchvision"),
        "ONNX Runtime": ort.__version__ + " " + str(ort.get_available_providers()),
        "onnx": version("onnx"),
        "OpenCV": cv2.__version__,
        "MediaPipe": version("mediapipe"),
        "NumPy": np.__version__,
        "SciPy": version("scipy"),
        "Gradio": version("gradio"),
        "pytorch-fid": version("pytorch_fid"),
    }

    checks = []
    img = cv2.imread(str(paths.DATA / "lfw" / "Colin_Powell" / "Colin_Powell_0001.jpg"))
    if img is None:
        img = np.full((250, 250, 3), 128, np.uint8)
    from facevision.analyzer import FaceAnalyzer
    for name, f in [("检测 SCRFD-10G", paths.DET_MODEL), ("关键点 2d106", paths.LMK106_MODEL),
                    ("3D 关键点 1k3d68", paths.LMK3D68_MODEL), ("识别 ArcFace-R50", paths.REC_MODEL),
                    ("性别年龄", paths.GENDERAGE_MODEL), ("MediaPipe 网格", paths.MESH_MODEL)]:
        checks.append({"模型": name, "文件": str(f.relative_to(ROOT)), "存在": f.exists(),
                       "大小MB": round(f.stat().st_size / 1e6, 1) if f.exists() else 0})
    fa = FaceAnalyzer(modules=("landmark_106", "landmark_3d_68", "recognition", "genderage"))
    fa.analyze(img)
    t = time.perf_counter()
    faces = fa.analyze(img)
    ms = (time.perf_counter() - t) * 1000
    smoke = {"检测到人脸数": len(faces), "全流程耗时ms": round(ms, 1),
             "特征维度": int(faces[0].embedding.shape[0]) if faces else 0,
             "性别年龄": f"{faces[0].gender}/{faces[0].age}" if faces else "-"}

    if torch.xpu.is_available():
        x = torch.randn(2048, 2048, device="xpu")
        x @ x                      # 预热: 第一次调用要编译内核, 不算进测速
        torch.xpu.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            x @ x
        torch.xpu.synchronize()
        smoke["XPU 矩阵乘 TFLOPS(fp32)"] = round(10 * 2 * 2048 ** 3 / (time.perf_counter() - t) / 1e12, 2)

    lines = ["# 环境验证报告", "", f"生成时间: {time.strftime('%Y-%m-%d %H:%M')}", "", "## 软件与硬件", "",
             "| 项目 | 版本 / 信息 |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in info.items()]
    lines += ["", "## 模型文件", "", "| 模型 | 文件 | 存在 | 大小(MB) |", "|---|---|---|---|"]
    lines += [f"| {c['模型']} | `{c['文件']}` | {'✅' if c['存在'] else '❌'} | {c['大小MB']} |" for c in checks]
    lines += ["", "## 推理冒烟测试 (LFW: Colin_Powell_0001.jpg)", "", "```json",
              json.dumps(smoke, ensure_ascii=False, indent=2), "```", ""]
    out = ROOT / "docs" / "ENV_REPORT.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
