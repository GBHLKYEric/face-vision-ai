# 环境验证报告

生成时间: 2026-11-06 19:00

## 软件与硬件

| 项目 | 版本 / 信息 |
|---|---|
| 系统 | Windows-11-10.0.26100-SP0 |
| CPU | Intel64 Family 6 Model 197 Stepping 2, GenuineIntel |
| Python | 3.12.14 |
| PyTorch | 2.15.0+xpu |
| PyTorch 加速设备 | Intel XPU: Intel(R) Arc(TM) 130T GPU (16GB) |
| torchvision | 0.29.2+xpu |
| ONNX Runtime | 1.31.1 ['AzureExecutionProvider', 'CPUExecutionProvider'] |
| onnx | 1.23.2 |
| OpenCV | 5.0.0 |
| MediaPipe | 1.2.0 |
| NumPy | 2.5.4 |
| SciPy | 1.18.1 |
| Gradio | 6.31.0 |
| pytorch-fid | 0.3.0 |

## 模型文件

| 模型 | 文件 | 存在 | 大小(MB) |
|---|---|---|---|
| 检测 SCRFD-10G | `weights\buffalo_l\det_10g.onnx` | ✅ | 16.9 |
| 关键点 2d106 | `weights\buffalo_l\2d106det.onnx` | ✅ | 5.0 |
| 3D 关键点 1k3d68 | `weights\buffalo_l\1k3d68.onnx` | ✅ | 143.6 |
| 识别 ArcFace-R50 | `weights\buffalo_l\w600k_r50.onnx` | ✅ | 174.4 |
| 性别年龄 | `weights\buffalo_l\genderage.onnx` | ✅ | 1.3 |
| MediaPipe 网格 | `weights\face_landmarker.task` | ✅ | 3.8 |

## 推理冒烟测试 (LFW: Colin_Powell_0001.jpg)

```json
{
  "检测到人脸数": 1,
  "全流程耗时ms": 102.7,
  "特征维度": 512,
  "性别年龄": "M/66",
  "XPU 矩阵乘 TFLOPS(fp32)": 3.44
}
```
