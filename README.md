# FaceVision · 人脸识别与特效

一个完整、可复现的人脸视觉项目：**检测 → 关键点 → 对齐 → 识别 → 实时特效 / 3D 重建 / 属性编辑**，实现了《字节跳动智能视觉 AI：人脸识别与特效技术》项目规划中四个阶段的全部任务，并附带一份从零讲起的中文教学讲义。

> 全部实验在一台**没有独立显卡**的笔记本（Intel Core Ultra 5 225H + Arc 130T 核显，32 GB 内存）上完成；训练用 PyTorch 的 Intel XPU 后端，推理用 ONNX Runtime CPU。

## 核心结果

<!-- RESULTS:START -->
| 任务 | 指标 | 结果 |
|---|---|---|
| 人脸识别（预训练 ArcFace-R50） | LFW 10 折准确率 | 99.850% |
| 人脸识别（自训练 IResNet18 + 蒸馏） | LFW 10 折准确率 | 98.867% |
| 人脸验证 | TAR@FAR=10⁻³ / AUC | 99.73% / 0.99944 |
| 人脸检测（SCRFD-10G，单尺度 640） | WIDER FACE Easy / Medium / Hard AP | 94.85 / 93.46 / 81.28 |
| 人脸检测（多尺度 640+1280） | WIDER FACE Easy / Medium / Hard AP | 95.18 / 94.21 / 88.95 |
| 模型压缩（INT8 静态量化） | 大小 / 速度 / LFW | 96.12→24.21 MB，12.51→7.5 ms，98.83% |
| 实时特效（CPU 640×480） | 美颜美妆全开 / 仅关键点 | 70 FPS / 144 FPS |
| 属性编辑（StarGAN） | FID（2000 张 CelebA 测试图） | 发色 25.47～33.9，性别 28.74，年龄 29.42 |
| 3D 重建（3DDFA-V2 移植） | 与官方实现逐点对比 | 最大误差 0（完全一致） |

LFW 明细（6000 对，官方 10 折协议，从原始图片开始的完整流程）：

| 模型 | 设置 | LFW 准确率 (%) | AUC | EER (%) | TAR@FAR=10⁻³ (%) | 错误对数 |
|---|---|---|---|---|---|---|
| ArcFace-R50（预训练） | 不补边，翻转测试 | 99.833 ± 0.197 | 0.99939 | 0.300 | 99.700 | 9 |
| ArcFace-R50（预训练） | 补边，无翻转 | 99.833 ± 0.197 | 0.99946 | 0.267 | 99.733 | 8 |
| ArcFace-R50（预训练） | 补边 + 翻转（最终） | 99.850 ± 0.203 | 0.99944 | 0.267 | 99.733 | 8 |
| IResNet18 自训练（ArcFace + 蒸馏） | 补边 + 翻转 | 98.867 ± 0.657 | 0.99846 | 1.267 | 96.300 | 67 |
| IResNet18 自训练 INT8 量化 | 补边 + 翻转 | 98.833 ± 0.596 | 0.99842 | 1.333 | 96.267 | 68 |
<!-- RESULTS:END -->

## 功能一览

| 模块 | 方法 | 代码 |
|---|---|---|
| 人脸检测 | SCRFD-10G（RetinaFace 改进版），手写 anchor 解码 + NMS，批量 GPU 版 | `facevision/detector.py` |
| 关键点 | 5 点 / 106 点 2D / 68 点 3D / MediaPipe 478 点网格 | `facevision/landmark.py` `facevision/mesh.py` |
| 对齐 | 手写 Umeyama 最小二乘相似变换 → 112×112 | `facevision/align.py` |
| 识别 | ArcFace-R50（推理）；IResNet / MobileFaceNet + Softmax / SphereFace / CosFace / ArcFace / CurricularFace / Triplet + 知识蒸馏（训练） | `facevision/recognizer.py` `facevision/training/` |
| 评测 | LFW 官方 10 折、ROC/AUC/EER/TAR@FAR；WIDER FACE 官方 AP（numpy 移植） | `facevision/verification.py` `facevision/widerface_eval.py` |
| 部署 | BN 融合、PyTorch 动态量化、ONNX 导出、ONNX Runtime 静态 INT8、剪枝 | `facevision/deploy.py` `tools/deploy_benchmark.py` |
| 实时特效 | AR 贴纸（墨镜/帽子/猫耳）、磨皮、美白、瘦脸、大眼、口红、腮红、表情驱动卡通头像、One Euro 去抖 | `facevision/effects/` |
| 3D 重建 | 3DDFA-V2 移植（3DMM 62 参数）+ 纯 numpy 渲染器、多视角、.obj 导出 | `facevision/face3d/` |
| 属性编辑 | StarGAN（发色 / 性别 / 年龄），任意照片贴回，FID 评测，完整训练代码 | `facevision/gan/` |
| 系统集成 | Gradio 网页（7 个标签页，含摄像头实时特效）+ OpenCV 实时演示 | `app/gradio_app.py` `tools/realtime_demo.py` |

## 快速开始

```bash
# 1. 环境 (Python 3.12); PyTorch 按硬件选择 (见 requirements.txt 顶部注释)
python -m venv .venv && .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu   # Intel GPU; 或 /cpu, /cu128
pip install -r requirements.txt

# 2. 下载模型与评测数据 (约 1 GB); CelebA 训练集另需 11.7 GB
python scripts/download_assets.py
python scripts/download_celeba.py        # 只有训练/StarGAN 评测需要

# 3. 环境验证 + 网页演示
python tools/env_check.py
python app/gradio_app.py                 # 打开 http://127.0.0.1:7860
```

## 复现实验

```bash
python tools/eval_lfw.py                              # LFW 10 折 (预训练 ArcFace-R50)
python tools/eval_widerface.py                        # WIDER FACE Easy/Medium/Hard AP
python tools/prepare_celeba.py --device xpu           # CelebA 检测 + 对齐 (20 万张)
python tools/extract_teacher.py                       # 老师特征 (蒸馏用)
python tools/train_recognition.py --backbone r18 --head arcface --kd-weight 50 --epochs 10 --lr 0.05 --name r18_arcface_kd
python tools/eval_lfw.py --torch-ckpt runs/r18_arcface_kd/final.pt --tag r18_arcface_kd
python tools/ablation_losses.py --device xpu --epochs 5
python tools/deploy_benchmark.py --ckpt runs/r18_arcface_kd/final.pt
python tools/reconstruct_3d.py --input data/lfw/Colin_Powell/Colin_Powell_0001.jpg
python tools/eval_stargan.py
python tools/realtime_demo.py --camera 0              # 摄像头实时特效, 按键切换
python -m pytest tests -q                             # 单元测试
```

## 目录结构

```
facevision/            算法库
  detector.py          SCRFD 检测 (前处理 / anchor 解码 / NMS / 批量 GPU 版)
  align.py landmark.py 对齐与关键点
  recognizer.py        ArcFace 特征提取
  analyzer.py          检测+关键点+识别+属性 一体化流水线
  verification.py      1:1 验证指标 (10 折 / ROC / TAR@FAR)
  widerface_eval.py    WIDER FACE 官方 AP
  gallery.py           1:N 人脸库检索
  deploy.py            BN 融合 / 量化 / 剪枝 / ONNX
  training/            骨干网络、损失函数、数据读取
  effects/ mesh.py     实时特效
  face3d/              3D 重建与渲染
  gan/                 StarGAN
tools/                 评测、训练、演示脚本
app/gradio_app.py      统一网页界面
docs/                  报告、环境报告、教学讲义 (docs/teaching)
outputs/               评测结果 JSON、曲线、效果图、演示视频
tests/                 单元测试
```

## 文档

- [项目总结报告](docs/REPORT.md)：每个阶段的方法、实验、结果与分析
- [环境验证报告](docs/ENV_REPORT.md)
- [教学讲义](docs/teaching/index.html)：12 个专题，从原理到逐行代码，含例题与必背清单（克隆后用浏览器打开）
- [项目汇报 PPT](docs/FaceVision_汇报.pptx)

## 致谢与许可

本仓库代码以 MIT 许可发布。使用的第三方模型与数据各有其许可，仅限研究用途：
InsightFace buffalo_l 模型（非商业研究）、3DDFA-V2（MIT）、StarGAN 官方权重（MIT）、MediaPipe Face Landmarker（Apache-2.0）、
LFW / CelebA / WIDER FACE 数据集（学术非商业）。演示视频来源见 `assets/demo/SOURCE.md`。
