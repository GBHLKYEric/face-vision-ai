"""模型部署优化: BN 融合、ONNX 导出、INT8 量化、剪枝、测速。

推理引擎 (如 ByteNN、TensorRT、MNN) 加速的几板斧, 这里各做一个能跑通的最小实现:
  1. 层融合: Conv + BN 合并成一个 Conv —— 推理时 BN 只是逐通道的仿射变换, 可以直接并进卷积权重
  2. 量化: FP32 -> INT8, 模型小 4 倍, CPU 上整数运算更快 (精度损失通常 <0.1%)
  3. 剪枝: 把绝对值小的权重置 0 (说明网络有大量冗余)
  4. 格式标准化: 导出 ONNX, 一份模型跑在 Windows/Linux/手机/浏览器上
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ---------------- 1. Conv + BN 融合 ----------------
def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """BN(Conv(x)) = γ·(W·x + b - μ)/√(σ²+ε) + β
                  = (γ/√(σ²+ε))·W · x + (γ·(b-μ)/√(σ²+ε) + β)
    所以新权重 W' = W·k, 新偏置 b' = (b-μ)·k + β, 其中 k = γ/√(σ²+ε) (逐输出通道)。
    """
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
                      conv.padding, conv.dilation, conv.groups, bias=True).to(conv.weight.device)
    k = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    fused.weight.data = conv.weight.data * k.reshape(-1, 1, 1, 1)
    b = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
    fused.bias.data = (b - bn.running_mean) * k + bn.bias
    return fused


def fuse_model(model: nn.Module) -> nn.Module:
    """把模型里所有"Conv 紧跟 BN"的结构融合 (IResNet 的 conv1/bn2、conv2/bn3、downsample 以及 MobileFaceNet 的 ConvBlock)。"""
    model = copy.deepcopy(model).eval()
    for module in model.modules():
        # IResNet 残差块
        if hasattr(module, "conv1") and hasattr(module, "bn2") and hasattr(module, "conv2") and hasattr(module, "bn3"):
            module.conv1 = fuse_conv_bn(module.conv1, module.bn2)
            module.bn2 = nn.Identity()
            module.conv2 = fuse_conv_bn(module.conv2, module.bn3)
            module.bn3 = nn.Identity()
        # 下采样分支 / MobileFaceNet 的 ConvBlock
        if isinstance(module, nn.Sequential) and len(module) == 2 and isinstance(module[0], nn.Conv2d) \
                and isinstance(module[1], nn.BatchNorm2d):
            module[0] = fuse_conv_bn(module[0], module[1])
            module[1] = nn.Identity()
        if hasattr(module, "conv") and hasattr(module, "bn") and isinstance(getattr(module, "bn"), nn.BatchNorm2d):
            module.conv = fuse_conv_bn(module.conv, module.bn)
            module.bn = nn.Identity()
    # IResNet 的 stem: conv1 -> bn1 -> prelu
    if hasattr(model, "conv1") and isinstance(getattr(model, "bn1", None), nn.BatchNorm2d):
        model.conv1 = fuse_conv_bn(model.conv1, model.bn1)
        model.bn1 = nn.Identity()
    return model


# ---------------- 2. ONNX 导出 ----------------
def export_onnx(model: nn.Module, path: Path, input_size: int = 112) -> Path:
    model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, input_size, input_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["embedding"],
                      dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17,
                      dynamo=False)
    return path


# ---------------- 3. 量化 ----------------
def quantize_dynamic_torch(model: nn.Module) -> nn.Module:
    """PyTorch 动态量化: 权重离线转 INT8, 激活值在运行时动态量化。只支持 Linear/LSTM 等层,
    对以卷积为主的人脸网络只会量化最后的全连接层 —— 这正是需要静态量化的原因。"""
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.Linear}, dtype=torch.qint8)


class _FaceCalibReader:
    """ONNX Runtime 静态量化的校准数据: 喂几百张真实人脸, 统计每层激活值的范围。"""

    def __init__(self, faces_bgr: np.ndarray, input_name: str, batch: int = 32):
        self.faces, self.name, self.batch, self.i = faces_bgr, input_name, batch, 0

    def get_next(self):
        import cv2
        if self.i >= len(self.faces):
            return None
        chunk = list(self.faces[self.i:self.i + self.batch])
        self.i += self.batch
        blob = cv2.dnn.blobFromImages(chunk, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
        return {self.name: blob}

    def rewind(self):
        self.i = 0


def quantize_static_onnx(fp32_path: Path, int8_path: Path, calib_faces: np.ndarray) -> Path:
    """静态 INT8 量化 (QDQ 格式, 权重逐通道): 用校准集确定激活值的缩放系数 scale 和零点 zero_point,
    q = round(x / scale) + zero_point, 反量化 x ≈ (q - zero_point) · scale。"""
    import onnx
    import onnxruntime as ort
    from onnx import version_converter
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    # 逐通道量化要用 DequantizeLinear 的 axis 属性, opset 13 才支持; 老模型要先升级算子集
    src = Path(fp32_path)
    model = onnx.load(str(src))
    opset = next(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    upgraded = int8_path.with_name(int8_path.stem + "_opset13.onnx")
    if opset < 13:
        onnx.save(version_converter.convert_version(model, 13), str(upgraded))
        src = upgraded

    pre = int8_path.with_name(int8_path.stem + "_pre.onnx")
    quant_pre_process(str(src), str(pre), skip_symbolic_shape=True)
    name = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"]).get_inputs()[0].name
    quantize_static(str(pre), str(int8_path), _FaceCalibReader(calib_faces, name),
                    quant_format=QuantFormat.QDQ, per_channel=True, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, calibrate_method=CalibrationMethod.Percentile,
                    extra_options={"CalibPercentile": 99.99})
    pre.unlink(missing_ok=True)
    upgraded.unlink(missing_ok=True)
    return int8_path


# ---------------- 4. 剪枝 ----------------
def magnitude_prune(model: nn.Module, amount: float) -> nn.Module:
    """全局非结构化剪枝: 所有卷积/全连接权重里, 绝对值最小的 amount 比例置 0。"""
    import torch.nn.utils.prune as prune
    model = copy.deepcopy(model).cpu().eval()
    params = [(m, "weight") for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for m, n in params:
        prune.remove(m, n)
    return model


# ---------------- 测速 / 大小 ----------------
def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / 1e6


def torch_size_mb(model: nn.Module) -> float:
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


def bench(fn, warmup: int = 5, runs: int = 30) -> float:
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t) / runs * 1000
