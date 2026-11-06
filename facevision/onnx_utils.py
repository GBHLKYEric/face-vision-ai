"""ONNX Runtime 会话的统一创建方式。"""
from __future__ import annotations

import os

import onnxruntime as ort


def create_session(model_path, providers=None, threads: int | None = None) -> ort.InferenceSession:
    """创建推理会话。

    providers 默认只用 CPU; 有 GPU 版 onnxruntime 时可以传 ["CUDAExecutionProvider", ...]。
    threads 控制单个会话用几个 CPU 线程, 多个模型并行时调小可以减少抢核。
    """
    opts = ort.SessionOptions()
    # 图优化: 常量折叠、Conv+BN 融合、算子融合等, 等价于"层融合"加速
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads is None:
        threads = int(os.environ.get("FV_ORT_THREADS", "0"))  # 0 = 让 ORT 自己决定
    opts.intra_op_num_threads = threads
    # 线程干完活后默认会空转等下一个任务。流水线里几个模型轮流运行时, 每个模型都有一组线程在空转,
    # 互相抢 CPU。关掉空转; 只测单个模型的峰值速度时可以用 FV_ORT_SPIN=1 打开
    if os.environ.get("FV_ORT_SPIN", "0") != "1":
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path), sess_options=opts,
                                providers=providers or ["CPUExecutionProvider"])
