"""特效模块 CPU 性能测试: 每个阶段的耗时 + 端到端 FPS, 以及 One Euro 去抖效果。

用法:
    python tools/benchmark_effects.py                        # 默认 640x480 和 1280x720, 各 60 帧
    python tools/benchmark_effects.py --frames 120 --sizes 640x480

输入是 assets/demo/demo_input.mp4 的前 N 帧 (960x540), 先裁成目标宽高比再缩放到目标分辨率
(1280x720 是从 960x540 放大来的, 人脸像素更多, 主要用于看耗时随分辨率的变化)。
结果写到 outputs/effects/perf.json, 并在终端打印 markdown 表格。
Windows 上默认先关闭本进程的 EcoQoS 后台节流 (见 disable_power_throttling), 加 --allow-throttling 可对比。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

import cv2
import numpy as np

from facevision.effects import EffectPipeline
from facevision.effects.filters import FaceSmoother
from facevision.effects.pipeline import disable_power_throttling
from facevision.paths import ASSETS, OUTPUTS

SINGLE_EFFECTS = ["sunglasses", "hat", "cat_ears", "mustache", "smooth", "whiten", "slim", "big_eyes",
                  "lipstick", "blush", "eyeliner", "mesh", "avatar"]
PRESETS = {
    "landmarks only": [],
    "stickers (sunglasses+hat)": ["sunglasses", "hat"],
    "beauty (smooth+whiten+slim+big_eyes)": ["smooth", "whiten", "slim", "big_eyes"],
    "makeup (lipstick+blush+eyeliner)": ["lipstick", "blush", "eyeliner"],
    "beauty + makeup (7 effects)": ["smooth", "whiten", "slim", "big_eyes", "lipstick", "blush", "eyeliner"],
    "avatar": ["avatar"],
}


def load_frames(path, n: int, size: tuple[int, int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    w, h = size
    while len(frames) < n:
        ok, f = cap.read()
        if not ok:
            break
        # 先按目标宽高比居中裁剪 (偏右一点, 人脸在画面中右侧), 再缩放
        fh, fw = f.shape[:2]
        if fw / fh > w / h:
            cw = int(fh * w / h)
            x0 = min(fw - cw, max(0, int(fw * 0.62 - cw / 2)))
            f = f[:, x0:x0 + cw]
        frames.append(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA if w < fw else cv2.INTER_LINEAR))
    cap.release()
    return frames


def stats(xs: list[float]) -> dict:
    xs = sorted(xs)
    return {"mean": round(statistics.fmean(xs), 2), "median": round(statistics.median(xs), 2),
            "p90": round(xs[int(0.9 * (len(xs) - 1))], 2)}


def bench_size(pipe: EffectPipeline, frames: list[np.ndarray], warmup: int = 5) -> dict:
    """同一个 pipe (同一个 MediaPipe 实例) 测完所有项目, 时间戳一直递增。"""
    fps_in = 30.0
    clock = [pipe.mesh._last_ts + 1000]                      # VIDEO 模式时间戳必须一直递增

    def ts(i: int) -> int:
        return clock[0] + int(i * 1000 / fps_in)

    pipe.smoother = FaceSmoother()
    # 1) 关键点 (MediaPipe) + One Euro: 每帧检测一次, 把结果缓存起来给后面的特效用
    lm, oe, faces_all = [], [], []
    for i, f in enumerate(frames):
        faces = pipe.detect(f, ts(i))
        if i >= warmup:
            lm.append(pipe.timings["landmarks"])
            oe.append(pipe.timings.get("one_euro", 0.0))
        faces_all.append(faces)
    detected = sum(1 for fs in faces_all if fs)
    res = {"frames": len(frames) - warmup, "face_detected_frames": detected,
           "stages_ms": {"landmarks (MediaPipe)": stats(lm), "one_euro": stats(oe)}}
    # 2) 每个特效单独开, 统计它自己的耗时 (单独开时皮肤掩码在特效内部第一次用到时计算, 已包含在内)
    skin = []
    for name in SINGLE_EFFECTS:
        pipe.effects = {name}
        eff = []
        for i, (f, faces) in enumerate(zip(frames, faces_all)):
            pipe.apply(f, faces)
            if i >= warmup and faces:
                eff.append(pipe.timings.get(name, 0.0))
                if name == "smooth":
                    skin.append(pipe.timings.get("skin_mask", 0.0))
        res["stages_ms"][name] = stats(eff) if eff else None
    res["stages_ms"]["  (skin_mask, shared)"] = stats(skin) if skin else None
    # 3) 端到端 (检测 + 特效, 不含解码/显示/编码): 重新跑一遍 detect + apply
    res["end_to_end"] = {}
    for title, effects in PRESETS.items():
        pipe.effects = set(effects)
        pipe.smoother = FaceSmoother()                     # 每组从头开始平滑
        clock[0] = pipe.mesh._last_ts + 1000                # 视频 "跳回开头", 跟踪会重新检测一次
        times = []
        for i, f in enumerate(frames):
            t0 = time.perf_counter()
            pipe.process(f, ts(i))
            if i >= warmup:
                times.append((time.perf_counter() - t0) * 1000)
        s = stats(times)
        s["fps"] = round(1000 / s["median"], 1)     # 用中位数: CPU 被别的任务抢占时的尖峰不影响结论
        res["end_to_end"][title] = s
    return res


def jitter_analysis(mesh, path, n: int = 300) -> dict:
    """One Euro 的去抖效果: 用二阶差分 |x_{t+1} - 2x_t + x_{t-1}| 衡量抖动 (匀速运动时为 0),
    用 |平滑 - 原始| 的平均值衡量滞后/偏差; 都以两眼距离 (IOD) 为单位。"""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sm = FaceSmoother()
    t0 = mesh._last_ts + 1000
    raw, smooth, iods = [], [], []
    i = 0
    while i < n:
        ok, f = cap.read()
        if not ok:
            break
        faces = mesh.detect(f, t0 + int(i * 1000 / fps))
        if faces:
            raw.append(faces[0].xy.copy())
            smooth.append(sm(faces, i / fps)[0].xy.copy())
            iods.append(faces[0].eye_distance())
        i += 1
    cap.release()
    raw, smooth = np.array(raw), np.array(smooth)
    iod = float(np.mean(iods))

    def jit(x):
        return float(np.linalg.norm(x[2:] - 2 * x[1:-1] + x[:-2], axis=-1).mean() / iod)

    return {"frames": len(raw), "mean_iod_px": round(iod, 1),
            "jitter_raw_iod": round(jit(raw), 5), "jitter_one_euro_iod": round(jit(smooth), 5),
            "jitter_reduction": round(1 - jit(smooth) / jit(raw), 3),
            "mean_offset_vs_raw_iod": round(float(np.linalg.norm(smooth - raw, axis=-1).mean() / iod), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ASSETS / "demo" / "demo_input.mp4"))
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--sizes", default="640x480,1280x720")
    ap.add_argument("--out", default=str(OUTPUTS / "effects" / "perf.json"))
    ap.add_argument("--allow-throttling", action="store_true",
                    help="不关闭 Windows 的后台进程节流 (数字会差 3 倍左右, 用来对比)")
    args = ap.parse_args()
    throttling_disabled = False if args.allow_throttling else disable_power_throttling()

    info = {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "python": platform.python_version(), "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "windows_power_throttling_disabled_for_this_process": throttling_disabled,
            "note": "CPU only; numbers measured while another long CPU job was running may be pessimistic"}
    try:
        import mediapipe
        info["mediapipe"] = mediapipe.__version__
    except Exception:
        pass
    try:
        import psutil
        info["cpu_load_percent_before"] = psutil.cpu_percent(interval=1.0)
    except Exception:
        pass

    report = {"system": info, "results": {}}
    pipe = EffectPipeline([], mode="video", num_faces=1, smoothing=True)   # 全程只建这一个
    for s in args.sizes.split(","):
        w, h = map(int, s.lower().split("x"))
        frames = load_frames(args.input, args.frames + 5, (w, h))
        print(f"[{s}] {len(frames)} frames ...", flush=True)
        report["results"][s] = bench_size(pipe, frames)
    print("[jitter] One Euro analysis ...", flush=True)
    report["one_euro_jitter"] = jitter_analysis(pipe.mesh, args.input)
    pipe.close()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- markdown 表格 ----
    sizes = list(report["results"])
    print(f"\nCPU: {info['processor']} ({info['cpu_count']} logical cores), OpenCV {info['opencv']}, "
          f"MediaPipe {info.get('mediapipe', '?')}\n")
    print("| stage (ms, median / p90) | " + " | ".join(sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for k in report["results"][sizes[0]]["stages_ms"]:
        cells = []
        for s in sizes:
            v = report["results"][s]["stages_ms"][k]
            cells.append("-" if v is None else f"{v['median']:.1f} / {v['p90']:.1f}")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\n| end-to-end (detect + effects) | " + " | ".join(f"{s} median ms / FPS" for s in sizes) + " |")
    print("|---|" + "---|" * len(sizes))
    for k in report["results"][sizes[0]]["end_to_end"]:
        cells = [f"{report['results'][s]['end_to_end'][k]['median']:.1f} / {report['results'][s]['end_to_end'][k]['fps']:.1f}"
                 for s in sizes]
        print(f"| {k} | " + " | ".join(cells) + " |")
    j = report["one_euro_jitter"]
    print(f"\nOne Euro: jitter {j['jitter_raw_iod']:.4f} -> {j['jitter_one_euro_iod']:.4f} IOD/frame "
          f"({100 * j['jitter_reduction']:.0f}% less), mean offset vs raw {j['mean_offset_vs_raw_iod']:.4f} IOD "
          f"(IOD ≈ {j['mean_iod_px']} px)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
