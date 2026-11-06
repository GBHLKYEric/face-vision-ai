"""实时人脸特效演示: 摄像头 / 视频文件 -> 贴纸、美颜、美妆、卡通头像。

用法 (在项目根目录, PYTHONPATH 指向项目根):
    python tools/realtime_demo.py --camera 0
    python tools/realtime_demo.py --input assets/demo/demo_input.mp4 --effects smooth,whiten,lipstick,blush
    python tools/realtime_demo.py --input assets/demo/demo_input.mp4 --output outputs/effects/demo_stickers.mp4 \
        --effects "sunglasses,hat;cat_ears;mustache,sunglasses" --cycle 5 --no-window
    python tools/realtime_demo.py --showcase outputs/effects        # 用 LFW 照片生成效果对比图

--effects 里用逗号分隔同时开启的特效, 用分号分隔多组; 配合 --cycle N 每 N 秒切换到下一组。

按键 (窗口模式):
    1 墨镜  2 礼帽  3 猫耳  4 胡子  5 磨皮  6 美白  7 瘦脸  8 大眼  9 口红  0 腮红
    e 眼线  m 网格  a 卡通头像  c 原图/效果对比  p 贴纸 2D/3D 模式  f 开关 One Euro 平滑
    s 截图  空格 暂停  q / Esc 退出
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from facevision.effects import EFFECT_ORDER, EffectPipeline, put_label
from facevision.effects.avatar import avatar_panel
from facevision.effects.filters import FaceSmoother
from facevision.effects.pipeline import disable_power_throttling
from facevision.paths import DATA, OUTPUTS

KEYMAP = {ord("1"): "sunglasses", ord("2"): "hat", ord("3"): "cat_ears", ord("4"): "mustache",
          ord("5"): "smooth", ord("6"): "whiten", ord("7"): "slim", ord("8"): "big_eyes",
          ord("9"): "lipstick", ord("0"): "blush", ord("e"): "eyeliner", ord("m"): "mesh",
          ord("a"): "avatar"}


def parse_groups(text: str) -> list[list[str]]:
    groups = [[e.strip() for e in g.split(",") if e.strip()] for g in text.split(";")]
    for g in groups:
        for e in g:
            if e not in EFFECT_ORDER:
                sys.exit(f"未知特效 '{e}', 可选: {', '.join(EFFECT_ORDER)}")
    return groups or [[]]


def fit_to(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把图像等比缩放并补黑边到固定大小 (VideoWriter 要求每帧尺寸一致)。"""
    w, h = size
    if img.shape[1] == w and img.shape[0] == h:
        return img
    s = min(w / img.shape[1], h / img.shape[0])
    r = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((h, w, 3), np.uint8)
    y0, x0 = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    canvas[y0:y0 + r.shape[0], x0:x0 + r.shape[1]] = r
    return canvas


def draw_hud(img: np.ndarray, fps: float, ms: float, effects: list[str], n_faces: int, extra: str = ""):
    """FPS = 1000 / (检测+特效 的平均耗时), 不含读视频、显示和写文件的时间。"""
    put_label(img, f"FPS {fps:5.1f}  ({ms:4.1f} ms/frame)  faces: {n_faces} {extra}", (10, 26), 0.6)
    put_label(img, " ".join(e for e in EFFECT_ORDER if e in effects) or "(no effect)", (10, 50), 0.5)
    if n_faces == 0:
        put_label(img, "no face", (10, 78), 0.7)


def open_source(args) -> tuple[cv2.VideoCapture, float, bool]:
    if args.input:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            sys.exit(f"打不开视频 {args.input}")
        return cap, cap.get(cv2.CAP_PROP_FPS) or 30.0, False
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY  # Windows 下 DirectShow 打开更快
    cap = cv2.VideoCapture(args.camera, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"打不开摄像头 {args.camera}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap, 30.0, True


def run_video(args):
    groups = parse_groups(args.effects)
    pipe = EffectPipeline(groups[0], mode="video", num_faces=args.num_faces, smoothing=not args.no_smooth,
                          params={"sticker": {"mode": args.sticker_mode}})
    cap, fps_in, live = open_source(args)
    writer, out_size = None, None
    compare = args.compare
    ms_ema, idx, paused, t_start = 0.0, 0, False, time.perf_counter()
    show = not args.no_window
    frame = None
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            if args.crop:
                x0, y0, x1, y1 = args.crop
                frame = frame[y0:y1, x0:x1].copy()
            if args.scale_width and frame.shape[1] != args.scale_width:
                s = args.scale_width / frame.shape[1]
                frame = cv2.resize(frame, (args.scale_width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
            if live:
                frame = cv2.flip(frame, 1)              # 摄像头画面做镜像, 像照镜子一样自然
            t_sec = (time.perf_counter() - t_start) if live else idx / fps_in
            if args.cycle > 0 and len(groups) > 1:      # 按时间轮换特效组
                pipe.effects = set(groups[int(t_sec // args.cycle) % len(groups)])
            t0 = time.perf_counter()
            out = pipe.process(frame, int(t_sec * 1000))
            dt_ms = (time.perf_counter() - t0) * 1000
            ms_ema = dt_ms if ms_ema == 0 else 0.9 * ms_ema + 0.1 * dt_ms   # 滑动平均, 数字不乱跳
            if not args.no_hud:
                draw_hud(out, 1000 / ms_ema, ms_ema, sorted(pipe.effects), len(pipe.faces),
                         "" if pipe.smoother else "(raw)")
            if compare:                                  # 左: 原图, 右: 效果
                left = frame.copy()
                put_label(left, "original", (10, left.shape[0] - 14), 0.7)
                if out.shape[1] > frame.shape[1]:        # avatar 开着时, 只对比左边的画面部分
                    out = out[:, :frame.shape[1]]
                out = np.hstack([left, out])
            if args.output:
                if writer is None:
                    out_size = (out.shape[1], out.shape[0])
                    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, out_size)
                writer.write(fit_to(out, out_size))
            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break
        if show:
            cv2.imshow("FaceVision effects", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in KEYMAP:
                pipe.toggle(KEYMAP[key])
            elif key == ord("c"):
                compare = not compare
            elif key == ord("p"):
                pipe.params["sticker"]["mode"] = "2d" if pipe.params["sticker"]["mode"] == "3d" else "3d"
            elif key == ord("f"):
                pipe.smoother = None if pipe.smoother else FaceSmoother()
            elif key == ord("s"):
                p = OUTPUTS / "effects" / f"snapshot_{int(time.time())}.jpg"
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), out)
                print("saved", p)
            elif key == ord(" "):
                paused = not paused
    cap.release()
    if writer is not None:
        writer.release()
        print(f"wrote {args.output}  ({idx} frames)")
    if show:
        cv2.destroyAllWindows()
    pipe.close()


# ---------------------------------------------------------------------------
# 静态效果对比图 (LFW 照片)
# ---------------------------------------------------------------------------
SHOWCASE_COLUMNS = [
    ("original", []), ("sunglasses", ["sunglasses"]), ("hat", ["hat"]), ("cat_ears", ["cat_ears"]),
    ("mustache", ["mustache"]), ("smooth", ["smooth"]), ("whiten", ["whiten"]),
    ("slim+big_eyes", ["slim", "big_eyes"]), ("lipstick", ["lipstick"]), ("blush+eyeliner", ["blush", "eyeliner"]),
    ("beauty all", ["smooth", "whiten", "slim", "big_eyes", "lipstick", "blush", "eyeliner"]),
    ("mesh", ["mesh"]), ("avatar", ["avatar"]),
]
SHOWCASE_FACES = ["Angelina_Jolie", "Colin_Powell", "Serena_Williams", "Tony_Blair"]


def lfw(name: str, scale: int) -> np.ndarray:
    img = cv2.imread(str(DATA / "lfw" / name / f"{name}_0001.jpg"))
    if img is None:
        sys.exit(f"找不到 LFW 图片 {name}")
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def run_showcase(out_dir: Path, tile: int = 300):
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = EffectPipeline([], mode="image", num_faces=3, smoothing=False)

    # 1) 总览大图: 每行一张脸, 每列一个特效 (照片先放大 2 倍, 让特效看得清)
    rows = []
    for name in SHOWCASE_FACES:
        img = lfw(name, 2)
        faces = pipe.detect(img)
        tiles = []
        for title, effects in SHOWCASE_COLUMNS:
            if title == "avatar":
                f = faces[0] if faces else None
                if len(faces) > 1:   # 多张脸时取最大的那张驱动头像
                    f = max(faces, key=lambda x: x.eye_distance())
                res = avatar_panel(pipe.avatar, f, img.shape[0])
            else:
                pipe.effects = set(effects)
                res = pipe.apply(img, faces)
            res = cv2.resize(res, (tile, tile), interpolation=cv2.INTER_AREA)
            tiles.append(put_label(res, title, (6, tile - 10), 0.55))
        rows.append(np.hstack(tiles))
    grid = np.vstack(rows)
    cv2.imwrite(str(out_dir / "effects_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])

    # 2) 美颜/美妆特写: 3 倍放大后裁出人脸
    rows = []
    cols = [("original", []), ("smooth", ["smooth"]), ("whiten", ["whiten"]), ("slim+big_eyes", ["slim", "big_eyes"]),
            ("lipstick+blush+eyeliner", ["lipstick", "blush", "eyeliner"]),
            ("all", ["smooth", "whiten", "slim", "big_eyes", "lipstick", "blush", "eyeliner"])]
    for name in SHOWCASE_FACES[:3]:
        img = lfw(name, 3)
        faces = pipe.detect(img)
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        pad = int(0.2 * (x2 - x1))
        cx, cy, half = (x1 + x2) // 2, (y1 + y2) // 2, (max(x2 - x1, y2 - y1)) // 2 + pad
        crop = lambda im: im[max(0, cy - half):cy + half, max(0, cx - half):cx + half]  # noqa: E731
        tiles = []
        for title, effects in cols:
            pipe.effects = set(effects)
            res = cv2.resize(crop(pipe.apply(img, faces)), (360, 360), interpolation=cv2.INTER_AREA)
            tiles.append(put_label(res, title, (6, 350), 0.55))
        rows.append(np.hstack(tiles))
    cv2.imwrite(str(out_dir / "beauty_closeup.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3) 贴纸跟随 roll / 缩放 / 2D vs 3D: 把同一张照片旋转、缩放后再贴
    img0 = lfw("Colin_Powell", 2)
    h, w = img0.shape[:2]
    rows = []
    for mode in ("2d", "3d"):
        pipe.params["sticker"]["mode"] = mode
        pipe.effects = {"sunglasses", "hat"}
        tiles = []
        for ang, sc in ((-25, 1.0), (-10, 0.75), (0, 1.0), (15, 0.6), (30, 0.9)):
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            img = cv2.warpAffine(img0, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            res = pipe.apply(img, pipe.detect(img))
            res = cv2.resize(res, (tile, tile), interpolation=cv2.INTER_AREA)
            tiles.append(put_label(res, f"{mode} rot {ang} scale {sc}", (6, tile - 10), 0.5))
        rows.append(np.hstack(tiles))
    # 第三行: 转头 (yaw) 的真人照片 + 多张脸; 2D 模式贴纸不会随转头变窄, 3D 模式会
    tiles = []
    for name, mode in (("Tony_Blair", "2d"), ("Tony_Blair", "3d"), ("Hugo_Chavez", "2d"),
                       ("Hugo_Chavez", "3d"), ("Angelina_Jolie", "3d")):
        img = lfw(name, 2)
        faces = pipe.detect(img)
        pipe.params["sticker"]["mode"] = mode
        pipe.effects = {"sunglasses", "cat_ears"}
        res = cv2.resize(pipe.apply(img, faces), (tile, tile), interpolation=cv2.INTER_AREA)
        yaw = max(faces, key=lambda f: f.eye_distance()).yaw if faces else 0.0
        tiles.append(put_label(res, f"{mode} yaw {yaw:.0f}", (6, tile - 10), 0.5))
    rows.append(np.hstack(tiles))
    pipe.params["sticker"]["mode"] = "3d"
    cv2.imwrite(str(out_dir / "stickers_pose.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    pipe.close()
    print("wrote", out_dir / "effects_grid.jpg", out_dir / "beauty_closeup.jpg", out_dir / "stickers_pose.jpg")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=None, help="摄像头编号, 例如 0")
    src.add_argument("--input", type=str, help="输入视频文件")
    src.add_argument("--showcase", type=str, help="输出目录: 用 LFW 照片生成静态效果对比图")
    ap.add_argument("--output", type=str, help="把处理结果写成 mp4 (mp4v 编码)")
    ap.add_argument("--effects", type=str, default="sunglasses,smooth,lipstick",
                    help="逗号分隔的特效, 分号分隔多组 (配合 --cycle)。可选: " + ",".join(EFFECT_ORDER))
    ap.add_argument("--cycle", type=float, default=0, help="每隔多少秒切换到下一组特效")
    ap.add_argument("--compare", action="store_true", help="左右对比: 原图 | 效果")
    ap.add_argument("--num-faces", type=int, default=2)
    ap.add_argument("--sticker-mode", choices=["2d", "3d"], default="3d")
    ap.add_argument("--no-smooth", action="store_true", help="关闭 One Euro 关键点平滑")
    ap.add_argument("--no-window", action="store_true", help="不弹窗口 (只写文件)")
    ap.add_argument("--no-hud", action="store_true", help="不画 FPS 等信息")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--scale-width", type=int, default=0, help="先把输入缩放到这个宽度")
    ap.add_argument("--crop", type=lambda s: [int(v) for v in s.split(",")], default=None,
                    help="先裁剪输入画面: x0,y0,x1,y1")
    ap.add_argument("--width", type=int, default=640, help="摄像头分辨率")
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()
    if args.no_window or args.showcase:
        disable_power_throttling()   # 无窗口时 Windows 会把进程当后台任务降速, HUD 上的 FPS 会偏低 3 倍
    if args.showcase:
        run_showcase(Path(args.showcase))
        return
    if args.input is None and args.camera is None:
        args.camera = 0
    run_video(args)


if __name__ == "__main__":
    main()
