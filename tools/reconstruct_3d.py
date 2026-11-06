"""单张图片 3D 人脸重建演示: SCRFD 检测 -> 3DDFA_V2 拟合 3DMM -> 稠密网格 -> 渲染与导出。

用法:
    python tools/reconstruct_3d.py --input data/lfw/Colin_Powell/Colin_Powell_0001.jpg
    python tools/reconstruct_3d.py --input my.jpg --outdir outputs/face3d/me --max-faces 3

输出 (默认 outputs/face3d/<文件名>/):
    overview.jpg       原图 | 68 点 | 网格叠加 | 深度图 | 姿态, 一张图看全流程
    landmarks68.jpg    68 个 3D 关键点 (投影到图上)
    mesh_overlay.jpg   带光照的网格半透明叠加 (检查对齐)
    depth.png          深度图 (越亮越近)
    pose.jpg           头部姿态坐标轴 + yaw/pitch/roll
    mesh.obj           带顶点颜色的网格, 可用 MeshLab / Blender 打开
    multiview.jpg      "摆正"后的纹理网格在不同 yaw / pitch 下的渲染 (最后一行为石膏像)
    turntable.mp4      转台视频 (左: 纹理, 右: 石膏)
    result.json        每张脸的检测框、姿态角、耗时
多张脸时, 第 2 张脸起的单脸文件加后缀 _1, _2, ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facevision.detector import SCRFD  # noqa: E402
from facevision.face3d import TDDFA3D, io, render  # noqa: E402
from facevision.paths import OUTPUTS  # noqa: E402

YAWS = (-60, -30, 0, 30, 60)
PITCHES = (-20, 0, 20)  # pitch > 0 是低头


def make_multiview(fv: np.ndarray, tri: np.ndarray, colors: np.ndarray, size: int) -> np.ndarray:
    rows = []
    for pitch in PITCHES:
        tiles = [render.render_view(fv, tri, colors, yaw, pitch, size) for yaw in YAWS]
        rows.append(tiles)
    rows.append([render.render_view(fv, tri, None, yaw, 0, size) for yaw in YAWS])  # 石膏像
    for r, tiles in enumerate(rows):
        for c, tile in enumerate(tiles):
            pitch = PITCHES[r] if r < len(PITCHES) else 0
            render.put_label(tile, f"yaw {YAWS[c]:+d} pitch {pitch:+d}", (4, 4), scale=size / 600)
    return np.vstack([np.hstack(t) for t in rows])


def make_turntable(path: Path, fv: np.ndarray, tri: np.ndarray, colors: np.ndarray,
                   size: int, frames: int, fps: int = 30) -> bool:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size * 2, size))
    if not writer.isOpened():
        return False
    for i in range(frames):
        phase = 2 * np.pi * i / frames
        yaw, pitch = 65 * np.sin(phase), 12 * np.sin(2 * phase)   # 左右转一圈, 顺带上下点头
        tex = render.render_view(fv, tri, colors, yaw, pitch, size)
        geo = render.render_view(fv, tri, None, yaw, pitch, size)
        writer.write(np.hstack([tex, geo]))
    writer.release()
    return True


def tile_row(images: list[np.ndarray], labels: list[str], height: int) -> np.ndarray:
    tiles = []
    for im, lab in zip(images, labels):
        im = cv2.resize(im, (int(round(im.shape[1] * height / im.shape[0])), height), interpolation=cv2.INTER_AREA)
        tiles.append(render.put_label(im, lab, (6, 6), scale=height / 700))
    return np.hstack(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="输入图片")
    ap.add_argument("--outdir", default=None, help="输出目录, 默认 outputs/face3d/<文件名>")
    ap.add_argument("--max-faces", type=int, default=0, help="最多处理几张脸, 0 = 全部")
    ap.add_argument("--view-size", type=int, default=256, help="多视角图中每格的边长")
    ap.add_argument("--video-size", type=int, default=360, help="转台视频每半边的边长")
    ap.add_argument("--frames", type=int, default=90, help="转台视频帧数 (30 fps)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--refit", action="store_true", help="用第一次的 68 点重新裁剪再拟合一次")
    ap.add_argument("--threads", type=int, default=1, help="PyTorch CPU 线程数 (小网络 1 个就够)")
    args = ap.parse_args()

    in_path = Path(args.input)
    img = cv2.imread(str(in_path))
    if img is None:
        sys.exit(f"读不到图片: {in_path}")
    outdir = Path(args.outdir) if args.outdir else OUTPUTS / "face3d" / in_path.stem
    outdir.mkdir(parents=True, exist_ok=True)

    det = SCRFD()
    tddfa = TDDFA3D(threads=args.threads)
    tri = tddfa.tri
    dets, _ = det.detect(img, max_num=args.max_faces)
    if len(dets) == 0:
        sys.exit("没有检测到人脸")
    print(f"{in_path.name}: {img.shape[1]}x{img.shape[0]}, 检测到 {len(dets)} 张脸")

    # 拟合 (第一次调用有 PyTorch 预热开销, 先空跑一次, 计时才准)
    tddfa.fit(img, dets[0])
    results, fit_ms = [], []
    for box in dets:
        t0 = time.perf_counter()
        res = tddfa.fit(img, box)
        if args.refit:
            res = tddfa.refit(img, res)
        fit_ms.append((time.perf_counter() - t0) * 1000)
        results.append(res)

    # 2D 可视化: LFW 只有 250x250, 先放大再画, 线条和网格看得更清楚
    k = max(1.0, 500 / min(img.shape[:2]))
    disp = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC) if k > 1 else img.copy()
    verts_disp = [r.vertices * k for r in results]
    t0 = time.perf_counter()
    overlay = render.render_mesh_overlay(disp, verts_disp, tri)
    overlay_ms = (time.perf_counter() - t0) * 1000
    depth = render.render_depth(disp.shape, verts_disp, tri)
    lmk_img, pose_img = disp.copy(), disp.copy()
    for r in results:
        lmk = r.landmarks68 * k
        lmk_img = render.draw_landmarks68(lmk_img, lmk)
        face_len = np.ptp(lmk[:, 1]) * 0.6
        pose_img = render.draw_pose_axes(pose_img, r.R, lmk[30], face_len)
        if len(results) > 1:  # 多张脸: 角度写在各自下巴下面 (限制在图内)
            x = int(np.clip(lmk[8, 0] - 60, 0, pose_img.shape[1] - 170))
            y = int(np.clip(lmk[8, 1] + 30, 20, pose_img.shape[0] - 8))
            cv2.putText(pose_img, "y%+.0f p%+.0f r%+.0f" % r.pose_deg, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    if len(results) == 1:
        render.put_label(pose_img, "yaw %+.1f  pitch %+.1f  roll %+.1f" % results[0].pose_deg, bottom=True)
    cv2.imwrite(str(outdir / "landmarks68.jpg"), lmk_img)
    cv2.imwrite(str(outdir / "mesh_overlay.jpg"), overlay)
    cv2.imwrite(str(outdir / "depth.png"), depth)
    cv2.imwrite(str(outdir / "pose.jpg"), pose_img)
    overview = tile_row([disp, lmk_img, overlay, depth, pose_img],
                        ["input", "68 landmarks", "3D mesh", "depth", "head pose"], 360)
    cv2.imwrite(str(outdir / "overview.jpg"), overview)

    # 每张脸: OBJ + 多视角 + 转台视频
    faces_json = []
    for i, (r, box) in enumerate(zip(results, dets)):
        suffix = "" if i == 0 else f"_{i}"
        colors = render.texture_from_image(img, r.vertices, tri, tddfa.mirror_index)
        io.save_obj(outdir / f"mesh{suffix}.obj", r.vertices, tri, colors, height=img.shape[0])
        fv = tddfa.frontal_vertices(r)
        t0 = time.perf_counter()
        render.render_view(fv, tri, colors, 30, 0, 512)
        view_ms = (time.perf_counter() - t0) * 1000
        cv2.imwrite(str(outdir / f"multiview{suffix}.jpg"), make_multiview(fv, tri, colors, args.view_size))
        video_ok = None
        if not args.no_video:
            video_ok = make_turntable(outdir / f"turntable{suffix}.mp4", fv, tri, colors, args.video_size, args.frames)
        faces_json.append({
            "bbox": [round(float(x), 1) for x in box[:4]], "score": round(float(box[4]), 3),
            "roi_box": [round(float(x), 1) for x in r.roi_box],
            "pose_deg": {"yaw": round(r.pose_deg[0], 2), "pitch": round(r.pose_deg[1], 2),
                         "roll": round(r.pose_deg[2], 2)},
            "fit_ms": round(fit_ms[i], 1), "render_view_512_ms": round(view_ms, 1),
            "video": video_ok,
        })
        print(f"  face {i}: yaw {r.pose_deg[0]:+.1f}  pitch {r.pose_deg[1]:+.1f}  roll {r.pose_deg[2]:+.1f}"
              f"  | fit {fit_ms[i]:.1f} ms, render 512px {view_ms:.0f} ms")

    summary = {"input": str(in_path), "image_size": [img.shape[1], img.shape[0]],
               "num_vertices": tddfa.num_vertices, "num_triangles": int(len(tri)),
               "mesh_overlay_ms": round(overlay_ms, 1), "faces": faces_json}
    (outdir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存到 {outdir}")


if __name__ == "__main__":
    main()
