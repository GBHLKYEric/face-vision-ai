"""FaceVision 统一演示界面 (Gradio)。

    python app/gradio_app.py            # 浏览器打开 http://127.0.0.1:7860

标签页: 人脸分析 | 1:1 人脸验证 | 1:N 人脸识别 | 实时特效(摄像头) | 美颜美妆/贴纸 | 属性编辑(StarGAN) | 3D 重建
所有模型懒加载: 第一次用到某个功能时才加载对应模型, 启动快、省内存。
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from facevision.align import norm_crop  # noqa: E402
from facevision.analyzer import FaceAnalyzer  # noqa: E402
from facevision.draw import draw_faces, put_label  # noqa: E402
from facevision.gallery import FaceGallery  # noqa: E402
from facevision.paths import DATA, OUTPUTS  # noqa: E402


def _lfw_threshold() -> float:
    """1:1 验证的阈值直接用 LFW 10 折交叉验证选出来的平均阈值。"""
    f = OUTPUTS / "lfw" / "result_arcface_r50.json"
    if f.exists():
        return float(json.loads(f.read_text(encoding="utf-8"))["threshold_cos"])
    return 0.25


THRESH = _lfw_threshold()


@functools.cache
def analyzer() -> FaceAnalyzer:
    return FaceAnalyzer(modules=("landmark_106", "recognition", "genderage"))


def rgb2bgr(img):
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------- 人脸分析 ----------------
def analyze_tab(img, show_106):
    if img is None:
        return None, "请上传图片"
    bgr = rgb2bgr(img)
    t = time.perf_counter()
    faces = analyzer().analyze(bgr)
    ms = (time.perf_counter() - t) * 1000
    info = [{"#": i, "置信度": round(f.score, 3), "性别": f.gender, "年龄": f.age,
             "框": [int(v) for v in f.bbox]} for i, f in enumerate(faces)]
    text = f"检测到 {len(faces)} 张人脸, 全流程 {ms:.0f} ms (检测+106点+识别特征+性别年龄)\n" + \
        json.dumps(info, ensure_ascii=False, indent=1)
    return bgr2rgb(draw_faces(bgr, faces, show_106=show_106)), text


# ---------------- 1:1 验证 ----------------
def verify_tab(img1, img2):
    if img1 is None or img2 is None:
        return None, "请上传两张图片"
    crops, embs = [], []
    for img in (img1, img2):
        bgr = rgb2bgr(img)
        faces = analyzer().analyze(bgr, max_num=1)
        if not faces:
            return None, "有一张图没有检测到人脸"
        crops.append(norm_crop(bgr, faces[0].kps)[0])
        embs.append(faces[0].embedding)
    sim = float(np.dot(embs[0], embs[1]))
    same = sim > THRESH
    vis = np.hstack([cv2.resize(c, (224, 224)) for c in crops])
    put_label(vis, f"cos={sim:.3f}", (5, 30), 1, (0, 200, 0) if same else (0, 0, 220))
    verdict = "✅ 同一个人" if same else "❌ 不是同一个人"
    return bgr2rgb(vis), (f"{verdict}\n余弦相似度 = {sim:.4f}  (阈值 {THRESH:.3f}, 由 LFW 10 折交叉验证选出)\n"
                          "左右两张是对齐后的 112x112 人脸, 这正是识别网络看到的输入。")


# ---------------- 1:N 识别 ----------------
GALLERY = FaceGallery(threshold=THRESH)


def gallery_demo_people():
    """默认用 LFW 里 8 位名人的第 1 张照片建库, 方便直接试 (查询时用他们的其他照片)。"""
    people = ["George_W_Bush", "Colin_Powell", "Tony_Blair", "Serena_Williams", "Angelina_Jolie",
              "Arnold_Schwarzenegger", "Jennifer_Lopez", "Hugo_Chavez"]
    for p in people:
        f = DATA / "lfw" / p / f"{p}_0001.jpg"
        if f.exists() and p not in GALLERY.names:
            faces = analyzer().analyze(cv2.imread(str(f)), max_num=1)
            if faces:
                GALLERY.add(p, faces[0].embedding)
    return "库中人员: " + ", ".join(GALLERY.names)


def register(img, name):
    if img is None or not name:
        return "请上传图片并填写姓名"
    faces = analyzer().analyze(rgb2bgr(img), max_num=1)
    if not faces:
        return "没有检测到人脸"
    GALLERY.add(name.strip(), faces[0].embedding)
    return f"已注册 {name}. 库中人员: " + ", ".join(GALLERY.names)


def identify(img):
    if img is None:
        return None, "请上传图片"
    bgr = rgb2bgr(img)
    faces = analyzer().analyze(bgr)
    labels, lines = [], []
    for i, f in enumerate(faces):
        name, score = GALLERY.identify(f.embedding)
        labels.append(f"{name} {score:.2f}")
        top = ", ".join(f"{n}:{s:.2f}" for n, s in GALLERY.search(f.embedding, 3))
        lines.append(f"人脸{i}: {name} (top3: {top})")
    return bgr2rgb(draw_faces(bgr, faces, show_106=False, labels=labels)), "\n".join(lines) or "没有人脸"


# ---------------- 特效 / StarGAN / 3D: 由各自模块提供, 加载失败时给出提示 ----------------
@functools.cache
def effect_pipeline():
    from facevision.effects.pipeline import EffectPipeline
    return EffectPipeline


EFFECT_NAMES = ["sunglasses", "hat", "cat_ears", "mustache", "smooth", "whiten", "slim", "big_eyes",
                "lipstick", "blush", "eyeliner", "mesh", "avatar"]
_pipes: dict = {}


def run_effects(img, effects, video=False):
    """静态图用 image 模式 (每张独立检测); 摄像头用 video 模式 (帧间跟踪 + One Euro 去抖)。"""
    if img is None:
        return None
    mode = "video" if video else "image"
    if mode not in _pipes:
        _pipes[mode] = effect_pipeline()(list(effects), mode=mode)
    pipe = _pipes[mode]
    pipe.effects = set(effects)          # 勾选变化时只改特效集合, 不重建模型
    return bgr2rgb(pipe.process(rgb2bgr(img)))


@functools.cache
def stargan():
    from facevision.gan.stargan import StarGANEditor
    return StarGANEditor()


STARGAN_EDITS = {"黑发": "black_hair", "金发": "blond_hair", "棕发": "brown_hair",
                 "性别反转": "flip_gender", "年龄反转": "flip_age"}


def stargan_tab(img, edit):
    if img is None:
        return None
    return bgr2rgb(stargan().edit_photo(rgb2bgr(img), edit))


def recon3d_tab(img):
    if img is None:
        return None, None, None
    from facevision.face3d.demo import reconstruct_for_ui
    overlay, views, obj_path = reconstruct_for_ui(rgb2bgr(img))
    return bgr2rgb(overlay), bgr2rgb(views), obj_path


def build_ui():
    sample = str(DATA / "lfw" / "Colin_Powell" / "Colin_Powell_0001.jpg")
    with gr.Blocks(title="FaceVision 人脸识别与特效") as demo:
        gr.Markdown("# FaceVision · 人脸识别与特效\n检测 SCRFD · 对齐 Umeyama · 识别 ArcFace-R50 "
                    "(LFW 10 折准确率见 README) · 特效 MediaPipe · 属性编辑 StarGAN · 3D 重建 3DDFA-V2")
        with gr.Tab("人脸分析"):
            with gr.Row():
                a_in = gr.Image(label="输入", value=sample if Path(sample).exists() else None)
                a_out = gr.Image(label="结果")
            show106 = gr.Checkbox(True, label="显示 106 个关键点")
            a_txt = gr.Textbox(label="详情", lines=8)
            gr.Button("分析", variant="primary").click(analyze_tab, [a_in, show106], [a_out, a_txt])
        with gr.Tab("1:1 人脸验证"):
            with gr.Row():
                v1, v2 = gr.Image(label="照片 A"), gr.Image(label="照片 B")
            v_out = gr.Image(label="对齐后的人脸")
            v_txt = gr.Textbox(label="结论", lines=3)
            gr.Button("比对", variant="primary").click(verify_tab, [v1, v2], [v_out, v_txt])
        with gr.Tab("1:N 人脸识别"):
            g_txt = gr.Textbox(label="人脸库", lines=2)
            with gr.Row():
                gr.Button("用 LFW 示例人物建库").click(gallery_demo_people, None, g_txt)
            with gr.Row():
                r_img, r_name = gr.Image(label="注册照片"), gr.Textbox(label="姓名")
            gr.Button("注册").click(register, [r_img, r_name], g_txt)
            with gr.Row():
                q_img, q_out = gr.Image(label="查询照片"), gr.Image(label="识别结果")
            q_txt = gr.Textbox(label="结果")
            gr.Button("识别", variant="primary").click(identify, q_img, [q_out, q_txt])
        with gr.Tab("实时特效 (摄像头)"):
            eff_live = gr.CheckboxGroup(EFFECT_NAMES, value=["sunglasses", "smooth", "lipstick"], label="特效")
            with gr.Row():
                cam = gr.Image(sources=["webcam"], streaming=True, label="摄像头")
                cam_out = gr.Image(label="输出")
            cam.stream(lambda im, e: run_effects(im, e, video=True), [cam, eff_live], cam_out,
                       stream_every=0.05, concurrency_limit=1)
        with gr.Tab("美颜美妆 / 贴纸"):
            eff = gr.CheckboxGroup(EFFECT_NAMES, value=["smooth", "whiten", "lipstick", "blush"], label="特效")
            with gr.Row():
                e_in, e_out = gr.Image(label="输入"), gr.Image(label="输出")
            gr.Button("应用", variant="primary").click(run_effects, [e_in, eff], e_out)
        with gr.Tab("属性编辑 (StarGAN)"):
            s_edit = gr.Radio(list(STARGAN_EDITS), value="金发", label="编辑")
            with gr.Row():
                s_in, s_out = gr.Image(label="输入"), gr.Image(label="输出")
            gr.Button("生成", variant="primary").click(lambda im, k: stargan_tab(im, STARGAN_EDITS[k]),
                                                       [s_in, s_edit], s_out)
        with gr.Tab("3D 人脸重建"):
            with gr.Row():
                d_in, d_out = gr.Image(label="输入"), gr.Image(label="网格叠加")
            d_views = gr.Image(label="多角度渲染")
            d_obj = gr.File(label="下载 .obj 网格")
            gr.Button("重建", variant="primary").click(recon3d_tab, d_in, [d_out, d_views, d_obj])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    build_ui().queue().launch(server_name=args.host, server_port=args.port)
