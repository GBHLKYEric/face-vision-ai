"""生成项目汇报 PPT: docs/FaceVision_汇报.pptx (数字从 outputs/ 的 JSON 读取, 图片用实验产出)。

用法: python tools/make_slides.py [--run r18_arcface_kd]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INK, MUTED, ACCENT, GREEN, BG = (RGBColor(0x16, 0x20, 0x1A), RGBColor(0x56, 0x60, 0x58), RGBColor(0x6A, 0x3F, 0xA8),
                                 RGBColor(0x1E, 0x6E, 0x44), RGBColor(0xF2, 0xF5, 0xF1))
FONT = "Microsoft YaHei"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.blank = self.prs.slide_layouts[6]
        self.n = 0

    def _text(self, slide, x, y, w, h, text, size=18, bold=False, color=INK, align=None):
        tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        lines = text if isinstance(text, list) else [text]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            run = p.add_run()
            run.text = line
            run.font.size, run.font.bold, run.font.name = Pt(size), bold, FONT
            run.font.color.rgb = color
            p.space_after = Pt(6)
            if align is not None:
                p.alignment = align
        return tb

    def slide(self, title, subtitle=None):
        s = self.prs.slides.add_slide(self.blank)
        self.n += 1
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = BG
        bar = s.shapes.add_shape(1, 0, 0, Inches(0.18), self.prs.slide_height)
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
        self._text(s, 0.5, 0.3, 12, 0.8, title, 30, True)
        if subtitle:
            self._text(s, 0.5, 1.0, 12, 0.5, subtitle, 15, False, MUTED)
        self._text(s, 12.3, 7.0, 0.9, 0.4, str(self.n), 11, False, MUTED)
        return s

    def bullets(self, s, items, x=0.5, y=1.6, w=6.0, size=17):
        return self._text(s, x, y, w, 5.5, [f"•  {t}" for t in items], size)

    def image(self, s, path: Path, x, y, w=None, h=None):
        if not path.exists():
            return
        kw = {}
        if w:
            kw["width"] = Inches(w)
        if h:
            kw["height"] = Inches(h)
        s.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)

    def table(self, s, headers, rows, x=0.5, y=1.7, w=12.3, size=13):
        shape = s.shapes.add_table(len(rows) + 1, len(headers), Inches(x), Inches(y), Inches(w),
                                   Inches(0.45 * (len(rows) + 1)))
        t = shape.table
        for j, h in enumerate(headers):
            c = t.cell(0, j)
            c.text = str(h)
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size, r.font.bold, r.font.name = Pt(size), True, FONT
                    r.font.color.rgb = RGBColor(255, 255, 255)
            c.fill.solid()
            c.fill.fore_color.rgb = ACCENT
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row):
                c = t.cell(i, j)
                c.text = str(v)
                for p in c.text_frame.paragraphs:
                    for r in p.runs:
                        r.font.size, r.font.name = Pt(size), FONT
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(255, 255, 255) if i % 2 else RGBColor(0xEE, 0xE7, 0xF8)
        return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r18_arcface_kd")
    args = ap.parse_args()
    lfw = load(OUT / "lfw" / "result_arcface_r50.json")
    lfw_np = load(OUT / "lfw" / "result_arcface_r50_nopad.json")
    r18 = load(OUT / "lfw" / f"result_{args.run}.json")
    wider = load(OUT / "widerface" / "result_scrfd10g_640.json")
    dep = load(OUT / "deploy" / "report.json") or []
    sg = load(OUT / "stargan" / "eval_result.json")
    perf = load(OUT / "effects" / "perf.json")

    d = Deck()
    # 1 封面
    s = d.slide("FaceVision：人脸识别与特效技术", "检测 · 对齐 · 识别 · 实时特效 · 3D 重建 · GAN 属性编辑 —— 项目汇报")
    d._text(s, 0.5, 2.2, 7, 3, ["四个阶段全部完成，全部在无独立显卡的笔记本上实测",
                                f"LFW 10 折准确率 {lfw['accuracy']:.2f}%（预训练 R50）" if lfw else "",
                                f"自训练 ResNet18 + 蒸馏：LFW {r18['accuracy']:.2f}%" if r18 else "",
                                "实时美颜美妆 CPU 70 FPS（640×480）"], 20, False, INK)
    d.image(s, OUT / "effects" / "stickers_pose.jpg", 7.6, 2.0, w=5.3)
    # 2 目录
    s = d.slide("汇报目录")
    d.bullets(s, ["阶段一  环境搭建与数据集分析", "阶段二  检测 / 关键点 / 对齐 / 识别 / 训练 / 部署优化",
                  "阶段三  实时特效、3D 人脸重建、StarGAN 属性编辑", "阶段四  系统集成与性能",
                  "问题与解决、总结与展望"], size=22, w=12)
    # 3 系统架构
    s = d.slide("系统架构", "一张照片进来，每一步做什么")
    d.bullets(s, ["SCRFD-10G 检测：人脸框 + 5 个关键点（手写 anchor 解码与 NMS）",
                  "Umeyama 相似变换：对齐到 112×112 标准模板",
                  "ArcFace-ResNet：512 维特征，余弦相似度比对（1:1 / 1:N）",
                  "MediaPipe 478 点网格 → 贴纸 / 美颜 / 美妆 / 表情驱动头像",
                  "3DDFA-V2：62 个 3DMM 参数 → 38365 顶点网格",
                  "StarGAN：发色 / 性别 / 年龄编辑", "Gradio 网页 + OpenCV 实时演示"], w=6.4, size=16)
    d.image(s, OUT / "opencv_basics" / "basics_grid.jpg", 7.2, 1.7, w=5.6)
    # 4 数据集
    s = d.slide("阶段一：环境与数据集", "PyTorch 2.15 (Intel XPU) · ONNX Runtime · OpenCV 5 · MediaPipe · Docker")
    d.table(s, ["数据集", "规模", "用途"], [["LFW", "13,233 张 / 5,749 人", "识别评测（6000 对 10 折）"],
                                         ["CelebA", "202,599 张 / 10,177 人 / 40 属性", "识别训练、StarGAN、FID"],
                                         ["WIDER FACE val", "3,226 张图", "检测评测（Easy/Medium/Hard）"]], w=12.3)
    d.image(s, OUT / "datasets" / "lfw_images_per_person.png", 0.5, 3.6, h=3.4)
    d.image(s, OUT / "datasets" / "wider_face_sizes.png", 6.8, 3.6, h=3.4)
    # 5 检测
    s = d.slide("阶段二：人脸检测（WIDER FACE）", "官方评测协议 numpy 逐行移植")
    wrows = []
    for tag, name in [("scrfd10g_640", "单尺度 640"), ("scrfd10g_1280", "单尺度 1280"),
                      ("scrfd10g_640+1280", "多尺度 640+1280（最终）")]:
        r = load(OUT / "widerface" / f"result_{tag}.json")
        if r:
            wrows.append([name, r["AP_easy"], r["AP_medium"], r["AP_hard"]])
    wrows.append(["论文报告（单尺度）", 95.16, 93.87, 83.05])
    d.table(s, ["SCRFD-10G", "Easy AP", "Medium AP", "Hard AP"], wrows, w=7.6)
    d.image(s, OUT / "widerface" / "pr_scrfd10g_640+1280.png", 8.5, 1.6, w=4.5)
    d.bullets(s, ["锚点 + FPN：stride 8/16/32 分管小/中/大脸", "大输入利于小脸、伤害大脸 → 多尺度融合两头兼顾",
                  "批量 GPU 版：onnx2torch 转换后对拍误差 1.5e-5"], y=4.3, w=7.6, size=16)
    # 6 识别
    s = d.slide("阶段二：人脸识别（LFW 官方 10 折）", "从原始图片开始的完整流程：检测 → 对齐 → 特征 → 10 折")
    rows = []
    for tag, name in [("arcface_r50_nopad", "R50 不补边"), ("arcface_r50_noflip", "R50 补边，无翻转"),
                      ("arcface_r50", "R50 补边+翻转（最终）"),
                      (args.run, "自训练 R18 + 蒸馏"), (f"{args.run}_int8", "自训练 R18 INT8")]:
        r = load(OUT / "lfw" / f"result_{tag}.json")
        if r:
            rows.append([name, f"{r['accuracy']:.3f}%", f"{r['TAR@FAR=0.001']:.2f}%", f"{r['auc']:.5f}",
                         r["num_errors_at_mean_threshold"]])
    d.table(s, ["模型", "准确率", "TAR@FAR=1e-3", "AUC", "错误对数"], rows, w=12.3)
    d.image(s, OUT / "lfw" / "errors_arcface_r50.jpg", 0.5, 4.2, w=8.0)
    d._text(s, 8.8, 4.2, 4.2, 2.8, ["剩余错误均为「同名不同人」等标注问题，", "LFW 实际上已到天花板。",
                                    f"补边修复检测错误：{lfw_np['accuracy']:.3f}% → {lfw['accuracy']:.3f}%" if lfw and lfw_np else ""],
            15, False, MUTED)
    # 7 训练
    s = d.slide("阶段二：自己训练识别模型", "CelebA 20 万张 + IResNet18 + ArcFace(s=64, m=0.5) + R50 特征蒸馏，Intel 核显 bf16")
    d.image(s, OUT / "train" / f"{args.run}_curves.png", 0.5, 1.6, w=12.3)
    if r18:
        d._text(s, 0.5, 5.6, 12, 1.2, [f"最终 LFW {r18['accuracy']:.3f}% ± {r18['accuracy_std']:.3f}（要求 ≥ 98.5%）",
                                       "CelebA 与 LFW 人物不重叠；报告最后一个 epoch，不按测试集挑模型"], 17, True, GREEN)
    # 8 部署
    s = d.slide("阶段二：模型优化与部署", "BN 融合 · PyTorch 动态量化 · ONNX · ORT 静态 INT8 · 剪枝")
    if dep:
        d.table(s, ["方案", "大小 MB", "延迟 ms", "LFW %"],
                [[x["variant"], x["size_mb"], x["latency_ms"], f"{x['lfw_acc']:.2f}"] for x in dep], w=12.3, size=12)
        d._text(s, 0.5, 6.75, 11.5, 0.4, "延迟：batch=1、CPU 4 线程。R50 INT8 与 FP32 不是同一时段测的，同时段 FP32 为 41.7 ms。",
                12, False, MUTED)
    # 9 特效
    s = d.slide("阶段三：实时特效", "AR 贴纸 · 磨皮美白 · 瘦脸大眼 · 口红腮红 · 表情驱动头像 · One Euro 去抖")
    d.image(s, OUT / "effects" / "effects_grid.jpg", 0.5, 1.6, w=12.3)
    if perf:
        e = perf["results"]["640x480"]["end_to_end"]
        d._text(s, 0.5, 5.6, 12, 1.2, [f"CPU 640×480：仅关键点 {e['landmarks only']['fps']:.0f} FPS，"
                                       f"美颜+美妆 7 项全开 {e['beauty + makeup (7 effects)']['fps']:.0f} FPS，"
                                       f"卡通头像 {e['avatar']['fps']:.0f} FPS"], 17, True, GREEN)
    # 10 3D
    s = d.slide("阶段三：3D 人脸重建", "3DDFA-V2 移植：3DMM = 平均脸 + 40 个形状基 + 10 个表情基；纯 numpy 渲染器")
    d.image(s, ROOT / "docs" / "teaching" / "img" / "face3d_overview.jpg", 0.5, 1.6, w=6.3)
    d.image(s, ROOT / "docs" / "teaching" / "img" / "face3d_multiview.jpg", 7.0, 1.6, h=4.2)
    d._text(s, 0.5, 6.0, 12, 1, "与官方实现逐点对比：62 个参数、68 个关键点、38365 个顶点完全一致（最大误差 0）",
            16, True, GREEN)
    # 11 StarGAN
    s = d.slide("阶段三：StarGAN 属性编辑", "WGAN-GP + 属性分类 + 循环一致性；任意照片对齐后编辑再贴回")
    d.image(s, OUT / "stargan" / "grid_lfw_photos.jpg", 0.5, 1.6, h=5.6)
    if sg:
        f = sg["fid"]
        d.table(s, ["编辑", "FID"], [["真图 vs 真图（下限）", f["real_A_vs_real_B (baseline: same distribution, finite-sample floor)"]],
                                     ["黑发", f["black_hair_vs_real_B"]], ["金发", f["blond_hair_vs_real_B"]],
                                     ["棕发", f["brown_hair_vs_real_B"]], ["性别反转", f["flip_gender_vs_real_B"]],
                                     ["年龄反转", f["flip_age_vs_real_B"]]], x=7.6, y=1.6, w=5.2)
    # 12 系统
    s = d.slide("阶段四：系统集成", "Gradio 网页 7 个标签页 + OpenCV 实时演示 + 一键脚本 + Docker")
    d.bullets(s, ["人脸分析：框、106 点、性别年龄", "1:1 验证：阈值取自 LFW 10 折", "1:N 识别：人脸库注册与检索",
                  "摄像头实时特效（流式）", "美颜美妆 / 贴纸", "StarGAN 属性编辑", "3D 重建 + .obj 下载",
                  "单元测试覆盖对齐、NMS、指标、损失、BN 融合、特效、3D、GAN"], w=12, size=18)
    # 13 问题与解决
    s = d.slide("遇到的问题与解决")
    d.bullets(s, ["无 NVIDIA 显卡 → PyTorch XPU 跑 Intel 核显训练，bf16 提速近一倍",
                  "MS-Celeb-1M 已下架 → 改用与 LFW 人物不重叠的 CelebA",
                  "ONNX 检测模型 batch 固定为 1 → onnx2torch 转换，发现并修正 batch 维顺序",
                  "Windows 后台进程被降频（EcoQoS）→ 仅对本进程关闭节流",
                  "5 个 ONNX 模型线程空转互抢 CPU → 关闭空转，全流程 2122 ms → 99 ms",
                  "Umeyama 两点镜像 bug → 改用 det(U)·det(V) 判断",
                  "LFW 选错脸 → 检测前补边，准确率 99.833% → 99.850%",
                  "StarGAN 官方权重在 eval 模式下失真 → 使用实例统计量"], w=12.3, size=17)
    # 14 总结
    s = d.slide("总结与展望")
    d.bullets(s, ["完成 PDF 四阶段全部任务与交付物：代码、评测报告、性能优化方案、特效演示视频、技术文档、教学讲义",
                  "识别准确率达到 LFW 数据集的实际上限；自训练模型超过 98.5% 目标",
                  "展望：更难的评测集（IJB-C、CFP-FP）、结构化剪枝 + 微调、移动端（MNN/NCNN）部署、"
                  "扩散模型做属性编辑、3D 高斯泼溅做真实感头像"], w=12.3, size=18)
    path = ROOT / "docs" / "FaceVision_汇报.pptx"
    d.prs.save(path)
    print("saved", path, d.n, "slides")


if __name__ == "__main__":
    main()
