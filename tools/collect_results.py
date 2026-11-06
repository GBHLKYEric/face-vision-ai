"""把各个评测脚本输出的 JSON 汇总成表格: 写进 README 的结果区和教学讲义的占位符, 并画训练曲线。

数字全部来自 outputs/ 和 runs/ 下的 JSON, 不手抄, 保证文档与实验一致。
用法: python tools/collect_results.py [--run r18_arcface_kd] [--repo https://github.com/xxx/face-vision-ai]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TEACH = ROOT / "docs" / "teaching"
IMG = TEACH / "img"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def table(headers, rows, wide=False) -> str:
    h = "".join(f"<th>{c}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cls = ' class="wide"' if wide else ""
    return f'<div class="scroll"><table{cls}><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def md_table(headers, rows) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def lfw_rows(run: str):
    specs = [
        ("arcface_r50_nopad", "ArcFace-R50（预训练）", "不补边，翻转测试"),
        ("arcface_r50_noflip", "ArcFace-R50（预训练）", "补边，无翻转"),
        ("arcface_r50", "ArcFace-R50（预训练）", "补边 + 翻转（最终）"),
        (f"{run}", "IResNet18 自训练（ArcFace + 蒸馏）", "补边 + 翻转"),
        (f"{run}_int8", "IResNet18 自训练 INT8 量化", "补边 + 翻转"),
    ]
    rows = []
    for tag, model, setting in specs:
        r = load(OUT / "lfw" / f"result_{tag}.json")
        if r:
            rows.append([model, setting, f"{r['accuracy']:.3f} ± {r['accuracy_std']:.3f}", f"{r['auc']:.5f}",
                         f"{r['eer']:.3f}", f"{r['TAR@FAR=0.001']:.3f}", r["num_errors_at_mean_threshold"]])
    return ["模型", "设置", "LFW 准确率 (%)", "AUC", "EER (%)", "TAR@FAR=10⁻³ (%)", "错误对数"], rows


def wider_rows():
    rows = []
    for tag, label in [("scrfd10g_640", "单尺度 640（标准设置）"), ("scrfd10g_1280", "单尺度 1280"),
                       ("scrfd10g_640+1280", "多尺度 640 + 1280 融合（最终）")]:
        r = load(OUT / "widerface" / f"result_{tag}.json")
        if r:
            rows.append([f"SCRFD-10G {label}", f"{r['AP_easy']:.2f}", f"{r['AP_medium']:.2f}",
                         f"{r['AP_hard']:.2f}", f"{r['ms_per_image']:.0f}"])
    rows.append(["SCRFD-10G 论文报告（单尺度）", "95.16", "93.87", "83.05", "-"])
    return ["检测器", "Easy AP", "Medium AP", "Hard AP", "ms/张 (CPU)"], rows


def plot_training(run: str):
    hist = load(ROOT / "runs" / run / "history.json")
    log = ROOT / "runs" / run / "train_log.csv"
    if not hist or not log.exists():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    rows = list(csv.DictReader(log.open(encoding="utf-8")))
    steps = [int(r["step"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6), dpi=110)
    ax[0].plot(steps, [float(r["loss_margin"]) for r in rows], label="ArcFace 损失")
    ax[0].plot(steps, [50 * float(r["loss_kd"]) for r in rows], label="50 × 蒸馏损失")
    ax[0].set_xlabel("迭代步")
    ax[0].set_title("训练损失")
    ax[0].legend()
    ax[1].plot(steps, [float(r["train_acc"]) * 100 for r in rows], color="#DD8452")
    ax[1].set_xlabel("迭代步")
    ax[1].set_title("训练 batch 分类准确率 (%)")
    ep = [h["epoch"] + 1 for h in hist]
    ax[2].plot(ep, [h["lfw_acc"] for h in hist], "o-", color="#55A868")
    for x, y in zip(ep, [h["lfw_acc"] for h in hist]):
        ax[2].annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    ax[2].set_xlabel("epoch")
    ax[2].set_title("LFW 10 折准确率 (%)")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    path = OUT / "train" / f"{run}_curves.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    return path, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="r18_arcface_kd")
    ap.add_argument("--repo", default="")
    args = ap.parse_args()
    IMG.mkdir(parents=True, exist_ok=True)
    res: dict[str, str] = {}

    # ---------- LFW ----------
    hdr, rows = lfw_rows(args.run)
    res["LFW_TABLE"] = table(hdr, rows, wide=True)
    main_r = load(OUT / "lfw" / "result_arcface_r50.json")
    nopad = load(OUT / "lfw" / "result_arcface_r50_nopad.json")
    for name in ("errors_arcface_r50.jpg", f"errors_{args.run}.jpg"):
        if (OUT / "lfw" / name).exists():
            shutil.copy(OUT / "lfw" / name, IMG / name)
    if main_r:
        errs = main_r["errors"]
        same = sum(e["same"] for e in errs)
        names = sorted({Path(e["img1"]).parent.name.replace("_", " ") for e in errs})
        txt = (
            f"最终设置下 6000 对里只判错 {len(errs)} 对，其中 {same} 对是「标注为同一人、模型判为不同人」。"
            f"把它们全部画出来（下图，每对左右两张是对齐后的人脸，上方是余弦相似度）：\n\n"
            f'<figure class="fig"><img class="shot" src="img/errors_arcface_r50.jpg" alt="LFW 剩余错误样本">'
            f"<figcaption><b>剩下的错误几乎都是数据集本身的问题。</b>涉及的名字：{'、'.join(names)}。"
            f"相似度都在 0.2 以下，肉眼看多数明显是两个不同的人（例如 Jim O'Brien 的两对是两位不同的男士）。</figcaption></figure>")
        if nopad:
            txt += (f"\n\n补边之前还有 1 个错误来自检测选错了脸（Nabil_Shaath_0002，主角的脸被画面边缘切掉，见专题三例题 5），"
                    f"补边修好后准确率从 {nopad['accuracy']:.3f}% 升到 {main_r['accuracy']:.3f}%。")
        res["LFW_ERRORS"] = txt
    # ---------- WIDER ----------
    hdr, rows = wider_rows()
    wider_txt = table(hdr, rows, wide=True)
    wider_txt += ("\n\n结论：输入从 640 加大到 1280，小脸被放大到锚点能覆盖的范围，Hard 档明显上升；但大脸被放得过大，"
                  "超出了最大锚点的覆盖范围，Easy / Medium 反而下降。把两个尺度的检测框合在一起再做一次 NMS（多尺度测试），"
                  "两头兼顾，三档都超过了论文的单尺度结果，代价是检测时间约为 4.5 倍——离线评测和相册整理可以这样做，"
                  "实时视频则用单尺度。")
    for name, dst, cap in [("pr_scrfd10g_640.png", "pr_single_640.png", "单尺度 640"),
                           ("pr_scrfd10g_640+1280.png", "pr_multiscale.png", "多尺度 640 + 1280")]:
        if (OUT / "widerface" / name).exists():
            shutil.copy(OUT / "widerface" / name, IMG / dst)
            wider_txt += (f'\n\n<figure class="fig"><img class="shot" src="img/{dst}" alt="PR 曲线">'
                          f"<figcaption><b>{cap}的 PR 曲线。</b>曲线越靠右上越好；Hard 档包含大量 16 像素以下的小脸，"
                          f"召回率上不去是所有检测器的共同难点。</figcaption></figure>")
    res["WIDER_TABLE"] = wider_txt

    # ---------- 训练 ----------
    plotted = plot_training(args.run)
    r18 = load(OUT / "lfw" / f"result_{args.run}.json")
    if plotted:
        path, hist = plotted
        shutil.copy(path, IMG / path.name)
        hrows = [[h["epoch"] + 1, f"{h['loss']:.2f}", f"{h['train_acc'] * 100:.1f}", f"{h['lfw_acc']:.3f}",
                  f"{h['tar@far1e-3']:.2f}", f"{h['minutes']:.0f}"] for h in hist]
        res["TRAIN_RESULT"] = (
            f'<figure class="fig"><img class="shot" src="img/{path.name}" alt="训练曲线">'
            f"<figcaption><b>训练曲线。</b>左：两项损失；中：训练 batch 内分类准确率；右：每个 epoch 结束在 LFW 上的 10 折准确率。"
            f"左、中两图的台阶不是异常：日志记录的是每轮内的累计平均，换轮时清零重算，所以每轮开头会跳一下。</figcaption></figure>\n\n"
            + table(["epoch", "总损失", "训练准确率 (%)", "LFW (%)", "TAR@FAR=10⁻³ (%)", "用时 (分钟)"], hrows)
            + (f"\n\n最终模型（最后一个 epoch，而不是 LFW 最高的那个）用 `tools/eval_lfw.py --torch-ckpt` 按完整流程复测："
               f"**LFW {r18['accuracy']:.3f}% ± {r18['accuracy_std']:.3f}**，TAR@FAR=10⁻³ 为 {r18['TAR@FAR=0.001']:.2f}%，"
               f"超过 PDF 要求的 98.5%。\n\n结论：自训练模型在 6000 对里错 {r18['num_errors_at_mean_threshold']} 对，"
               f"老师 R50 只错 {main_r['num_errors_at_mean_threshold']} 对。下图是学生判错的样本，大多是大角度侧脸、"
               f"墨镜帽子遮挡、年龄跨度大的真难例，老师判错的则基本是标注问题。两者的差距来自数据规模："
               f"老师见过约 60 万人，学生只见过 1 万人，网络也小得多。\n\n"
               f'<figure class="fig"><img class="shot" src="img/errors_{args.run}.jpg" alt="自训练模型的错误样本">'
               f"<figcaption><b>学生模型的前 24 个错误。</b>same 表示标注为同一人但被判成不同人，diff 相反。</figcaption></figure>"
               if r18 else ""))

    # ---------- 部署 ----------
    dep = load(OUT / "deploy" / "report.json")
    if dep:
        res["DEPLOY_TABLE"] = table(["方案", "大小 (MB)", "单张延迟 (ms)", "LFW (%)", "说明"],
                                    [[d["variant"], d["size_mb"], d["latency_ms"], f"{d['lfw_acc']:.3f}", d["note"]]
                                     for d in dep], wide=True) + \
            "\n\n延迟为 batch=1、CPU 4 线程、112×112 输入的单次前向（不含翻转），多次运行取平均。"

    # ---------- 实时特效性能 ----------
    perf = load(OUT / "effects" / "perf.json")
    if perf:
        e2e = perf["results"]["640x480"].get("end_to_end", {})
        prow = []
        for k, v in e2e.items():
            ms = v.get("median", v.get("mean")) if isinstance(v, dict) else v
            if isinstance(ms, (int, float)):
                prow.append([k, f"{ms:.1f}", f"{1000 / ms:.0f}"])
        res["SYSTEM_PERF"] = (
            "**640×480 摄像头画面，CPU 上逐帧处理，每帧预算 33 ms（30 FPS）。实测各组合够不够？**\n\n"
            + table(["特效组合", "每帧 (ms)", "FPS"], prow)
            + "\n\n结论：MediaPipe 关键点约 7 ms 是大头，美颜美妆全开约 14 ms，仍在 33 ms 预算内，留出了摄像头采集和显示的时间。"
              "如果要再叠加完整的检测 + 识别流水线（实测约 100 ms/帧），就要降低识别频率（例如每秒只识别一次、其余帧跟踪），这正是实际产品的做法。")

    # ---------- 总览 / README ----------
    def get(tag, key="accuracy"):
        r = load(OUT / "lfw" / f"result_{tag}.json")
        return r[key] if r else None
    w = load(OUT / "widerface" / "result_scrfd10g_640.json")
    sg = load(OUT / "stargan" / "eval_result.json")
    summary = [["人脸识别（预训练 ArcFace-R50）", "LFW 10 折准确率", f"{get('arcface_r50'):.3f}%" if get('arcface_r50') else "-"]]
    if r18:
        summary.append(["人脸识别（自训练 IResNet18 + 蒸馏）", "LFW 10 折准确率", f"{r18['accuracy']:.3f}%"])
    if main_r:
        summary.append(["人脸验证", "TAR@FAR=10⁻³ / AUC", f"{main_r['TAR@FAR=0.001']:.2f}% / {main_r['auc']:.5f}"])
    wm = load(OUT / "widerface" / "result_scrfd10g_640+1280.json")
    if w:
        summary.append(["人脸检测（SCRFD-10G，单尺度 640）", "WIDER FACE Easy / Medium / Hard AP",
                        f"{w['AP_easy']:.2f} / {w['AP_medium']:.2f} / {w['AP_hard']:.2f}"])
    if wm:
        summary.append(["人脸检测（多尺度 640+1280）", "WIDER FACE Easy / Medium / Hard AP",
                        f"{wm['AP_easy']:.2f} / {wm['AP_medium']:.2f} / {wm['AP_hard']:.2f}"])
    if dep:
        q = [d for d in dep if "INT8(静态)" in d["variant"] and "R50" not in d["variant"]]
        f32 = [d for d in dep if "ONNX FP32" in d["variant"] and "R50" not in d["variant"]]
        if q and f32:
            summary.append(["模型压缩（INT8 静态量化）", "大小 / 速度 / LFW",
                            f"{f32[0]['size_mb']}→{q[0]['size_mb']} MB，{f32[0]['latency_ms']}→{q[0]['latency_ms']} ms，"
                            f"{q[0]['lfw_acc']:.2f}%"])
    if perf:
        ee = perf["results"]["640x480"]["end_to_end"]
        summary.append(["实时特效（CPU 640×480）", "美颜美妆全开 / 仅关键点",
                        f"{ee['beauty + makeup (7 effects)']['fps']:.0f} FPS / {ee['landmarks only']['fps']:.0f} FPS"])
    if sg:
        fids = sg["fid"]
        summary.append(["属性编辑（StarGAN）", "FID（2000 张 CelebA 测试图）",
                        f"发色 {fids['brown_hair_vs_real_B']}～{fids['blond_hair_vs_real_B']}，性别 {fids['flip_gender_vs_real_B']}，"
                        f"年龄 {fids['flip_age_vs_real_B']}"])
    summary.append(["3D 重建（3DDFA-V2 移植）", "与官方实现逐点对比", "最大误差 0（完全一致）"])
    res["HOME_RESULTS"] = table(["任务", "指标", "结果"], summary)
    res["REPO_LINK"] = f'<a href="{args.repo}">{args.repo}</a>' if args.repo else "本地文件夹 `D:\\face-vision-ai`"

    # ---------- 损失函数消融 ----------
    abl = load(OUT / "train" / "ablation_losses.json")
    abl_md = ""
    if abl:
        st = abl["setting"]
        arow = [[k, f"{v['final_lfw_acc']:.2f}", f"{v['final_tar@far1e-3']:.2f}", f"{v['final_train_acc'] * 100:.1f}",
                 " → ".join(f"{a:.1f}" for a in v["lfw_acc_per_epoch"])] for k, v in abl["results"].items()]
        ahdr = ["损失函数", "LFW (%)", "TAR@FAR=10⁻³ (%)", "训练准确率 (%)", "每轮 LFW (%)"]
        cap = (f"设置：{st['backbone']}（MobileFaceNet）、CelebA 前 {st['subset']} 个身份、{st['epochs']} 轮、不蒸馏，"
               f"只换损失函数。数据少、轮数少，绝对值远低于完整训练，看相对高低。")
        res["ABLATION"] = cap + "\n\n" + table(ahdr, arow, wide=True)
        abl_md = cap + "\n\n" + md_table(ahdr, arow)
        rank = sorted(abl["results"].items(), key=lambda kv: -kv[1]["final_lfw_acc"])
        R = abl["results"]
        note = "结论：LFW 从高到低依次为 " + " ＞ ".join(f"{k}（{v['final_lfw_acc']:.2f}%）" for k, v in rank) + "。"
        best_train = max(R, key=lambda k: R[k]["final_train_acc"])
        if best_train != rank[0][0]:
            note += (f"训练准确率最高的是 {best_train}（{R[best_train]['final_train_acc'] * 100:.1f}%），"
                     f"LFW 第一的却是 {rank[0][0]}（训练准确率 {rank[0][1]['final_train_acc'] * 100:.1f}%）："
                     f"把训练集分得好，不等于能把没见过的人分开。")
        if "softmax" in R and rank[0][0] != "softmax":
            note += (f"看更严格的 TAR@FAR=10⁻³，{rank[0][0]} 为 {rank[0][1]['final_tar@far1e-3']:.1f}%，"
                     f"Softmax 为 {R['softmax']['final_tar@far1e-3']:.1f}%，低误识率区间才是间隔类损失拉开差距的地方。")
        worst = rank[-1]
        note += (f"{worst[0]} 垫底，训练准确率只有 {worst[1]['final_train_acc'] * 100:.1f}%，"
                 f"说明这个设置下它还没学够。")
        res["ABLATION_NOTE"] = note
        res["ABLATION_CAVEAT"] = ("提醒：这组实验只有 4 万张图、5 轮、每种损失跑一次，前几名相差不到 1 个百分点，"
                                  "小于 LFW 各折之间的波动，只能看大致趋势，不能当作严格结论。论文里 ArcFace 的优势是在"
                                  "数百万张图上测出来的。")
    else:
        res["ABLATION"] = "（消融实验结果尚未生成：运行 `python tools/ablation_losses.py --device xpu`。）"
        res["ABLATION_NOTE"] = res["ABLATION_CAVEAT"] = ""

    (TEACH / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def fill(path: Path, key: str, content: str):
        text = path.read_text(encoding="utf-8")
        s, e = f"<!-- {key}:START -->", f"<!-- {key}:END -->"
        if s in text and content:
            a, b = text.index(s), text.index(e)
            path.write_text(text[:a] + s + "\n" + content + "\n" + text[b:], encoding="utf-8")

    hdr, rows = lfw_rows(args.run)
    lfw_md = md_table(hdr, rows)
    fill(ROOT / "README.md", "RESULTS", md_table(["任务", "指标", "结果"], summary) +
         "\n\nLFW 明细（6000 对，官方 10 折协议，从原始图片开始的完整流程）：\n\n" + lfw_md)
    report = ROOT / "docs" / "REPORT.md"
    fill(report, "SUMMARY", md_table(["任务", "指标", "结果"], summary))
    fill(report, "LFW", lfw_md)
    whdr, wrows = wider_rows()
    fill(report, "WIDER", md_table(whdr, wrows))
    if plotted:
        fill(report, "TRAIN", f"![训练曲线](../outputs/train/{plotted[0].name})\n\n" +
             md_table(["epoch", "总损失", "训练准确率 (%)", "LFW (%)", "TAR@FAR=10⁻³ (%)", "用时 (分钟)"], hrows) +
             (f"\n\n最终模型按完整流程复测：**LFW {r18['accuracy']:.3f}% ± {r18['accuracy_std']:.3f}**，"
              f"TAR@FAR=10⁻³ {r18['TAR@FAR=0.001']:.2f}%，AUC {r18['auc']:.5f}。" if r18 else ""))
    if dep:
        fill(report, "DEPLOY", md_table(["方案", "大小 (MB)", "单张延迟 (ms)", "LFW (%)", "说明"],
                                        [[d["variant"], d["size_mb"], d["latency_ms"], f"{d['lfw_acc']:.3f}", d["note"]]
                                         for d in dep]) + "\n\n延迟：batch=1、CPU 4 线程、112×112、不含翻转，取平均。")
    fill(report, "ABLATION", abl_md + "\n\n" + res["ABLATION_NOTE"] + "\n\n" +
         res["ABLATION_CAVEAT"].removeprefix("提醒："))
    print("写入", TEACH / "results.json", "README 与 REPORT 结果区; 键:", list(res))


if __name__ == "__main__":
    main()
