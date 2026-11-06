"""StarGAN (官方 CelebA 权重) 评测: FID、属性编辑成功率、重建误差, 以及展示图。

1) 画质 / 真实感: FID (Fréchet Inception Distance)
   把图像送进 ImageNet 上训练的 Inception-v3, 取 2048 维池化特征, 两组图像各拟合一个多元高斯 N(μ, Σ):
       FID = ‖μ₁ - μ₂‖² + Tr(Σ₁ + Σ₂ - 2·(Σ₁Σ₂)^½)
   即两个高斯之间的 Fréchet (Wasserstein-2) 距离, 越小说明生成图的整体分布越接近真实照片。
   注意: (a) 样本少时 FID 系统性偏高, 所以同时给出"真图 vs 真图"的基线 (同样 2000 张);
         (b) "全部变金发"的分布本来就和"各种发色混合"的真实分布不同, 所以发色编辑另外给出
             与"真实金发/黑发/棕发照片"比较的 FID (fid_vs_target_domain)。
2) 属性有没有真的改到位: 用独立的 GenderAge 模型 (InsightFace, 与 StarGAN 无关) 判断
   性别是否翻转、预测年龄是否朝目标方向变化。
3) 只改属性、不乱改其它: G(x, 原标签) 应该≈x (重建 L1)。

用法:
    python tools/eval_stargan.py                 # 默认 2000 张 CelebA test 图, 设备自动选择 (xpu 优先)
    python tools/eval_stargan.py --n 200 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facevision.gan.data import decode_png, iter_rows  # noqa: E402
from facevision.gan.stargan import (ATTRS, HAIR_ATTRS, SINGLE_EDITS, StarGANEditor, load_generator,  # noqa: E402
                                    official_test_labels, preprocess_aligned, to_image, to_tensor)
from facevision.gan.viz import save_grid  # noqa: E402
from facevision.paths import DATA, OUTPUTS  # noqa: E402

OUT = OUTPUTS / "stargan"
LFW_NAMES = ["Colin_Powell", "Angelina_Jolie", "George_W_Bush", "Serena_Williams", "Tony_Blair"]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        name = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
    return torch.device(name)


def load_faces(split: str, start: int, n: int, attr_filter: str | None = None):
    """从 parquet 顺序读取, 返回 (128x128 RGB uint8 [n,...], 标签 [n,5], 文件名)。

    attr_filter: 只保留该属性为 1 的图 (用于构造"真实金发"等参考集)。
    """
    faces, labels, names = [], [], []
    k = ATTRS.index(attr_filter) if attr_filter else None
    for name, b, lab in iter_rows(split, ATTRS, start=start):
        if k is not None and lab[k] < 0.5:
            continue
        faces.append(preprocess_aligned(np.asarray(decode_png(b))))
        labels.append(lab)
        names.append(name)
        if len(faces) == n:
            break
    return np.stack(faces), np.stack(labels), names


class InceptionFeatures:
    """pytorch-fid 的 Inception-v3 (与原版 TF FID 权重一致), 输出 2048 维 pool3 特征。"""

    def __init__(self, device: torch.device, batch: int = 50):
        from pytorch_fid.inception import InceptionV3
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device).eval()
        self.device, self.batch = device, batch

    @torch.no_grad()
    def __call__(self, imgs: np.ndarray) -> np.ndarray:
        """imgs: uint8 RGB (N,H,W,3)。模型内部会双线性缩放到 299 并归一化到 [-1, 1]。"""
        feats = []
        for i in range(0, len(imgs), self.batch):
            x = torch.from_numpy(imgs[i:i + self.batch]).permute(0, 3, 1, 2).float().div(255)
            feats.append(self.model(x.to(self.device))[0].flatten(1).cpu().numpy().astype(np.float64))
        return np.concatenate(feats)


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    """两个高斯 N(mu1,s1), N(mu2,s2) 的 Fréchet 距离 = ‖mu1-mu2‖² + Tr(s1) + Tr(s2) - 2·Tr((s1·s2)^½)。

    Tr((s1·s2)^½) 的稳定算法: 令 R = s1^½ (对称半正定, 用特征分解求), 则 s1·s2 与对称矩阵
    R·s2·R 相似, 特征值相同, 所以 Tr((s1·s2)^½) = Σ sqrt(eig(R·s2·R))。
    (pytorch-fid 0.3.0 自带的 calculate_frechet_distance 调用 scipy.linalg.sqrtm(..., disp=False),
    而 scipy 1.18 已删除 disp 参数, 会直接报错, 所以这里自己实现。)
    """
    w, v = np.linalg.eigh(s1)
    r = (v * np.sqrt(np.clip(w, 0, None))) @ v.T
    eig = np.linalg.eigvalsh(r @ s2 @ r)
    tr_covmean = np.sqrt(np.clip(eig, 0, None)).sum()
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(s1) + np.trace(s2) - 2 * tr_covmean)


def fid(f1: np.ndarray, f2: np.ndarray) -> float:
    """f1, f2: (N, 2048) Inception 特征 -> FID。"""
    return frechet_distance(f1.mean(0), np.cov(f1, rowvar=False), f2.mean(0), np.cov(f2, rowvar=False))


def pick_showcase_rows(labels: np.ndarray, k: int = 8, min_gap: int = 40) -> list[int]:
    """挑 k 张多样的展示图: 男/女 x 年轻/年长 各 2 张, 相互间隔 >= min_gap 行 (尽量不是同一个人)。"""
    m, y = ATTRS.index("Male"), ATTRS.index("Young")
    want = {(0, 1): 2, (1, 1): 2, (0, 0): 2, (1, 0): 2}
    chosen: list[int] = []
    for i, lab in enumerate(labels):
        key = (int(lab[m]), int(lab[y]))
        if want.get(key, 0) > 0 and all(abs(i - j) >= min_gap for j in chosen):
            chosen.append(i)
            want[key] -= 1
        if len(chosen) == k:
            break
    return sorted(chosen)


def genderage_eval(faces: np.ndarray, labels: np.ndarray, edited: dict[str, np.ndarray]) -> dict:
    """用独立的 GenderAge 模型检查性别 / 年龄编辑是否成功。faces/edited 都是 128x128 RGB。"""
    from facevision.attributes import GenderAge
    from facevision.detector import SCRFD

    det = SCRFD(input_size=(256, 256), threads=4)
    ga = GenderAge(threads=4)
    boxes = []
    for f in tqdm(faces, desc="detect (GenderAge bbox)"):
        d, _ = det.detect(np.ascontiguousarray(f[..., ::-1]), max_num=1)
        boxes.append(d[0, :4] if len(d) else None)
    found = [b for b in boxes if b is not None]
    median_box = np.median(np.stack(found), 0)
    boxes = [b if b is not None else median_box for b in boxes]   # 极少数检测失败: 用平均框兜底

    def predict(imgs):
        g, a = [], []
        for img, b in zip(imgs, boxes):
            gi, ai = ga.predict(np.ascontiguousarray(img[..., ::-1]), b)   # 编辑前后几何不变, 复用同一个框
            g.append(gi == "M")
            a.append(ai)
        return np.array(g), np.array(a, np.float64)

    male_gt = labels[:, ATTRS.index("Male")] > 0.5
    young_gt = labels[:, ATTRS.index("Young")] > 0.5
    g_in, a_in = predict(tqdm(faces, desc="GenderAge input"))
    g_ed, _ = predict(tqdm(edited["flip_gender"], desc="GenderAge gender-edit"))
    _, a_ed = predict(tqdm(edited["flip_age"], desc="GenderAge age-edit"))

    res = {"detector_found_face": f"{len(found)}/{len(faces)}",
           "genderage_reliability_on_real_inputs": {
               "gender_acc_vs_celeba_label": round(float((g_in == male_gt).mean()), 4),
               "mean_pred_age_young_label": round(float(a_in[young_gt].mean()), 1),
               "mean_pred_age_not_young_label": round(float(a_in[~young_gt].mean()), 1)}}
    flipped = g_ed != g_in
    res["gender_flip"] = {
        "success_rate_pred_changed": round(float(flipped.mean()), 4),
        "success_rate_matches_target_label": round(float((g_ed == ~male_gt).mean()), 4),
        "male_to_female": {"n": int(male_gt.sum()), "rate": round(float(flipped[male_gt].mean()), 4)},
        "female_to_male": {"n": int((~male_gt).sum()), "rate": round(float(flipped[~male_gt].mean()), 4)},
    }
    d_age = a_ed - a_in
    res["age_flip"] = {
        "young_to_old": {"n": int(young_gt.sum()),
                         "success_rate_age_increased": round(float((d_age[young_gt] > 0).mean()), 4),
                         "success_rate_increase_ge_5y": round(float((d_age[young_gt] >= 5).mean()), 4),
                         "mean_delta_age": round(float(d_age[young_gt].mean()), 2)},
        "old_to_young": {"n": int((~young_gt).sum()),
                         "success_rate_age_decreased": round(float((d_age[~young_gt] < 0).mean()), 4),
                         "success_rate_decrease_ge_5y": round(float((d_age[~young_gt] <= -5).mean()), 4),
                         "mean_delta_age": round(float(d_age[~young_gt].mean()), 2)},
    }
    res["age_flip"]["overall_success_rate"] = round(float(
        np.concatenate([d_age[young_gt] > 0, d_age[~young_gt] < 0]).mean()), 4)
    return res


def lfw_showcase(editor: StarGANEditor, names: list[str]) -> dict:
    """在 LFW 原始照片上做 edit_photo, 拼成展示图; 另存一张 residual / replace 两种贴回方式的对比。"""
    rows, rows_cmp, info_all = [], [], {}
    for name in names:
        p = DATA / "lfw" / name / f"{name}_0001.jpg"
        img = cv2.imread(str(p))
        outs, info = editor.edit_photo_many(img, list(SINGLE_EDITS), return_info=True)
        rows.append([img, *outs])
        rep = editor.edit_photo_many(img, ["blond_hair", "flip_gender"], blend="replace")
        crop = cv2.resize(info["face128"], img.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        rows_cmp.append([img, crop, outs[1], rep[0], outs[3], rep[1]])
        info_all[name] = {"estimated_src_attrs_by_D": info["src_attrs"]}
    save_grid(rows, OUT / "grid_lfw_photos.jpg", ["original", *SINGLE_EDITS])
    save_grid(rows_cmp, OUT / "grid_lfw_blend_compare.jpg",
              ["original", "aligned 128 input", "blond (residual)", "blond (replace)",
               "gender (residual)", "gender (replace)"])
    return info_all


def cpu_timing(names: list[str], threads: int) -> dict:
    """CPU 上的耗时: 只跑 G (1 张 128x128) / 完整的 edit_photo (检测+对齐+D 估计属性+G+贴回)。"""
    torch.set_num_threads(threads)
    ed = StarGANEditor(device="cpu")
    x = torch.zeros(1, 3, 128, 128)
    c = torch.zeros(1, 5)
    with torch.no_grad():
        for _ in range(2):
            ed.G(x, c)
        t0 = time.perf_counter()
        for _ in range(10):
            ed.G(x, c)
        g_ms = (time.perf_counter() - t0) / 10 * 1000
        xb, cb = x.repeat(5, 1, 1, 1), c.repeat(5, 1)
        ed.G(xb, cb)
        t0 = time.perf_counter()
        for _ in range(3):
            ed.G(xb, cb)
        g5_ms = (time.perf_counter() - t0) / 3 * 1000
    img = cv2.imread(str(DATA / "lfw" / names[0] / f"{names[0]}_0001.jpg"))
    ed.edit_photo(img, "blond_hair")          # 预热 (加载 D、SCRFD)
    t0 = time.perf_counter()
    for _ in range(5):
        ed.edit_photo(img, "blond_hair")
    photo_ms = (time.perf_counter() - t0) / 5 * 1000
    t0 = time.perf_counter()
    for _ in range(3):
        ed.edit_photo_many(img, list(SINGLE_EDITS))
    photo5_ms = (time.perf_counter() - t0) / 3 * 1000
    return {"torch_threads": threads,
            "G_forward_1img_ms": round(g_ms, 1),
            "G_forward_batch5_ms": round(g5_ms, 1),
            "edit_photo_single_edit_ms (250x250 LFW, incl. SCRFD+D+G+paste)": round(photo_ms, 1),
            "edit_photo_many_5_edits_ms": round(photo5_ms, 1),
            "note": "测量时 CPU 正被其它预处理任务占用, 数字偏慢, 仅供量级参考"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="评测用的 CelebA test 图片数")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lfw", nargs="*", default=LFW_NAMES)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    n = args.n
    t_start = time.time()

    # ---------- 数据: A = 待编辑的输入, B = 不相交的真实参考集 (test 划分按身份聚集, A/B 基本不同人) ----------
    faces_a, labels_a, names_a = load_faces("test", 0, n)
    faces_b, _, names_b = load_faces("test", n, n)
    print(f"A: {len(faces_a)} ({names_a[0]}...), B: {len(faces_b)} ({names_b[0]}...)")

    editor = StarGANEditor(device=device)
    inc = InceptionFeatures(device)
    feats: dict[str, np.ndarray] = {"real_A": inc(faces_a), "real_B": inc(faces_b)}
    del faces_b

    # ---------- 生成: 5 个单属性编辑 + 重建, 用 CelebA 真实标签作为 c_org (与官方测试一致) ----------
    keep = {"flip_gender": [], "flip_age": []}       # GenderAge 要用的编辑结果
    feat_lists = {k: [] for k in [*SINGLE_EDITS, "recon"]}
    d_correct_real, d_target_hits = np.zeros(5), {k: 0 for k in SINGLE_EDITS}
    l1_sum = 0.0
    for i in tqdm(range(0, n, args.batch), desc="StarGAN translate"):
        x = to_tensor(faces_a[i:i + args.batch])
        c_org = torch.from_numpy(labels_a[i:i + args.batch])
        rec = editor.generate(x, c_org)
        l1_sum += float((rec - x).abs().mean((1, 2, 3)).sum())
        feat_lists["recon"].append(inc(to_image(rec)))
        d_correct_real += ((editor.classify(x) > 0.5).float() == c_org).float().sum(0).numpy()
        for name, c_trg in zip(SINGLE_EDITS, official_test_labels(c_org)):
            out = editor.generate(x, c_trg)
            imgs = to_image(out)
            feat_lists[name].append(inc(imgs))
            if name in keep:
                keep[name].append(imgs)
            # D 的辅助分类器认为"目标属性"达成了吗? (G 就是对着这个 D 训练的, 所以偏乐观)
            j = {"black_hair": 0, "blond_hair": 1, "brown_hair": 2, "flip_gender": 3, "flip_age": 4}[name]
            d_target_hits[name] += int(((editor.classify(out)[:, j] > 0.5).float() == c_trg[:, j]).sum())
    for k, v in feat_lists.items():
        feats[k] = np.concatenate(v)
    edited = {k: np.concatenate(v) for k, v in keep.items()}

    # ---------- 重建误差: 正确做法 (InstanceNorm 用单图统计量) vs 错误做法 (G.eval() 用 running stats) ----------
    g_bad = load_generator(device=device).eval()
    m = min(n, 200)
    with torch.no_grad():
        x = to_tensor(faces_a[:m])
        c = torch.from_numpy(labels_a[:m])
        bad = torch.cat([g_bad(x[i:i + 25].to(device), c[i:i + 25].to(device)).cpu() for i in range(0, m, 25)])
    good = torch.cat([editor.generate(x[i:i + 25], c[i:i + 25]) for i in range(0, m, 25)])
    l1_bad, l1_good = float((bad - x).abs().mean()), float((good - x).abs().mean())
    del g_bad

    # ---------- 目标域参考集: 真实的黑/金/棕发照片 (不含 A) ----------
    hair_ref_n = {}
    for h in HAIR_ATTRS:
        f, _, _ = load_faces("test", n, n, attr_filter=h)
        feats[f"real_{h}"] = inc(f)
        hair_ref_n[h] = len(f)

    # ---------- FID ----------
    res_fid = {"real_A_vs_real_B (baseline: same distribution, finite-sample floor)": round(fid(feats["real_A"], feats["real_B"]), 2),
               "recon_vs_real_B": round(fid(feats["recon"], feats["real_B"]), 2)}
    for name in SINGLE_EDITS:
        res_fid[f"{name}_vs_real_B"] = round(fid(feats[name], feats["real_B"]), 2)
    fid_domain = {}
    for name, h in zip(["black_hair", "blond_hair", "brown_hair"], HAIR_ATTRS):
        fid_domain[name] = {"fid_edit_vs_real_target_domain": round(fid(feats[name], feats[f"real_{h}"]), 2),
                            "fid_real_A_vs_real_target_domain (reference)": round(fid(feats["real_A"], feats[f"real_{h}"]), 2),
                            "n_ref": hair_ref_n[h]}
    print(json.dumps(res_fid, indent=2))

    # ---------- GenderAge 成功率 ----------
    ga_res = genderage_eval(faces_a, labels_a, edited)

    # ---------- 展示图 ----------
    rows_idx = pick_showcase_rows(labels_a)
    grid_rows = []
    for i in rows_idx:
        outs = editor.edit_many(faces_a[i], list(SINGLE_EDITS), src_attrs=labels_a[i], bgr=False)
        grid_rows.append([faces_a[i][..., ::-1]] + [o[..., ::-1] for o in outs])
    save_grid(grid_rows, OUT / "grid_celeba_test.jpg", ["input", *SINGLE_EDITS])
    lfw_info = lfw_showcase(editor, args.lfw)

    result = {
        "model": "official StarGAN CelebA 128x128 (yunjey/stargan, 200000 iters)",
        "data": {"split": "CelebA official test split (parquet file order)", "n_inputs_A": n,
                 "n_reference_B": n, "A_first_last": [names_a[0], names_a[-1]],
                 "B_first_last": [names_b[0], names_b[-1]],
                 "c_org": "CelebA ground-truth labels (as in the official test code)"},
        "fid": res_fid,
        "fid_hair_vs_target_domain": fid_domain,
        "reconstruction": {
            "L1_G(x,c_org)_vs_x_[-1,1]": round(l1_sum / n, 4),
            "mean_abs_error_0_255": round(l1_sum / n * 127.5, 2),
            f"L1_if_G.eval()_uses_running_stats_(first {m})": round(l1_bad, 4),
            f"L1_instance_stats_(first {m}, same images)": round(l1_good, 4)},
        "aux_classifier_D": {
            "acc_on_real_inputs_per_attr": {a: round(float(v / n), 4) for a, v in zip(ATTRS, d_correct_real)},
            "target_attr_reached_on_edits (optimistic: G was trained against this D)":
                {k: round(v / n, 4) for k, v in d_target_hits.items()}},
        "genderage_independent_check": ga_res,
        "lfw_photos": lfw_info,
        "device": str(device),
        "eval_seconds": round(time.time() - t_start, 1),
    }
    result["cpu_timing"] = cpu_timing(args.lfw, args.threads)
    (OUT / "eval_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
