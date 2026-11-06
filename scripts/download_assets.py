"""下载项目所需的预训练模型与数据集（只依赖 Python 标准库）。

用法:
    python scripts/download_assets.py            # 下载全部
    python scripts/download_assets.py models     # 只下载模型
    python scripts/download_assets.py lfw wider  # 只下载指定条目

所有文件放到 weights/ 和 data/ 下，这两个目录不进 git。
"""
from __future__ import annotations

import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"

# (分组, 名称, URL, 保存路径, 是否解压到的目录)
ASSETS = [
    # ---------- 模型 ----------
    # InsightFace buffalo_l: SCRFD-10G 检测 + 2d106 关键点 + ArcFace R50(WebFace600K) 识别
    ("models", "buffalo_l",
     "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
     WEIGHTS / "buffalo_l.zip", WEIGHTS / "buffalo_l"),
    # MediaPipe Face Landmarker: 478 个 3D 关键点 + 52 个表情系数 + 头部姿态矩阵
    ("models", "face_landmarker",
     "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
     WEIGHTS / "face_landmarker.task", None),
    # StarGAN 官方 CelebA 128x128 五属性预训练模型 (黑/金/棕发, 性别, 年龄)
    ("models", "stargan",
     "https://www.dropbox.com/s/7e966qq0nlxwte4/celeba-128x128-5attrs.zip?dl=1",
     WEIGHTS / "stargan_celeba_128.zip", WEIGHTS / "stargan"),
    # 3DDFA_V2 (MIT License): 3DMM 参数回归网络 + BFM 形状模型
    ("models", "3ddfa_v2",
     "https://github.com/cleardusk/3DDFA_V2/archive/refs/heads/master.zip",
     WEIGHTS / "3ddfa_v2.zip", WEIGHTS / "3ddfa_v2"),
    # OpenCV 5 不再自带 Haar 级联文件, 用于和深度学习检测器做对比
    ("models", "haar",
     "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml",
     WEIGHTS / "haarcascade_frontalface_default.xml", None),
    # ---------- 数据集 ----------
    # LFW 原图 + 官方 10 折 6000 对 pairs.txt (sklearn 使用的 figshare 镜像)
    ("lfw", "lfw_images", "https://ndownloader.figshare.com/files/5976018",
     DATA / "lfw.tgz", DATA / "lfw"),
    ("lfw", "lfw_pairs", "https://ndownloader.figshare.com/files/5976006",
     DATA / "lfw" / "pairs.txt", None),
    # WIDER FACE 验证集 + 标注 + 官方评测用 ground truth (.mat)
    ("wider", "wider_val",
     "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/WIDER_val.zip",
     DATA / "WIDER_val.zip", DATA / "widerface"),
    ("wider", "wider_split",
     "https://huggingface.co/datasets/CUHK-CSE/wider_face/resolve/main/data/wider_face_split.zip",
     DATA / "wider_face_split.zip", DATA / "widerface"),
    ("wider", "wider_gt",
     "https://github.com/biubug6/Pytorch_Retinaface/archive/refs/heads/master.zip",
     DATA / "pytorch_retinaface.zip", DATA / "widerface" / "_retinaface_repo"),
]


def _progress(block, block_size, total):
    if block % 200:          # 每 200 个块刷新一次, 避免刷屏
        return
    done = block * block_size
    if total > 0:
        pct = min(100.0, done * 100.0 / total)
        sys.stdout.write(f"\r    {done / 1e6:8.1f} / {total / 1e6:.1f} MB  {pct:5.1f}%")
    else:
        sys.stdout.write(f"\r    {done / 1e6:8.1f} MB")
    sys.stdout.flush()


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  已存在, 跳过: {dest.relative_to(ROOT)}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    print(f"  下载 {url}")
    urllib.request.urlretrieve(url, tmp, _progress)
    print()
    tmp.replace(dest)


def extract(archive: Path, out_dir: Path) -> None:
    marker = out_dir / f".extracted_{archive.name}"
    if marker.exists():
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  解压到 {out_dir.relative_to(ROOT)}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(out_dir, filter="data")
    # 压缩包里只有一层同名目录时, 把内容提上来
    children = [p for p in out_dir.iterdir() if not p.name.startswith(".")]
    if len(children) == 1 and children[0].is_dir() and children[0].name in (out_dir.name, "lfw"):
        inner = children[0]
        for item in inner.iterdir():
            shutil.move(str(item), out_dir / item.name)
        inner.rmdir()
    marker.touch()


def main(selected: list[str]) -> None:
    for group, name, url, dest, out_dir in ASSETS:
        if selected and group not in selected and name not in selected:
            continue
        print(f"[{group}] {name}")
        download(url, dest)
        if out_dir is not None:
            extract(dest, out_dir)
    print("完成。")


if __name__ == "__main__":
    main(sys.argv[1:])
