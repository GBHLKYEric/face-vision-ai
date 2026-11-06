"""从 Hugging Face 下载 CelebA (对齐图 + 身份 ID + 40 个属性 + 官方划分)。

官方 Google Drive 经常限流, 这里用 flwrlabs/celeba 的 parquet 镜像 (约 11.7GB)。
CelebA 的 10177 个身份与 LFW 的人物互不重叠 (DeepID2 论文构建 CelebFaces+ 时特意去重),
所以用它训练、在 LFW 上测试是干净的协议。
"""
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    path = snapshot_download(
        repo_id="flwrlabs/celeba", repo_type="dataset",
        allow_patterns=["img_align+identity+attr/*.parquet"],
        local_dir=ROOT / "data" / "celeba" / "hf", max_workers=4)
    print("saved to", path)
