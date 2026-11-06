"""加载自己训练的 PyTorch 模型, 提取人脸特征 (接口与 ArcFaceONNX.embed 一致)。"""
from __future__ import annotations

import numpy as np
import torch

from .backbones import get_backbone
from .data import to_tensor


def load_backbone(ckpt_path, device="cpu") -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = get_backbone(ckpt["args"]["backbone"])
    model.load_state_dict(ckpt["backbone"])
    return model.to(device).eval(), ckpt


class TorchEmbedder:
    def __init__(self, ckpt_path, device="cpu"):
        self.device = device
        self.model, self.ckpt = load_backbone(ckpt_path, device)

    @torch.no_grad()
    def __call__(self, faces, flip: bool = True) -> np.ndarray:
        x = to_tensor(np.stack(faces), self.device)
        feats = self.model(x)
        if flip:
            feats = feats + self.model(x.flip(-1))
        return torch.nn.functional.normalize(feats.float(), dim=1).cpu().numpy()
