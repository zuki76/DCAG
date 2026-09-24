import os
from typing import List, Tuple

import torch
import torch.nn.functional as F

from transformers import ViTModel, ViTConfig


class CandleFusionViTEncoder(torch.nn.Module):
    """Frozen CandleFusion ViT returning one (B, N, D) feature tensor per layer."""

    def __init__(
        self,
        fin_dir: str,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        image_size: int = 224,
        input_norm: str = "01",
    ):
        super().__init__()
        self.fin_dir = fin_dir
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self.input_norm = input_norm

        cfg_path = os.path.join(fin_dir, "config.json")
        if os.path.exists(cfg_path):
            vit_cfg = ViTConfig.from_pretrained(fin_dir, local_files_only=True)
        else:
            raise FileNotFoundError(
                f"CandleFusion ViT configuration not found: {cfg_path}"
            )

        self.vit = ViTModel(vit_cfg).to(device=self.device, dtype=self.dtype)
        self.vit.eval()

        ckpt_path = os.path.join(fin_dir, "pytorch_model.bin")

        sd = torch.load(ckpt_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]

        vit_sd = {k[len("vit.") :]: v for k, v in sd.items() if k.startswith("vit.")}
        assert len(vit_sd) > 0, "No vit.* keys found in checkpoint!"

        missing, unexpected = self.vit.load_state_dict(vit_sd, strict=False)

        if len(vit_sd) == 0:
            missing, unexpected = self.vit.load_state_dict(sd, strict=False)

        self.feature_dim = self.vit.config.hidden_size
        self.num_layers = self.vit.config.num_hidden_layers

    @staticmethod
    def _extract_sub_state_dict(state_dict: dict, prefixes: Tuple[str, ...]) -> dict:
        out = {}
        for k, v in state_dict.items():
            for p in prefixes:
                if k.startswith(p):
                    out[k[len(p) :]] = v
                    break
        return out

    def _runtime_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        ref = next(self.vit.parameters())
        return ref.device, ref.dtype

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert the configured input normalization to ViT pixel values.

        DCAG supplies images in [0, 1] with input_norm="01".
        """

        if self.input_norm == "none":
            return x

        if self.input_norm == "clip":
            mean = torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            std = torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            x01 = x * std + mean
        elif self.input_norm == "01":
            x01 = x
        elif self.input_norm == "imagenet":
            mean_im = torch.tensor(
                [0.485, 0.456, 0.406], device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            std_im = torch.tensor(
                [0.229, 0.224, 0.225], device=x.device, dtype=x.dtype
            ).view(1, 3, 1, 1)
            x01 = x * std_im + mean_im
        else:
            raise ValueError(f"Unknown input_norm={self.input_norm}")

        x01 = x01.clamp(0.0, 1.0)

        mean_vit = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        std_vit = torch.tensor([0.5, 0.5, 0.5], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        out = (x01 - mean_vit) / std_vit

        return out

    @torch.no_grad()
    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Return per-layer token features for images shaped (B, 3, H, W)."""
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        x = images.to(device=runtime_device, dtype=runtime_dtype)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x.float(),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=runtime_dtype)

        x = self._normalize(x)

        out = self.vit(pixel_values=x, output_hidden_states=True, return_dict=True)
        hs = out.hidden_states
        layer_tokens = [
            h.to(device=runtime_device, dtype=runtime_dtype) for h in hs[1:]
        ]
        return layer_tokens
