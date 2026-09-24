from __future__ import annotations
from typing import List, Literal, Tuple

import torch
import torch.nn as nn

from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor


InputNorm = Literal["none", "clip"]


class Pix2StructVisionBackbone(nn.Module):
    """Frozen Pix2Struct vision encoder.

    Accepts (B, 3, H, W) images and returns per-layer (B, N, out_dim) features.
    """

    def __init__(
        self,
        model_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        num_return_layers: int = 12,
        out_dim: int = 768,
        input_norm: InputNorm = "none",
        local_files_only: bool = True,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.num_return_layers = num_return_layers
        self.out_dim = out_dim
        self.input_norm = input_norm

        self.processor = Pix2StructProcessor.from_pretrained(
            model_dir, local_files_only=local_files_only
        )

        full = Pix2StructForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        )

        self.vision = full.encoder
        self.vision.eval()

        vision_cfg = getattr(full.config, "vision_config", None)
        if vision_cfg is None:
            hidden_size = self.vision.config.hidden_size
            num_layers = self.vision.config.num_hidden_layers
        else:
            hidden_size = vision_cfg.hidden_size
            num_layers = vision_cfg.num_hidden_layers

        self._vision_hidden = int(hidden_size)
        self._vision_layers = int(num_layers)

        if self._vision_hidden != out_dim:
            raise ValueError(
                f"Expected Pix2Struct feature width {out_dim}, got {self._vision_hidden}"
            )
        else:
            self.proj = nn.Identity()

        for p in self.parameters():
            p.requires_grad = False

        self.to(device=device, dtype=dtype)

        self.feature_dim = out_dim

    def _runtime_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        ref = next(self.vision.parameters())
        return ref.device, ref.dtype

    def _maybe_unnormalize_clip(self, x: torch.Tensor) -> torch.Tensor:
        """Undo CLIP normalization and clamp pixel values to [0, 1]."""
        if self.input_norm != "clip":
            return x

        mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        x = x * std + mean
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        """Return num_return_layers token tensors, each shaped (B, N, out_dim)."""
        if images.ndim != 4:
            raise ValueError(f"Expected (B,3,H,W), got {tuple(images.shape)}")

        runtime_device, runtime_dtype = self._runtime_device_dtype()

        x = images.to(device=runtime_device, dtype=runtime_dtype)
        x = self._maybe_unnormalize_clip(x)

        imgs = [t.detach().float().cpu() for t in x]

        proc = self.processor(images=imgs, return_tensors="pt")
        flattened_patches = proc["flattened_patches"].to(
            device=runtime_device, dtype=runtime_dtype
        )

        attention_mask = proc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=runtime_device)

        out = self.vision(
            flattened_patches=flattened_patches,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = out.hidden_states

        if hs is None:
            raise RuntimeError(
                "Pix2Struct vision did not return hidden_states. Please check transformers version."
            )

        take = min(self.num_return_layers, len(hs) - 1)
        selected = list(hs[-take:])

        feat_list: List[torch.Tensor] = []
        for h in selected:
            h = h.to(device=runtime_device, dtype=runtime_dtype)
            feat_list.append(h)

        if take < self.num_return_layers:
            pad = [feat_list[0]] * (self.num_return_layers - take)
            feat_list = pad + feat_list

        return feat_list
