import math
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def split_mosaic_2x3(x: torch.Tensor) -> List[torch.Tensor]:
    B, C, H, W = x.shape
    assert H % 2 == 0 and W % 3 == 0, (
        f"mosaic size must be divisible by (2,3), got {(H, W)}"
    )
    h2, w3 = H // 2, W // 3
    views: List[torch.Tensor] = []
    for r in range(2):
        for c in range(3):
            views.append(x[:, :, r * h2 : (r + 1) * h2, c * w3 : (c + 1) * w3])
    return views


def _unwrap_state_dict(obj) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")
    for k in ("state_dict", "model", "model_state", "model_state_dict"):
        if k in obj and isinstance(obj[k], dict):
            obj = obj[k]
            break
    return {kk: vv for kk, vv in obj.items() if isinstance(vv, torch.Tensor)}


def _extract_backbone_subdict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    keys = list(sd.keys())
    cand_prefixes = (
        "module.img_backbone.",
        "img_backbone.",
        "module.backbone.",
        "backbone.",
        "module.",
    )
    best_prefix = ""
    best_cnt = -1
    for p in cand_prefixes:
        cnt = sum(1 for k in keys if k.startswith(p))
        if cnt > best_cnt:
            best_cnt = cnt
            best_prefix = p

    if best_cnt <= 0:
        return sd

    return {
        k[len(best_prefix) :]: v for k, v in sd.items() if k.startswith(best_prefix)
    }


def _remap_mmcv_to_timm_swin_keys(
    sd: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    def repl(k: str) -> str:
        k2 = k
        k2 = k2.replace("stages.", "layers.")
        k2 = k2.replace("patch_embed.projection.", "patch_embed.proj.")
        k2 = k2.replace(".attn.w_msa.", ".attn.")
        k2 = k2.replace(".ffn.layers.0.0.", ".mlp.fc1.")
        k2 = k2.replace(".ffn.layers.1.", ".mlp.fc2.")
        k2 = k2.replace(".ffn.", ".mlp.")
        return k2

    return {repl(k): v for k, v in sd.items()}


def _interp_rel_pos_bias_table(
    src: torch.Tensor, dst_shape: Tuple[int, int]
) -> torch.Tensor:
    L_new, nH_new = dst_shape
    L_old, nH_old = src.shape
    if nH_old != nH_new:
        return src

    s_old = int(math.sqrt(L_old))
    s_new = int(math.sqrt(L_new))
    if s_old * s_old != L_old or s_new * s_new != L_new:
        return src

    src_2d = src.permute(1, 0).contiguous().view(1, nH_old, s_old, s_old)
    src_2d = F.interpolate(
        src_2d, size=(s_new, s_new), mode="bicubic", align_corners=False
    )
    out = src_2d.view(nH_old, s_new * s_new).permute(1, 0).contiguous()
    return out


def _fix_and_filter_swin_state_dict_for_timm(
    model: nn.Module, sd: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    model_sd = model.state_dict()
    out: Dict[str, torch.Tensor] = {}

    for k, v in sd.items():
        if k.endswith("relative_position_index"):
            continue
        if k not in model_sd:
            continue

        tgt = model_sd[k]
        if v.shape == tgt.shape:
            out[k] = v
            continue

        if (
            k.endswith("relative_position_bias_table")
            and v.dim() == 2
            and tgt.dim() == 2
        ):
            v2 = _interp_rel_pos_bias_table(v, (tgt.shape[0], tgt.shape[1]))
            if v2.shape == tgt.shape:
                out[k] = v2
                continue

    return out


class GeoMIMSwinBackbone(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        dtype: torch.dtype,
        img_size: int = 224,
        num_return_layers: int = 12,
        input_norm: str = "clip",
        verbose: bool = True,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.device = device
        self.dtype = dtype
        self.img_size = img_size
        self.num_return_layers = num_return_layers
        self.input_norm = input_norm
        self.verbose = verbose
        self.ad_variant = str(os.getenv("AD_FEATURE_VARIANT", "A0")).strip().upper()
        self.view_topk_local = int(os.getenv("AD_VIEW_TOPK_LOCAL", "4"))

        from timm.models.swin_transformer import SwinTransformer

        prev_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float32)
            self.swin = SwinTransformer(
                img_size=224,
                patch_size=4,
                in_chans=3,
                num_classes=1000,
                embed_dim=128,
                depths=(2, 2, 18, 2),
                num_heads=(4, 8, 16, 32),
                window_size=14,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_rate=0.0,
                attn_drop_rate=0.0,
                drop_path_rate=0.2,
                ape=False,
                patch_norm=True,
                use_checkpoint=False,
            )
        finally:
            torch.set_default_dtype(prev_default_dtype)

        self.swin.to(device=self.device)
        if self.dtype != torch.float32:
            self.swin.to(dtype=self.dtype)
        self.swin.eval()

        self.feature_dim = 512

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        sd = _unwrap_state_dict(ckpt)
        bb = _extract_backbone_subdict(sd)

        bb_direct = _fix_and_filter_swin_state_dict_for_timm(self.swin, bb)
        missing_a, unexpected_a = self.swin.load_state_dict(bb_direct, strict=False)
        need_remap = ("patch_embed.proj.weight" in missing_a) or (len(bb_direct) < 300)

        if need_remap:
            bb2 = _remap_mmcv_to_timm_swin_keys(bb)
            bb2 = _fix_and_filter_swin_state_dict_for_timm(self.swin, bb2)
            missing_b, unexpected_b = self.swin.load_state_dict(bb2, strict=False)
            if self.verbose:
                ign_prefix = ("head.",)
                ign_suffix = ("relative_position_index", "attn_mask")
                filtered = [
                    k
                    for k in missing_b
                    if not k.startswith(ign_prefix) and not k.endswith(ign_suffix)
                ]
                print(f"[GeoMIM][timm] missing (filtered) = {len(filtered)}")
                if len(filtered) > 0:
                    print("[GeoMIM][timm] filtered missing sample:", filtered[:50])
        else:
            if self.verbose:
                print(
                    f"[GeoMIM][timm] load direct: loaded={len(bb_direct)} "
                    f"missing={len(missing_a)} unexpected={len(unexpected_a)}"
                )

        if self.verbose:
            try:
                w = self.swin.patch_embed.proj.weight.detach().float().cpu()
                print(
                    f"[GeoMIM][timm] patch_embed.proj.weight mean={w.mean():.6f} std={w.std():.6f}"
                )
            except Exception:
                pass

        assert hasattr(self.swin, "layers")
        assert len(self.swin.layers) == 4

        stage3 = self.swin.layers[2]
        assert hasattr(stage3, "blocks")
        blocks = list(stage3.blocks)
        assert len(blocks) >= self.num_return_layers

        self._need_stage2 = self.ad_variant in {"A2", "A3"}
        self._need_stage4 = self.ad_variant in {"A3"}

        if self._need_stage2:
            stage2 = self.swin.layers[1]
            blocks2 = list(stage2.blocks)
            assert len(blocks2) >= 2
            self._stage2_pick_idx = list(range(len(blocks2) - 2, len(blocks2)))
        else:
            blocks2 = []
            self._stage2_pick_idx = []

        if self._need_stage4:
            stage4 = self.swin.layers[3]
            blocks4 = list(stage4.blocks)
            assert len(blocks4) >= 2
            self._stage4_pick_idx = list(range(len(blocks4) - 2, len(blocks4)))
        else:
            blocks4 = []
            self._stage4_pick_idx = []

        self._stage3_pick_idx = list(
            range(len(blocks) - self.num_return_layers, len(blocks))
        )
        self._stage2_block_outs: List[torch.Tensor] = []
        self._stage3_block_outs: List[torch.Tensor] = []
        self._stage4_block_outs: List[torch.Tensor] = []
        self._hooks = []
        for bi in self._stage2_pick_idx:
            self._hooks.append(
                blocks2[bi].register_forward_hook(self._hook_stage2_block)
            )
        for bi in self._stage3_pick_idx:
            self._hooks.append(
                blocks[bi].register_forward_hook(self._hook_stage3_block)
            )
        for bi in self._stage4_pick_idx:
            self._hooks.append(
                blocks4[bi].register_forward_hook(self._hook_stage4_block)
            )

    def _runtime_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        ref = next(self.swin.parameters())
        return ref.device, ref.dtype

    def _hook_stage2_block(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self._stage2_block_outs.append(out)

    def _hook_stage3_block(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self._stage3_block_outs.append(out)

    def _hook_stage4_block(self, module, inp, out):
        if isinstance(out, torch.Tensor):
            self._stage4_block_outs.append(out)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
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
        else:
            raise ValueError(f"Unknown input_norm={self.input_norm}")

        mean_im = torch.tensor(
            [0.485, 0.456, 0.406], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        std_im = torch.tensor(
            [0.229, 0.224, 0.225], device=x.device, dtype=x.dtype
        ).view(1, 3, 1, 1)
        return (x01 - mean_im) / std_im

    @torch.no_grad()
    def _to_tokens(
        self, outs: List[torch.Tensor], expected_dim: int, stage_name: str
    ) -> List[torch.Tensor]:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        layers: List[torch.Tensor] = []
        for t in outs:
            if t.dim() == 4:
                if t.shape[-1] == expected_dim:
                    t = t.reshape(t.shape[0], -1, t.shape[-1])
                else:
                    t = t.flatten(2).transpose(1, 2)
            if t.dim() != 3:
                raise RuntimeError(
                    f"Unexpected {stage_name} output shape: {tuple(t.shape)}"
                )
            if t.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"Expected {stage_name} channel={expected_dim}, got {t.shape[-1]}"
                )
            layers.append(t.to(device=runtime_device, dtype=runtime_dtype))
        return layers

    def _stage2_to_512(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] != 256:
            raise RuntimeError(f"Expected stage2 tensor dim=256, got {t.shape[-1]}")
        return torch.cat([t, t], dim=-1)

    def _stage4_to_512(self, t: torch.Tensor) -> torch.Tensor:
        if t.shape[-1] != 1024:
            raise RuntimeError(f"Expected stage4 tensor dim=1024, got {t.shape[-1]}")
        return t[..., :512]

    def _select_local_topk(self, t: torch.Tensor) -> torch.Tensor:
        k = min(int(self.view_topk_local), t.size(1))
        if k <= 0 or k >= t.size(1):
            return t
        score = t.norm(dim=-1)
        idx = score.topk(k, dim=1).indices
        return t.gather(1, idx.unsqueeze(-1).expand(-1, -1, t.size(-1)))

    @torch.no_grad()
    def _forward_one_view(self, view: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        v = view.to(device=runtime_device, dtype=runtime_dtype)
        v = F.interpolate(
            v.float(),
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=runtime_dtype)
        v = self._normalize(v)

        self._stage2_block_outs = []
        self._stage3_block_outs = []
        self._stage4_block_outs = []
        _ = self.swin(v)

        outs2 = self._stage2_block_outs
        outs3 = self._stage3_block_outs
        if len(outs3) != self.num_return_layers:
            raise RuntimeError(
                f"Expected {self.num_return_layers} stage3 outputs, got {len(outs3)}"
            )
        if self._need_stage2 and len(outs2) != 2:
            raise RuntimeError(f"Expected 2 stage2 outputs, got {len(outs2)}")
        if self._need_stage4 and len(self._stage4_block_outs) != 2:
            raise RuntimeError(
                f"Expected 2 stage4 outputs, got {len(self._stage4_block_outs)}"
            )

        out = {
            "stage2": self._to_tokens(outs2, expected_dim=256, stage_name="stage2"),
            "stage3": self._to_tokens(outs3, expected_dim=512, stage_name="stage3"),
        }
        if self._need_stage4:
            out["stage4"] = self._to_tokens(
                self._stage4_block_outs, expected_dim=1024, stage_name="stage4"
            )
        return out

    def _collect_variant_layers(
        self, per_view_outputs: Dict[str, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        out_layers: List[torch.Tensor] = []

        if self.ad_variant == "A0":
            out_layers.extend(per_view_outputs["stage3"])
            return out_layers

        if self.ad_variant == "A1":
            out_layers.extend(
                self._select_local_topk(layer) for layer in per_view_outputs["stage3"]
            )
            return out_layers

        if self.ad_variant == "A2":
            for li in range(2):
                out_layers.append(
                    self._select_local_topk(
                        self._stage2_to_512(per_view_outputs["stage2"][li])
                    )
                )
            for li in range(10):
                out_layers.append(
                    self._select_local_topk(per_view_outputs["stage3"][li + 2])
                )
            return out_layers

        if self.ad_variant == "A3":
            for li in range(2):
                out_layers.append(
                    self._select_local_topk(
                        self._stage2_to_512(per_view_outputs["stage2"][li])
                    )
                )
            for li in range(8):
                out_layers.append(
                    self._select_local_topk(per_view_outputs["stage3"][li + 4])
                )
            for li in range(2):
                out_layers.append(
                    self._select_local_topk(
                        self._stage4_to_512(per_view_outputs["stage4"][li])
                    )
                )
            return out_layers

        raise ValueError(f"Unknown AD_FEATURE_VARIANT={self.ad_variant}")

    @torch.no_grad()
    def extract_layer_tokens_by_view_from_mosaic(
        self, mosaic: torch.Tensor
    ) -> List[torch.Tensor]:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        mosaic = mosaic.to(device=runtime_device, dtype=runtime_dtype)
        views = split_mosaic_2x3(mosaic)
        collected_by_view = [
            self._collect_variant_layers(self._forward_one_view(view)) for view in views
        ]
        out_layers: List[torch.Tensor] = []
        for layer_idx in range(self.num_return_layers):
            out_layers.append(
                torch.stack(
                    [view_layers[layer_idx] for view_layers in collected_by_view], dim=1
                )
            )
        return out_layers

    @torch.no_grad()
    def extract_layer_tokens_from_mosaic(
        self, mosaic: torch.Tensor
    ) -> List[torch.Tensor]:
        structured_layers = self.extract_layer_tokens_by_view_from_mosaic(mosaic)
        return [
            layer.reshape(
                layer.shape[0], layer.shape[1] * layer.shape[2], layer.shape[3]
            )
            for layer in structured_layers
        ]

    @torch.no_grad()
    def extract_layer_tokens_by_view(self, images: torch.Tensor) -> List[torch.Tensor]:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        images = images.to(device=runtime_device, dtype=runtime_dtype)
        if images.dim() == 5:
            batch_size, num_views, channels, height, width = images.shape
            if num_views != 6:
                raise ValueError(f"Expected 6 AD views, got {num_views}")
            top_row = torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=-1)
            bottom_row = torch.cat([images[:, 3], images[:, 4], images[:, 5]], dim=-1)
            images = torch.cat([top_row, bottom_row], dim=-2)
        return self.extract_layer_tokens_by_view_from_mosaic(images)

    @torch.no_grad()
    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        return self.extract_layer_tokens_from_mosaic(images)

    def __del__(self):
        try:
            for h in getattr(self, "_hooks", []):
                h.remove()
        except Exception:
            pass
