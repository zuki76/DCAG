"""Domain-conditioned adapter generation and continual preservation."""

import json
import math
import os
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


DCAG_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

DCAG_DOMAIN_TO_ID = {
    "RS": 0,
    "Med": 1,
    "AD": 2,
    "Sci": 3,
    "Fin": 4,
}

DCAG_ID_TO_DOMAIN = {v: k for k, v in DCAG_DOMAIN_TO_ID.items()}


_LLAMA_7B_HIDDEN = 4096
_LLAMA_7B_INTERMEDIATE = 11008


NUM_LLM_SLOTS = 32 * len(DCAG_TARGET_MODULES)
NUM_PROJECTOR_SLOTS = 2
NUM_TOTAL_SLOTS = NUM_LLM_SLOTS + NUM_PROJECTOR_SLOTS
PROJECTOR_SLOT_OFFSET = NUM_LLM_SLOTS


def _llm_slot_id(layer_idx: int, target_idx: int) -> int:
    return layer_idx * len(DCAG_TARGET_MODULES) + target_idx


def _projector_slot_id(projector_idx: int) -> int:
    return PROJECTOR_SLOT_OFFSET + projector_idx


_WARNED_OPTIONS: set = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_OPTIONS:
        return
    _WARNED_OPTIONS.add(key)
    warnings.warn(message, stacklevel=2)


DCAG_DOMAIN_PATH_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "rs_ckpt_path": ("checkpoints", "domains", "rs", "vit-b-checkpoint-1599.pth"),
    "med_ckpt_path": ("checkpoints", "domains", "pubmedclip"),
    "ad_ckpt_path": ("checkpoints", "domains", "ad", "geomim_base.pth"),
    "sci_model_dir": ("checkpoints", "domains", "pix2struct-base"),
    "fin_model_dir": ("checkpoints", "domains", "candlefusion"),
}


def resolve_dcag_repo_path(
    configured_path: Optional[str],
    repo_relative_parts: Sequence[str],
    key: str,
) -> Optional[str]:
    if configured_path is None:
        return None
    configured_text = str(configured_path)
    if os.path.exists(os.path.expanduser(configured_text)):
        return configured_text

    fallback = os.path.join(REPO_ROOT, *repo_relative_parts)
    if os.path.exists(fallback):
        print(
            f"[DCAG][path] override {key}: {configured_text} -> {fallback}",
            flush=True,
        )
        return fallback
    return configured_text


def normalize_dcag_config_paths(
    config_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Normalize DCAG domain-model paths in a mutable config dict."""
    if not isinstance(config_dict, dict):
        return config_dict
    for key, rel_parts in DCAG_DOMAIN_PATH_FALLBACKS.items():
        if key in config_dict:
            config_dict[key] = resolve_dcag_repo_path(
                config_dict.get(key), rel_parts, key
            )
    return config_dict


def _unwrap_checkpoint_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module", "encoder"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _strip_state_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _select_matching_rs_state(
    raw_state: Dict[str, torch.Tensor],
    model_state: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], str, int, int, float, List[str], List[str]]:
    candidates: List[Tuple[str, Dict[str, torch.Tensor]]] = [("as_is", raw_state)]
    for prefix in ("module.", "model.", "encoder.", "backbone.", "module.model."):
        if any(isinstance(k, str) and k.startswith(prefix) for k in raw_state.keys()):
            candidates.append(
                (f"strip_{prefix.rstrip('.')}", _strip_state_prefix(raw_state, prefix))
            )

    total_params = float(sum(v.numel() for v in model_state.values()))
    best: Optional[
        Tuple[Dict[str, torch.Tensor], str, int, int, float, List[str], List[str]]
    ] = None
    best_numel = -1

    for label, candidate in candidates:
        matched: Dict[str, torch.Tensor] = {}
        unexpected: List[str] = []
        for key, value in candidate.items():
            ref = model_state.get(key)
            if (
                ref is not None
                and hasattr(value, "shape")
                and tuple(value.shape) == tuple(ref.shape)
            ):
                matched[key] = value
            else:
                unexpected.append(key)
        missing = [key for key in model_state.keys() if key not in matched]
        matched_numel = int(sum(v.numel() for v in matched.values()))
        ratio = matched_numel / max(total_params, 1.0)
        record = (
            matched,
            label,
            len(matched),
            matched_numel,
            ratio,
            missing,
            unexpected,
        )
        if matched_numel > best_numel:
            best = record
            best_numel = matched_numel

    if best is None:
        raise RuntimeError(
            "RS checkpoint state dict did not contain any loadable tensor"
        )
    return best


def _parse_optional_float_sequence(value: Optional[Any]) -> Optional[Tuple[float, ...]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null"}:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            value = [item.strip() for item in text.split(",") if item.strip()]
    return tuple(float(x) for x in value)


def _parse_optional_int_sequence(value: Optional[Any]) -> Optional[Tuple[int, ...]]:
    parsed = _parse_optional_float_sequence(value)
    if parsed is None:
        return None
    return tuple(max(1, int(x)) for x in parsed)


def _safe_qr(x: torch.Tensor) -> torch.Tensor:
    """Compute QR in float32 for half-precision inputs; return Q in the input dtype."""
    orig_dtype = x.dtype
    if orig_dtype in (torch.bfloat16, torch.float16):
        q, _ = torch.linalg.qr(x.float())
        return q.to(dtype=orig_dtype)
    q, _ = torch.linalg.qr(x)
    return q


def _safe_svd_U(x: torch.Tensor) -> torch.Tensor:
    """Compute SVD in float32 for half-precision inputs; return U in the input dtype."""
    orig_dtype = x.dtype
    if orig_dtype in (torch.bfloat16, torch.float16):
        U_svd, _, _ = torch.linalg.svd(x.float())
        return U_svd.to(dtype=orig_dtype)
    U_svd, _, _ = torch.linalg.svd(x)
    return U_svd


@dataclass
class DCAGConfig:
    """Configuration for domain-conditioned adapter generation."""

    enabled: bool = False
    feature_dim: int = 512
    alpha: Optional[float] = None
    anti_forgetting_weight: float = 0.1
    anti_forgetting_weight_per_domain: Optional[Sequence[float]] = None
    feature_layers: Optional[Sequence[int]] = None
    train_domain_backbones: bool = False
    domain_model_device: str = "cuda"
    domain_model_dtype: str = "float16"
    rs_ckpt_path: Optional[str] = None
    rs_extractor_type: str = "rvsa"
    med_ckpt_path: Optional[str] = None
    med_extractor_type: str = "pubmedclip"
    ad_ckpt_path: Optional[str] = None
    ad_extractor_type: str = "geomim"
    sci_model_dir: Optional[str] = None
    sci_extractor_type: str = "pix2struct"
    fin_model_dir: Optional[str] = None
    fin_extractor_type: str = "candlefusion"

    domain_image_resize_mode: str = "squash"
    basis_lr: Optional[float] = None
    generator_lr: Optional[float] = None

    r: int = 32
    h_d_llm: int = 2
    h_d_projector: int = 1
    k_d: int = 4
    r_fisher: int = 32
    num_centroids: int = 4
    num_centroids_per_domain: Optional[Sequence[int]] = None
    generator_hidden_dim: int = 512
    generator_layers: int = 4
    generator_heads: int = 8
    factored_head_dim: int = 8

    generator_dropout: float = 0.1
    a_l2_penalty: float = 1e-3

    feature_token_dropout: float = 0.05
    feature_noise_std: float = 0.01
    feature_regularize_generator_only: bool = True

    projector_preservation_multiplier: float = 3.0
    projector_warmup_steps: int = 500
    projector_cr_lr_multiplier: float = 0.5
    projector_disable_dynamic_B: bool = True

    preservation_trunk_coeff: float = 3.0
    preservation_uv_coeff: float = 1.0

    dynamic_current_B: bool = False

    shared_projection_snapshot_enabled: bool = True

    hidden_state_consistency_enabled: bool = True

    rope_layer_enabled: bool = True

    domain_embed_enabled: bool = True

    spectral_balance_coeff: float = 1e-4
    spectral_balance_every_n_steps: int = 50

    task_idx: Optional[int] = None

    B_mode: str = "column_partitioned_CR"
    R_retraction: str = "qr"
    Wd_mode: str = "fisher_oja_per_slot"
    slot_weight_mode: str = "projector_multiplier"
    projector_mode: str = "integrated_with_policy"
    generator_arch: str = "crossattn_adaln"
    domain_feature_mode: str = "all_layers_crossattn"
    preservation_target: str = "UV_factor_mse"

    generator_cache_mode: str = "global_all_slots"

    b_composition_mode: str = "full"
    domain_id_source: str = "oracle"

    new_slice_suppression_coeff: float = 0.0

    functional_preservation_coeff: float = 1.0

    rolling_function_preservation: bool = True
    rolling_start_task_idx: int = 4

    alignment_coeff: float = 0.0

    instruction_conditioning: bool = False
    instruction_feature_dim: int = 4096

    attn_pool_queries: int = 1

    residual_rank: int = 0
    residual_lr: float = 2e-4
    residual_alpha: Optional[float] = None

    def __post_init__(self):

        if isinstance(self.feature_layers, str):
            try:
                self.feature_layers = tuple(
                    int(x.strip()) for x in self.feature_layers.split(",") if x.strip()
                )
            except ValueError:
                raise ValueError(
                    f"feature_layers must be CSV of ints, got {self.feature_layers!r}"
                )
        self.anti_forgetting_weight_per_domain = _parse_optional_float_sequence(
            self.anti_forgetting_weight_per_domain
        )
        self.num_centroids_per_domain = _parse_optional_int_sequence(
            self.num_centroids_per_domain
        )
        if self.anti_forgetting_weight_per_domain is not None and len(
            self.anti_forgetting_weight_per_domain
        ) != len(DCAG_DOMAIN_TO_ID):
            raise ValueError(
                "anti_forgetting_weight_per_domain must have length "
                f"{len(DCAG_DOMAIN_TO_ID)}, got {len(self.anti_forgetting_weight_per_domain)}"
            )
        if self.num_centroids_per_domain is not None:
            if len(self.num_centroids_per_domain) != len(DCAG_DOMAIN_TO_ID):
                raise ValueError(
                    "num_centroids_per_domain must have length "
                    f"{len(DCAG_DOMAIN_TO_ID)}, got {len(self.num_centroids_per_domain)}"
                )
            self.num_centroids = max(
                int(self.num_centroids), max(self.num_centroids_per_domain)
            )
        if not (0.0 <= float(self.feature_token_dropout) < 1.0):
            raise ValueError(
                f"feature_token_dropout must be in [0, 1), got {self.feature_token_dropout}"
            )
        if float(self.feature_noise_std) < 0.0:
            raise ValueError(
                f"feature_noise_std must be non-negative, got {self.feature_noise_std}"
            )
        if float(self.new_slice_suppression_coeff) < 0.0:
            raise ValueError(
                "new_slice_suppression_coeff must be non-negative, "
                f"got {self.new_slice_suppression_coeff}"
            )
        if float(self.functional_preservation_coeff) < 0.0:
            raise ValueError(
                "functional_preservation_coeff must be non-negative, "
                f"got {self.functional_preservation_coeff}"
            )
        if int(self.rolling_start_task_idx) < 0:
            raise ValueError(
                "rolling_start_task_idx must be non-negative, "
                f"got {self.rolling_start_task_idx}"
            )
        if int(self.residual_rank) < 0:
            raise ValueError(
                f"residual_rank must be non-negative, got {self.residual_rank}"
            )
        if self.residual_alpha is None:
            self.residual_alpha = 2.0 * float(self.residual_rank)
        if int(self.residual_rank) > 0 and float(self.residual_alpha) <= 0.0:
            raise ValueError(
                "residual_alpha must be positive when residual_rank > 0, "
                f"got {self.residual_alpha}"
            )
        for domain, expected in (
            ("rs", "rvsa"),
            ("med", "pubmedclip"),
            ("ad", "geomim"),
            ("sci", "pix2struct"),
            ("fin", "candlefusion"),
        ):
            key = f"{domain}_extractor_type"
            value = str(getattr(self, key)).strip().lower()
            if value != expected:
                raise ValueError(f"{key} must be {expected!r}, got {value!r}")
            setattr(self, key, value)
        self.domain_image_resize_mode = (
            str(self.domain_image_resize_mode).strip().lower()
        )
        if self.domain_image_resize_mode not in ("squash", "center_crop"):
            raise ValueError(
                "domain_image_resize_mode must be one of {'squash', 'center_crop'}, "
                f"got {self.domain_image_resize_mode!r}"
            )
        if self.b_composition_mode not in ("full", "domain_prefix"):
            raise ValueError(
                "b_composition_mode must be one of {'full', 'domain_prefix'}, "
                f"got {self.b_composition_mode!r}"
            )
        if self.domain_id_source not in ("oracle",):
            raise ValueError(
                "domain_id_source must be 'oracle': explicit domain IDs are supplied "
                "by the dataset or the external prototype router. "
                f"Got {self.domain_id_source!r}"
            )


def _resize_domain_image(x: torch.Tensor, image_size: int, mode: str) -> torch.Tensor:
    h, w = x.shape[-2:]
    if mode == "center_crop":
        scale = float(image_size) / float(min(h, w))
        new_h = max(image_size, int(round(h * scale)))
        new_w = max(image_size, int(round(w * scale)))
        if (new_h, new_w) != (h, w):
            x = F.interpolate(
                x.float(), size=(new_h, new_w), mode="bilinear", align_corners=False
            ).to(dtype=x.dtype)
        top = (new_h - image_size) // 2
        left = (new_w - image_size) // 2
        return x[..., top : top + image_size, left : left + image_size]
    if (h, w) != (image_size, image_size):
        x = F.interpolate(
            x.float(),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).to(dtype=x.dtype)
    return x


class PubMedCLIPFeatureExtractor(nn.Module):
    """Frozen PubMedCLIP features with one (B, N, 768) tensor per layer."""

    def __init__(
        self,
        model_path: str,
        device: torch.device,
        dtype: torch.dtype,
        resize_mode: str = "squash",
    ):
        from transformers import CLIPModel, CLIPVisionModel

        super().__init__()
        if model_path is None:
            raise FileNotFoundError("PubMedCLIP model path is required but was None")
        model_path = os.path.expanduser(str(model_path))
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"PubMedCLIP model directory not found: {model_path}"
            )
        if not os.path.exists(os.path.join(model_path, "config.json")):
            raise FileNotFoundError(
                f"PubMedCLIP config.json not found under: {model_path}"
            )

        self.device = device
        self.dtype = dtype
        self.resize_mode = resize_mode
        load_source = "CLIPVisionModel"
        try:
            self.encoder = CLIPVisionModel.from_pretrained(
                model_path,
                local_files_only=True,
                output_hidden_states=True,
            )
        except Exception as vision_error:
            try:
                clip_model = CLIPModel.from_pretrained(
                    model_path,
                    local_files_only=True,
                    output_hidden_states=True,
                )
                self.encoder = clip_model.vision_model
                load_source = "CLIPModel.vision_model"
            except Exception as clip_error:
                raise RuntimeError(
                    "Failed to load PubMedCLIP as CLIPVisionModel or CLIPModel "
                    f"from {model_path}; vision_error={vision_error}; clip_error={clip_error}"
                ) from clip_error
        hidden_size = int(getattr(self.encoder.config, "hidden_size", 0) or 0)
        if hidden_size != 768:
            raise ValueError(
                "PubMedCLIP extractor expects hidden_size=768 to match the "
                f"current Med aggregator, got hidden_size={hidden_size} at {model_path}"
            )
        image_size = getattr(self.encoder.config, "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            image_size = int(image_size[0])
        self.image_size = int(image_size or 224)
        self.register_buffer(
            "pixel_mean",
            torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype
            ).view(1, 3, 1, 1),
            persistent=False,
        )
        self.encoder.eval()
        self.encoder.to(device=device, dtype=dtype)
        for p in self.encoder.parameters():
            p.requires_grad = False
        num_layers = int(getattr(self.encoder.config, "num_hidden_layers", 0) or 0)
        print(
            f"[DCAG][Med] med_extractor_type=pubmedclip path={model_path} "
            f"source={load_source} image_size={self.image_size} hidden_size={hidden_size} layers={num_layers}",
            flush=True,
        )

    def _runtime_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        ref = next(self.encoder.parameters())
        return ref.device, ref.dtype

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        x = images.to(device=runtime_device, dtype=runtime_dtype)
        x = _resize_domain_image(x, self.image_size, self.resize_mode)
        pixel_mean = self.pixel_mean.to(device=runtime_device, dtype=runtime_dtype)
        pixel_std = self.pixel_std.to(device=runtime_device, dtype=runtime_dtype)
        return (x - pixel_mean) / pixel_std

    @torch.no_grad()
    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        x = self._preprocess(images)
        _, runtime_dtype = self._runtime_device_dtype()
        outputs = self.encoder(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) <= 1:
            raise RuntimeError("PubMedCLIP did not return transformer hidden_states")
        tokens: List[torch.Tensor] = []
        for hidden in hidden_states[1:]:
            if hidden.dim() != 3 or hidden.shape[-1] != 768:
                raise RuntimeError(
                    f"Unexpected PubMedCLIP hidden state shape: {tuple(hidden.shape)}"
                )
            tokens.append(hidden[:, 1:, :].to(dtype=runtime_dtype))
        return tokens


class RSFeatureExtractor(nn.Module):
    def __init__(
        self, checkpoint_path: Optional[str], device: torch.device, dtype: torch.dtype
    ):
        from domain_encoders.rs.model_mae_af import mae_vit_base_patch16

        super().__init__()
        self.device = device
        self.dtype = dtype
        self.input_size = 224
        self.encoder = mae_vit_base_patch16()
        if checkpoint_path is None:
            raise FileNotFoundError(
                "RS domain checkpoint path is required but was None"
            )
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"RS domain checkpoint not found: {checkpoint_path}"
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        raw_state = _unwrap_checkpoint_state_dict(checkpoint)
        if not isinstance(raw_state, dict):
            raise RuntimeError(
                f"RS checkpoint did not contain a state dict: {checkpoint_path}"
            )
        (
            matched,
            label,
            matched_count,
            matched_numel,
            matched_ratio,
            missing,
            unexpected,
        ) = _select_matching_rs_state(
            raw_state,
            self.encoder.state_dict(),
        )
        if matched_ratio < 0.50 or matched_count < 10:
            raise RuntimeError(
                "RS checkpoint has too few matching MAE encoder weights: "
                f"path={checkpoint_path}, candidate={label}, matched_tensors={matched_count}, "
                f"matched_ratio={matched_ratio:.3f}, missing={len(missing)}, unexpected_or_mismatch={len(unexpected)}"
            )
        self.encoder.load_state_dict(matched, strict=False)
        print(
            f"[DCAG][RS] loaded checkpoint {checkpoint_path}; candidate={label}, "
            f"matched_tensors={matched_count}, matched_params={matched_numel}, "
            f"matched_ratio={matched_ratio:.3f}, missing={len(missing)}, "
            f"unexpected_or_mismatch={len(unexpected)}",
            flush=True,
        )
        self.register_buffer(
            "pixel_mean",
            torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(
                1, 3, 1, 1
            ),
            persistent=False,
        )
        self.encoder.eval()
        self.encoder.to(device=device, dtype=dtype)
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _runtime_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        ref = next(self.encoder.parameters())
        return ref.device, ref.dtype

    @torch.no_grad()
    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        runtime_device, runtime_dtype = self._runtime_device_dtype()
        x = images.to(device=runtime_device, dtype=runtime_dtype)
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x.float(),
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=runtime_dtype)
        pixel_mean = self.pixel_mean.to(device=runtime_device, dtype=runtime_dtype)
        pixel_std = self.pixel_std.to(device=runtime_device, dtype=runtime_dtype)
        x = (x - pixel_mean) / pixel_std
        features = self.encoder.extract_features(x)
        return [feat.to(dtype=runtime_dtype) for feat in features]


class LazyExtractorProxy(nn.Module):
    def __init__(
        self, factory: Callable[[], nn.Module], feature_dim: int, trainable: bool
    ):
        super().__init__()
        self.factory = factory
        self.feature_dim = feature_dim
        self.trainable = trainable
        self.extractor: Optional[nn.Module] = None

    def _get_extractor(self) -> nn.Module:
        if self.extractor is None:
            print(
                f"[DCAG][lazy] building extractor feature_dim={self.feature_dim}",
                flush=True,
            )
            extractor = self.factory()
            if not self.trainable:
                extractor.eval()
                for p in extractor.parameters():
                    p.requires_grad = False
            self.extractor = extractor
            print(
                f"[DCAG][lazy] extractor ready feature_dim={self.feature_dim}",
                flush=True,
            )
        return self.extractor

    def extract_layer_tokens(self, images: torch.Tensor) -> List[torch.Tensor]:
        return self._get_extractor().extract_layer_tokens(images)

    def extract_layer_tokens_by_view(self, images: torch.Tensor) -> List[torch.Tensor]:
        extractor = self._get_extractor()
        return extractor.extract_layer_tokens_by_view(images)


class DomainFeatureAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        selected_layers: Optional[Sequence[int]] = None,
        mode: str = "all_layers_crossattn",
        num_pool_queries: int = 1,
    ):
        super().__init__()
        self.mode = mode

        self.num_pool_queries = max(1, int(num_pool_queries))

        self.selected_layers = (
            tuple(selected_layers) if selected_layers else (-3, -2, -1)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layer_tokens = 32
        if mode == "last_3_layers_pooled":
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim * len(self.selected_layers)),
                nn.Linear(input_dim * len(self.selected_layers), output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif mode == "last_layer_only":
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif mode in (
            "all_layers_crossattn",
            "all_layers_mean_lte",
            "all_layers_attn_pool",
        ):
            if mode == "all_layers_mean_lte":
                self.layer_embedding = nn.Parameter(
                    torch.zeros(self.max_layer_tokens, input_dim)
                )
                self.type_embedding = nn.Parameter(torch.zeros(2, input_dim))
                nn.init.normal_(self.layer_embedding, mean=0.0, std=0.02)
                nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
            if mode == "all_layers_attn_pool":
                self.token_score = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 1),
                )
                nn.init.zeros_(self.token_score[-1].weight)
                nn.init.zeros_(self.token_score[-1].bias)

                self.token_score_extra = nn.ModuleList()
                for _ in range(self.num_pool_queries - 1):
                    head = nn.Sequential(
                        nn.LayerNorm(input_dim),
                        nn.Linear(input_dim, 1),
                    )
                    nn.init.zeros_(head[-1].weight)
                    nn.init.zeros_(head[-1].bias)
                    self.token_score_extra.append(head)

            self.token_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )
        else:
            raise ValueError(f"Unknown domain_feature_mode {mode!r}")

    def _select_layers(self, layer_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        selected = []
        num_layers = len(layer_tokens)
        for idx in self.selected_layers:
            actual_idx = idx if idx >= 0 else num_layers + idx
            actual_idx = max(0, min(actual_idx, num_layers - 1))
            selected.append(layer_tokens[actual_idx])
        return selected

    def _attention_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:

        score_param = next(self.token_score.parameters())
        tokens = tokens.to(device=score_param.device, dtype=score_param.dtype)
        scores = self.token_score(tokens)
        weights = torch.softmax(scores, dim=1)
        pooled = (tokens * weights).sum(dim=1)
        if self.num_pool_queries <= 1:
            return pooled

        outs = [pooled]
        for head in self.token_score_extra:
            scores_e = head(tokens)
            weights_e = torch.softmax(scores_e, dim=1)
            outs.append((tokens * weights_e).sum(dim=1))
        return torch.stack(outs, dim=1)

    def forward(self, layer_tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last_3_layers_pooled":
            proj_param = next(self.proj.parameters())
            pooled = [
                layer.mean(dim=1).to(device=proj_param.device, dtype=proj_param.dtype)
                for layer in self._select_layers(layer_tokens)
            ]
            return self.proj(torch.cat(pooled, dim=-1))
        elif self.mode == "last_layer_only":
            proj_param = next(self.proj.parameters())
            last = (
                layer_tokens[-1]
                .mean(dim=1)
                .to(device=proj_param.device, dtype=proj_param.dtype)
            )
            return self.proj(last)
        elif self.mode == "all_layers_crossattn":
            proj_param = next(self.token_proj.parameters())
            per_layer = [
                layer.mean(dim=1).to(device=proj_param.device, dtype=proj_param.dtype)
                for layer in layer_tokens
            ]
            stacked = torch.stack(per_layer, dim=1)
            global_summary = stacked.mean(dim=1, keepdim=True)
            all_tokens = torch.cat([stacked, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        elif self.mode == "all_layers_attn_pool":
            pooled_per_layer = [
                self._attention_pool_tokens(layer) for layer in layer_tokens
            ]

            if self.num_pool_queries <= 1:
                stacked = torch.stack(pooled_per_layer, dim=1)
            else:
                stacked = torch.stack(pooled_per_layer, dim=1)
                stacked = stacked.flatten(1, 2)
            global_summary = stacked.mean(dim=1, keepdim=True)
            all_tokens = torch.cat([stacked, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        elif self.mode == "all_layers_mean_lte":
            proj_param = next(self.token_proj.parameters())
            per_layer = [
                layer.mean(dim=1).to(device=proj_param.device, dtype=proj_param.dtype)
                for layer in layer_tokens
            ]
            stacked = torch.stack(per_layer, dim=1)
            layer_count = stacked.shape[1]
            if layer_count > self.max_layer_tokens:
                raise ValueError(
                    f"all_layers_mean_lte supports at most {self.max_layer_tokens} layer tokens, got {layer_count}"
                )
            layer_emb = (
                self.layer_embedding[:layer_count]
                .to(device=stacked.device, dtype=stacked.dtype)
                .unsqueeze(0)
            )
            layer_type = (
                self.type_embedding[0]
                .to(device=stacked.device, dtype=stacked.dtype)
                .view(1, 1, -1)
            )
            summary_type = (
                self.type_embedding[1]
                .to(device=stacked.device, dtype=stacked.dtype)
                .view(1, 1, -1)
            )
            layer_tokens_with_id = stacked + layer_emb + layer_type
            global_summary = stacked.mean(dim=1, keepdim=True) + summary_type
            all_tokens = torch.cat([layer_tokens_with_id, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        else:
            raise ValueError(self.mode)


class ADViewAwareFeatureAggregator(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        selected_layers: Optional[Sequence[int]] = None,
        mode: str = "all_layers_crossattn",
    ):
        super().__init__()
        self.mode = mode
        self.selected_layers = (
            tuple(selected_layers) if selected_layers else (-3, -2, -1)
        )
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_layer_tokens = 32
        self.view_score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, 1),
        )
        if mode == "last_3_layers_pooled":
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim * len(self.selected_layers)),
                nn.Linear(input_dim * len(self.selected_layers), output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif mode == "last_layer_only":
            self.proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
            )
        elif mode in (
            "all_layers_crossattn",
            "all_layers_mean_lte",
            "all_layers_attn_pool",
        ):
            if mode == "all_layers_mean_lte":
                self.layer_embedding = nn.Parameter(
                    torch.zeros(self.max_layer_tokens, input_dim)
                )
                self.type_embedding = nn.Parameter(torch.zeros(2, input_dim))
                nn.init.normal_(self.layer_embedding, mean=0.0, std=0.02)
                nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)
            if mode == "all_layers_attn_pool":
                self.token_score = nn.Sequential(
                    nn.LayerNorm(input_dim),
                    nn.Linear(input_dim, 1),
                )
                nn.init.zeros_(self.token_score[-1].weight)
                nn.init.zeros_(self.token_score[-1].bias)
            self.token_proj = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.LayerNorm(output_dim),
            )
        else:
            raise ValueError(f"Unknown domain_feature_mode {mode!r}")

    def _fuse_views(self, layer_tokens_view: torch.Tensor) -> torch.Tensor:

        if layer_tokens_view.dim() != 4:
            raise ValueError(
                f"Expected AD layer shape (B, 6, N, D), got {tuple(layer_tokens_view.shape)}"
            )
        view_param = next(self.view_score.parameters())
        if self.mode == "all_layers_attn_pool":
            batch_size, num_views, num_tokens, hidden_dim = layer_tokens_view.shape
            pooled = self._attention_pool_tokens(
                layer_tokens_view.reshape(
                    batch_size * num_views, num_tokens, hidden_dim
                )
            ).reshape(batch_size, num_views, hidden_dim)
            per_view = pooled.to(device=view_param.device, dtype=view_param.dtype)
        else:
            per_view = layer_tokens_view.mean(dim=2).to(
                device=view_param.device, dtype=view_param.dtype
            )
        view_scores = self.view_score(per_view)
        view_weights = torch.softmax(view_scores, dim=1)
        fused = (per_view * view_weights).sum(dim=1)
        return fused

    def _select_layers(self, layer_tokens: List[torch.Tensor]) -> List[torch.Tensor]:
        selected = []
        num_layers = len(layer_tokens)
        for idx in self.selected_layers:
            actual_idx = idx if idx >= 0 else num_layers + idx
            actual_idx = max(0, min(actual_idx, num_layers - 1))
            selected.append(layer_tokens[actual_idx])
        return selected

    def _attention_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:

        score_param = next(self.token_score.parameters())
        tokens = tokens.to(device=score_param.device, dtype=score_param.dtype)
        scores = self.token_score(tokens)
        weights = torch.softmax(scores, dim=1)
        return (tokens * weights).sum(dim=1)

    def forward(self, layer_tokens: List[torch.Tensor]) -> torch.Tensor:
        if self.mode == "last_3_layers_pooled":
            proj_param = next(self.proj.parameters())
            fused_layers = [
                self._fuse_views(layer).to(
                    device=proj_param.device, dtype=proj_param.dtype
                )
                for layer in self._select_layers(layer_tokens)
            ]
            return self.proj(torch.cat(fused_layers, dim=-1))
        elif self.mode == "last_layer_only":
            proj_param = next(self.proj.parameters())
            fused = self._fuse_views(layer_tokens[-1]).to(
                device=proj_param.device, dtype=proj_param.dtype
            )
            return self.proj(fused)
        elif self.mode == "all_layers_crossattn":
            proj_param = next(self.token_proj.parameters())
            fused_per_layer = [
                self._fuse_views(layer).to(
                    device=proj_param.device, dtype=proj_param.dtype
                )
                for layer in layer_tokens
            ]
            stacked = torch.stack(fused_per_layer, dim=1)
            global_summary = stacked.mean(dim=1, keepdim=True)
            all_tokens = torch.cat([stacked, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        elif self.mode == "all_layers_attn_pool":
            proj_param = next(self.token_proj.parameters())
            fused_per_layer = [
                self._fuse_views(layer).to(
                    device=proj_param.device, dtype=proj_param.dtype
                )
                for layer in layer_tokens
            ]
            stacked = torch.stack(fused_per_layer, dim=1)
            global_summary = stacked.mean(dim=1, keepdim=True)
            all_tokens = torch.cat([stacked, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        elif self.mode == "all_layers_mean_lte":
            proj_param = next(self.token_proj.parameters())
            fused_per_layer = [
                self._fuse_views(layer).to(
                    device=proj_param.device, dtype=proj_param.dtype
                )
                for layer in layer_tokens
            ]
            stacked = torch.stack(fused_per_layer, dim=1)
            layer_count = stacked.shape[1]
            if layer_count > self.max_layer_tokens:
                raise ValueError(
                    f"all_layers_mean_lte supports at most {self.max_layer_tokens} layer tokens, got {layer_count}"
                )
            layer_emb = (
                self.layer_embedding[:layer_count]
                .to(device=stacked.device, dtype=stacked.dtype)
                .unsqueeze(0)
            )
            layer_type = (
                self.type_embedding[0]
                .to(device=stacked.device, dtype=stacked.dtype)
                .view(1, 1, -1)
            )
            summary_type = (
                self.type_embedding[1]
                .to(device=stacked.device, dtype=stacked.dtype)
                .view(1, 1, -1)
            )
            layer_tokens_with_id = stacked + layer_emb + layer_type
            global_summary = stacked.mean(dim=1, keepdim=True) + summary_type
            all_tokens = torch.cat([layer_tokens_with_id, global_summary], dim=1)
            B, T, D = all_tokens.shape
            projected = self.token_proj(all_tokens.reshape(B * T, D)).reshape(
                B, T, self.output_dim
            )
            return projected
        else:
            raise ValueError(self.mode)


class DomainFeatureBatchFormatter(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 3:
            return images.unsqueeze(0)
        return images


class ADFeatureBatchFormatter(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            return images.unsqueeze(0)
        return images


class ADMosaicFormatter(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 5:
            return images
        batch_size, num_views, channels, height, width = images.shape
        if num_views != 6:
            raise ValueError(f"Expected 6 AD views, got {num_views}")
        top_row = torch.cat([images[:, 0], images[:, 1], images[:, 2]], dim=-1)
        bottom_row = torch.cat([images[:, 3], images[:, 4], images[:, 5]], dim=-1)
        return torch.cat([top_row, bottom_row], dim=-2)


class DomainFeaturePipeline(nn.Module):
    def __init__(
        self,
        extractor: nn.Module,
        aggregator: nn.Module,
        formatter: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.extractor = extractor
        self.aggregator = aggregator
        self.formatter = formatter or DomainFeatureBatchFormatter()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prepared = self.formatter(images)
        tokens = self.extractor.extract_layer_tokens(prepared)
        return self.aggregator(tokens)


class ADFeaturePipeline(nn.Module):
    def __init__(self, extractor: nn.Module, aggregator: nn.Module):
        super().__init__()
        self.extractor = extractor
        self.aggregator = aggregator
        self.batch_formatter = ADFeatureBatchFormatter()
        self.mosaic_formatter = ADMosaicFormatter()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        prepared = self.batch_formatter(images)
        prepared = self.mosaic_formatter(prepared)
        tokens = self.extractor.extract_layer_tokens_by_view(prepared)
        return self.aggregator(tokens)


class SingleSampleFeaturePipeline(nn.Module):
    def __init__(self, pipeline: nn.Module):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.pipeline(images)


class DCAGDomainFeatureEncoder(nn.Module):
    """Select domain encoders and unify their features.

    Returns (B, feature_dim) or (B, T, feature_dim), depending on feature mode.
    """

    def __init__(self, config: DCAGConfig):
        super().__init__()
        device = torch.device(config.domain_model_device)
        dtype = getattr(torch, config.domain_model_dtype)
        self.device = device
        self.dtype = dtype
        self.train_domain_backbones = config.train_domain_backbones
        self.selected_layers = config.feature_layers
        self.feature_mode = config.domain_feature_mode
        self.pipelines = nn.ModuleDict()

        if config.rs_ckpt_path is not None:
            rs_type = str(getattr(config, "rs_extractor_type", "rvsa")).strip().lower()
            resize_mode = (
                str(getattr(config, "domain_image_resize_mode", "squash"))
                .strip()
                .lower()
            )
            if rs_type != "rvsa":
                raise ValueError("DCAG requires rvsa for this domain")
            rs_extractor = self._make_extractor(
                feature_dim=768,
                factory=lambda: RSFeatureExtractor(config.rs_ckpt_path, device, dtype),
            )
            rs_input_dim = 768
            print(
                f"[DCAG][RS] selected rs_extractor_type={rs_type} input_dim={rs_input_dim}",
                flush=True,
            )
            self.pipelines["RS"] = SingleSampleFeaturePipeline(
                DomainFeaturePipeline(
                    rs_extractor,
                    DomainFeatureAggregator(
                        rs_input_dim,
                        config.feature_dim,
                        self.selected_layers,
                        mode=self.feature_mode,
                        num_pool_queries=getattr(config, "attn_pool_queries", 1),
                    ),
                )
            )
        if config.med_ckpt_path is not None:
            med_type = (
                str(getattr(config, "med_extractor_type", "pubmedclip")).strip().lower()
            )
            resize_mode = (
                str(getattr(config, "domain_image_resize_mode", "squash"))
                .strip()
                .lower()
            )
            if med_type != "pubmedclip":
                raise ValueError("DCAG requires pubmedclip for this domain")
            med_factory = lambda: PubMedCLIPFeatureExtractor(
                config.med_ckpt_path, device, dtype, resize_mode
            )
            med_input_dim = 768
            print(
                f"[DCAG][Med] selected med_extractor_type={med_type} resize_mode={resize_mode} input_dim={med_input_dim}",
                flush=True,
            )
            self.pipelines["Med"] = SingleSampleFeaturePipeline(
                DomainFeaturePipeline(
                    self._make_extractor(
                        feature_dim=med_input_dim,
                        factory=med_factory,
                    ),
                    DomainFeatureAggregator(
                        med_input_dim,
                        config.feature_dim,
                        self.selected_layers,
                        mode=self.feature_mode,
                        num_pool_queries=getattr(config, "attn_pool_queries", 1),
                    ),
                )
            )
        if config.ad_ckpt_path is not None:
            ad_type = (
                str(getattr(config, "ad_extractor_type", "geomim")).strip().lower()
            )
            if ad_type != "geomim":
                raise ValueError("DCAG requires geomim for this domain")
            ad_extractor = self._make_extractor(
                feature_dim=512,
                factory=lambda: __import__(
                    "domain_encoders.ad.geomim", fromlist=["GeoMIMSwinBackbone"]
                ).GeoMIMSwinBackbone(
                    config.ad_ckpt_path, device, dtype, input_norm="01", verbose=False
                ),
            )
            print(
                f"[DCAG][AD] selected ad_extractor_type={ad_type} feature_dim={ad_extractor.feature_dim}",
                flush=True,
            )
            self.pipelines["AD"] = SingleSampleFeaturePipeline(
                ADFeaturePipeline(
                    ad_extractor,
                    ADViewAwareFeatureAggregator(
                        ad_extractor.feature_dim,
                        config.feature_dim,
                        self.selected_layers,
                        mode=self.feature_mode,
                    ),
                )
            )
        if config.sci_model_dir is not None:
            sci_type = (
                str(getattr(config, "sci_extractor_type", "pix2struct")).strip().lower()
            )
            if sci_type != "pix2struct":
                raise ValueError("DCAG requires pix2struct for this domain")
            sci_extractor = self._make_extractor(
                feature_dim=768,
                factory=lambda: __import__(
                    "domain_encoders.sci.pix2struct",
                    fromlist=["Pix2StructVisionBackbone"],
                ).Pix2StructVisionBackbone(config.sci_model_dir, device, dtype),
            )
            sci_input_dim = 768
            print(
                f"[DCAG][Sci] selected sci_extractor_type={sci_type} input_dim={sci_input_dim}",
                flush=True,
            )
            self.pipelines["Sci"] = SingleSampleFeaturePipeline(
                DomainFeaturePipeline(
                    sci_extractor,
                    DomainFeatureAggregator(
                        sci_input_dim,
                        config.feature_dim,
                        self.selected_layers,
                        mode=self.feature_mode,
                        num_pool_queries=getattr(config, "attn_pool_queries", 1),
                    ),
                )
            )
        if config.fin_model_dir is not None:
            fin_type = (
                str(getattr(config, "fin_extractor_type", "candlefusion"))
                .strip()
                .lower()
            )
            if fin_type != "candlefusion":
                raise ValueError("DCAG requires candlefusion for this domain")
            fin_extractor = self._make_extractor(
                feature_dim=768,
                factory=lambda: __import__(
                    "domain_encoders.fin.candlefusion",
                    fromlist=["CandleFusionViTEncoder"],
                ).CandleFusionViTEncoder(
                    config.fin_model_dir, device, dtype, input_norm="01"
                ),
            )
            fin_input_dim = 768
            print(
                f"[DCAG][Fin] selected fin_extractor_type={fin_type} input_dim={fin_input_dim}",
                flush=True,
            )
            self.pipelines["Fin"] = SingleSampleFeaturePipeline(
                DomainFeaturePipeline(
                    fin_extractor,
                    DomainFeatureAggregator(
                        fin_input_dim,
                        config.feature_dim,
                        self.selected_layers,
                        mode=self.feature_mode,
                        num_pool_queries=getattr(config, "attn_pool_queries", 1),
                    ),
                )
            )

        self.domain_pipeline_names = tuple(self.pipelines.keys())
        self.shared_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.feature_dim),
            nn.GELU(),
            nn.LayerNorm(config.feature_dim),
        )

    def _make_extractor(
        self, feature_dim: int, factory: Callable[[], nn.Module]
    ) -> nn.Module:
        if self.train_domain_backbones:
            return factory()
        return LazyExtractorProxy(
            factory=factory, feature_dim=feature_dim, trainable=False
        )

    def _select_one_sample(self, images, batch_index: int) -> torch.Tensor:
        if isinstance(images, (list, tuple)):
            sample = images[batch_index]
            if sample.dim() == 3 or sample.dim() == 4:
                return sample.unsqueeze(0)
            return sample
        return images[batch_index : batch_index + 1]

    def _apply_shared(self, features: torch.Tensor) -> torch.Tensor:
        shared_param = next(self.shared_projection.parameters())
        features = features.to(device=shared_param.device, dtype=shared_param.dtype)
        if features.dim() == 2:
            return self.shared_projection(features)

        B, T, D = features.shape
        return self.shared_projection(features.reshape(B * T, D)).reshape(B, T, D)

    def forward(self, images, domain_ids: torch.Tensor) -> torch.Tensor:
        unique_ids = torch.unique(domain_ids)
        homogeneous = unique_ids.numel() == 1 and not isinstance(images, (list, tuple))
        if homogeneous:
            domain_name = DCAG_ID_TO_DOMAIN[int(unique_ids.item())]
            if domain_name not in self.pipelines:
                raise KeyError(f"No domain pipeline configured for {domain_name}")
            features = self.pipelines[domain_name](images)
        else:
            per_sample = []
            for batch_index, domain_id in enumerate(domain_ids):
                domain_name = DCAG_ID_TO_DOMAIN[int(domain_id.item())]
                if domain_name not in self.pipelines:
                    raise KeyError(f"No domain pipeline configured for {domain_name}")
                sample_images = self._select_one_sample(images, batch_index)
                per_sample.append(self.pipelines[domain_name](sample_images))
            features = torch.cat(per_sample, dim=0)
        return self._apply_shared(features)


class PrototypeCentroidManager(nn.Module):
    """Maintain per-domain centroids with online EMA and periodic Lloyd updates."""

    def __init__(
        self,
        num_domains: int,
        num_centroids: int,
        feature_dim: int,
        refresh_every: int = 100,
        ring_buffer_size: int = 512,
        num_centroids_per_domain: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.num_domains = num_domains
        if num_centroids_per_domain is None:
            self.num_centroids_per_domain = [max(1, int(num_centroids))] * num_domains
        else:
            if len(num_centroids_per_domain) != num_domains:
                raise ValueError(
                    f"num_centroids_per_domain length {len(num_centroids_per_domain)} "
                    f"!= num_domains {num_domains}"
                )
            self.num_centroids_per_domain = [
                max(1, int(k)) for k in num_centroids_per_domain
            ]
        self.num_centroids = max(self.num_centroids_per_domain)
        self.feature_dim = feature_dim
        self.refresh_every = int(refresh_every)
        self.ring_buffer_size = int(ring_buffer_size)
        self.register_buffer(
            "prototype_features",
            torch.zeros(num_domains, self.num_centroids, feature_dim),
        )
        self.register_buffer(
            "prototype_counts", torch.zeros(num_domains, self.num_centroids)
        )

        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))

        self._recent_features: Dict[int, torch.Tensor] = {}
        self._recent_cursor: Dict[int, int] = {}
        self._recent_count: Dict[int, int] = {}

    @torch.no_grad()
    def _append_recent(self, d: int, feat: torch.Tensor) -> None:
        if self.num_centroids <= 1 or self.ring_buffer_size <= 0:
            return
        if d not in self._recent_features:
            self._recent_features[d] = torch.zeros(
                self.ring_buffer_size,
                self.feature_dim,
                dtype=self.prototype_features.dtype,
            )
            self._recent_cursor[d] = 0
            self._recent_count[d] = 0
        cursor = self._recent_cursor[d]
        self._recent_features[d][cursor].copy_(
            feat.detach().to(dtype=self.prototype_features.dtype, device="cpu")
        )
        self._recent_cursor[d] = (cursor + 1) % self.ring_buffer_size
        self._recent_count[d] = min(self._recent_count[d] + 1, self.ring_buffer_size)

    @torch.no_grad()
    def _lloyd_refresh(self) -> None:
        """Refine centroids with one Lloyd reassignment over recent feature summaries."""
        if self.num_centroids <= 1 or self.refresh_every <= 0:
            return
        for d, count in list(self._recent_count.items()):
            k_d = self.num_centroids_per_domain[d]
            if k_d <= 1 or count < k_d:
                continue
            buffer = self._recent_features[d][:count].to(
                device=self.prototype_features.device
            )
            centroids = self.prototype_features[d, :k_d]

            dists = torch.cdist(buffer, centroids)
            assignments = dists.argmin(dim=1)
            for k in range(k_d):
                mask = assignments == k
                if mask.any():
                    new_centroid = buffer[mask].mean(dim=0)
                    self.prototype_features[d, k].copy_(new_centroid)

    @torch.no_grad()
    def update(self, features: torch.Tensor, domain_ids: torch.Tensor):
        """Update centroids from (B, D) features, pooling tokens for (B, T, D) inputs."""
        if features.dim() == 3:
            features = features.mean(dim=1)
        features = features.detach()
        for feat, did in zip(features, domain_ids.detach()):
            d = int(did.item())
            k_d = self.num_centroids_per_domain[d]

            if k_d == 1:
                count = float(self.prototype_counts[d, 0].item())
                if count == 0.0:
                    self.prototype_features[d, 0].copy_(
                        feat.to(self.prototype_features.dtype)
                    )
                else:
                    momentum = count / (count + 1.0)
                    self.prototype_features[d, 0].mul_(momentum).add_(
                        feat.to(self.prototype_features.dtype) * (1.0 - momentum)
                    )
                self.prototype_counts[d, 0].add_(1.0)
                continue

            unfilled = (self.prototype_counts[d, :k_d] == 0).nonzero(as_tuple=True)[0]
            if unfilled.numel() > 0:
                target = int(unfilled[0].item())
                self.prototype_features[d, target].copy_(
                    feat.to(self.prototype_features.dtype)
                )
                self.prototype_counts[d, target].add_(1.0)
                self._append_recent(d, feat)
                continue

            centroids = self.prototype_features[d, :k_d]
            dists = torch.norm(
                centroids - feat.unsqueeze(0).to(centroids.dtype), dim=-1
            )
            k_star = int(dists.argmin().item())
            count = float(self.prototype_counts[d, k_star].item())
            momentum = count / (count + 1.0)
            self.prototype_features[d, k_star].mul_(momentum).add_(
                feat.to(self.prototype_features.dtype) * (1.0 - momentum)
            )
            self.prototype_counts[d, k_star].add_(1.0)
            self._append_recent(d, feat)

        self.step_counter += 1
        if (
            self.refresh_every > 0
            and self.num_centroids > 1
            and int(self.step_counter.item()) % self.refresh_every == 0
        ):
            self._lloyd_refresh()

    def completed_domain_indices(self, current_domain_id: Optional[int]) -> List[int]:
        """Return populated domains, excluding the current training domain."""
        populated = (
            (self.prototype_counts.sum(dim=-1) > 0).nonzero(as_tuple=True)[0].tolist()
        )
        if current_domain_id is None:
            return populated
        return [d for d in populated if d != current_domain_id]

    @torch.no_grad()
    def sync_across_ranks(self):
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        if not self.prototype_counts.is_cuda:
            return
        counts_total = self.prototype_counts.clone()
        torch.distributed.all_reduce(counts_total, op=torch.distributed.ReduceOp.SUM)
        weighted = self.prototype_features * self.prototype_counts.unsqueeze(-1)
        torch.distributed.all_reduce(weighted, op=torch.distributed.ReduceOp.SUM)
        safe_total = counts_total.clamp_min(1.0)
        self.prototype_features.copy_(weighted / safe_total.unsqueeze(-1))
        self.prototype_counts.copy_(counts_total)

    @torch.no_grad()
    def sync_completed_domains_only(self, current_domain_id: Optional[int]):
        """Synchronize completed-domain prototypes across ranks at task start."""
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        if not self.prototype_counts.is_cuda:
            return

        cur_features = None
        cur_counts = None
        if (
            current_domain_id is not None
            and 0 <= int(current_domain_id) < self.num_domains
        ):
            d = int(current_domain_id)
            cur_features = self.prototype_features[d].clone()
            cur_counts = self.prototype_counts[d].clone()
        counts_total = self.prototype_counts.clone()
        weighted = self.prototype_features * self.prototype_counts.unsqueeze(-1)
        torch.distributed.all_reduce(counts_total, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(weighted, op=torch.distributed.ReduceOp.SUM)
        safe_total = counts_total.clamp_min(1.0)
        self.prototype_features.copy_(weighted / safe_total.unsqueeze(-1))
        self.prototype_counts.copy_(counts_total)
        if cur_features is not None:
            d = int(current_domain_id)
            self.prototype_features[d].copy_(cur_features)
            self.prototype_counts[d].copy_(cur_counts)

    @torch.no_grad()
    def sync_current_domain_only(self, current_domain_id: Optional[int]):
        """Merge current-domain prototypes across ranks using count-weighted averaging."""
        if current_domain_id is None:
            return
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return
        if not self.prototype_counts.is_cuda:
            return
        d = int(current_domain_id)
        if d < 0 or d >= self.num_domains:
            return
        cur_counts_total = self.prototype_counts[d].clone()
        cur_weighted = self.prototype_features[d] * self.prototype_counts[d].unsqueeze(
            -1
        )
        torch.distributed.all_reduce(
            cur_counts_total, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(cur_weighted, op=torch.distributed.ReduceOp.SUM)
        safe_total = cur_counts_total.clamp_min(1.0)
        self.prototype_features[d].copy_(cur_weighted / safe_total.unsqueeze(-1))
        self.prototype_counts[d].copy_(cur_counts_total)


class _AdaLN(nn.Module):
    """Adaptive LayerNorm: scale/shift produced from a conditioning vector."""

    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:

        h = self.norm(x)
        scale_shift = self.to_scale_shift(cond)
        if scale_shift.dim() == 2:
            scale_shift = scale_shift.unsqueeze(1)
        scale, shift = scale_shift.chunk(2, dim=-1)
        return h * (1 + scale) + shift


class _CrossAttnBlock(nn.Module):
    """Cross-attention block: slots attend to domain-feature tokens."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        cond_dim: int,
        use_adaln: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_adaln = use_adaln
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if use_adaln:
            self.norm1 = _AdaLN(hidden_dim, cond_dim)
            self.norm2 = _AdaLN(hidden_dim, cond_dim)
        else:
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self, slots: torch.Tensor, kv: torch.Tensor, cond: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.use_adaln:
            h = self.norm1(slots, cond)
        else:
            h = self.norm1(slots)
        attn_out, _ = self.cross_attn(h, kv, kv, need_weights=False)
        x = slots + self.attn_dropout(attn_out)
        if self.use_adaln:
            h = self.norm2(x, cond)
        else:
            h = self.norm2(x)
        return x + self.ffn_dropout(self.ffn(h))


class _MLPAdditiveTrunk(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, num_total_slots: int):
        super().__init__()
        self.slot_embed = nn.Parameter(torch.randn(num_total_slots, hidden_dim) * 0.02)
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:

        if features.dim() == 3:
            features = features.mean(dim=1)
        hidden = self.trunk(features)

        hidden_per_slot = hidden.unsqueeze(1) + self.slot_embed.unsqueeze(0)
        return hidden_per_slot


class DCAGGenerator(nn.Module):
    """Generate slot-specific low-rank factors with cross-attention and AdaLN.

    Decoder slots occupy indices 0-223; projector slots occupy 224-225.
    Each slot yields U: (B, r, h_head) and V: (B, h_head, d_in).
    """

    D_IN_BUCKETS = (1024, 4096, 11008)

    def __init__(
        self,
        config: DCAGConfig,
        num_hidden_layers: int,
    ):
        super().__init__()
        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.r = config.r
        self.h_head = config.factored_head_dim
        self.hidden_dim = config.generator_hidden_dim
        self.num_targets = len(DCAG_TARGET_MODULES)
        self.num_total_slots = (
            num_hidden_layers * self.num_targets + NUM_PROJECTOR_SLOTS
        )
        self.arch = config.generator_arch
        self.cond_dim = config.feature_dim

        self.rope_layer_enabled = bool(getattr(config, "rope_layer_enabled", False))
        if not self.rope_layer_enabled:
            self.layer_embed = nn.Parameter(
                torch.randn(num_hidden_layers, self.hidden_dim) * 0.02
            )
        self.target_embed = nn.Parameter(
            torch.randn(self.num_targets, self.hidden_dim) * 0.02
        )
        self.projector_slot_embed = nn.Parameter(
            torch.randn(NUM_PROJECTOR_SLOTS, self.hidden_dim) * 0.02
        )

        if self.rope_layer_enabled:
            assert self.hidden_dim % 2 == 0, "RoPE requires even hidden_dim"
            half = self.hidden_dim // 2
            inv_freq = 1.0 / (
                10000.0 ** (torch.arange(0, half, dtype=torch.float32) / half)
            )
            pos = torch.arange(num_hidden_layers, dtype=torch.float32)
            angles = pos.unsqueeze(1) * inv_freq.unsqueeze(0)
            self.register_buffer("rope_layer_cos", angles.cos(), persistent=False)
            self.register_buffer("rope_layer_sin", angles.sin(), persistent=False)

        self.domain_embed_enabled = bool(getattr(config, "domain_embed_enabled", False))
        if self.domain_embed_enabled:
            self.domain_embed = nn.Parameter(
                torch.zeros(len(DCAG_DOMAIN_TO_ID), self.hidden_dim)
            )
        else:
            self.register_parameter("domain_embed", None)

        if self.arch == "mlp_additive":
            self.mlp_trunk = _MLPAdditiveTrunk(
                config.feature_dim, self.hidden_dim, self.num_total_slots
            )
        else:
            use_adaln = self.arch in ("crossattn_adaln", "adaln_mlp")
            use_crossattn = self.arch in ("crossattn_adaln", "crossattn_only")

            self.kv_in = nn.Linear(config.feature_dim, self.hidden_dim)
            blocks = []
            dropout_p = float(getattr(config, "generator_dropout", 0.0) or 0.0)
            for _ in range(config.generator_layers):
                if use_crossattn:
                    blocks.append(
                        _CrossAttnBlock(
                            self.hidden_dim,
                            config.generator_heads,
                            self.cond_dim,
                            use_adaln=use_adaln,
                            dropout=dropout_p,
                        )
                    )
                else:
                    blocks.append(
                        _NonAttnAdaLNBlock(
                            self.hidden_dim, self.cond_dim, dropout=dropout_p
                        )
                    )
            self.trunk_blocks = nn.ModuleList(blocks)

        self.U_head = nn.Linear(self.hidden_dim, self.r * self.h_head)

        self.V_heads = nn.ModuleDict()
        for bucket in self.D_IN_BUCKETS:
            self.V_heads[str(bucket)] = nn.Linear(self.hidden_dim, self.h_head * bucket)

        self.gamma_enabled = bool(config.dynamic_current_B)
        if self.gamma_enabled:
            self.gamma_head = nn.Linear(self.hidden_dim, config.h_d_llm)
        else:
            self.gamma_head = None

        self._zero_init_heads()

    def _zero_init_heads(self):

        nn.init.zeros_(self.U_head.bias)
        for head in self.V_heads.values():
            nn.init.zeros_(head.bias)
        if self.gamma_head is not None:
            nn.init.zeros_(self.gamma_head.weight)

            nn.init.constant_(self.gamma_head.bias, math.log(math.expm1(1.0)))

    @staticmethod
    def _apply_rope_half_split(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Apply half-split RoPE to x: (..., H), using cos and sin: (..., H/2)."""
        H = x.shape[-1]
        half = H // 2
        x1 = x[..., :half]
        x2 = x[..., half:]

        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        return torch.cat([rot1, rot2], dim=-1)

    def _build_slot_tokens(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
        domain_id: Optional[int] = None,
    ) -> torch.Tensor:
        if self.rope_layer_enabled:
            llm_targets = self.target_embed.unsqueeze(0).expand(
                self.num_hidden_layers, -1, -1
            )
            cos = self.rope_layer_cos.to(device=device, dtype=dtype)
            sin = self.rope_layer_sin.to(device=device, dtype=dtype)

            cos_b = cos.unsqueeze(1).expand(-1, self.num_targets, -1)
            sin_b = sin.unsqueeze(1).expand(-1, self.num_targets, -1)
            llm_targets_t = llm_targets.to(device=device, dtype=dtype)
            llm_slots = self._apply_rope_half_split(llm_targets_t, cos_b, sin_b)
            llm_slots = llm_slots.reshape(-1, self.hidden_dim)
        else:
            llm_slots = self.layer_embed.unsqueeze(1) + self.target_embed.unsqueeze(0)
            llm_slots = llm_slots.reshape(-1, self.hidden_dim)
            llm_slots = llm_slots.to(device=device, dtype=dtype)

        proj_slots = self.projector_slot_embed.to(device=device, dtype=dtype)
        all_slots = torch.cat([llm_slots, proj_slots], dim=0)

        if (
            self.domain_embed_enabled
            and self.domain_embed is not None
            and domain_id is not None
        ):
            d = int(domain_id)
            if 0 <= d < self.domain_embed.shape[0]:
                d_vec = self.domain_embed[d].to(device=device, dtype=dtype)
                all_slots = all_slots + d_vec.unsqueeze(0)

        return all_slots.unsqueeze(0).expand(batch, -1, -1).contiguous()

    def _pooled_cond(self, domain_features: torch.Tensor) -> torch.Tensor:
        """Pool the domain feature sequence to a (B, feature_dim) conditioning vector."""
        if domain_features.dim() == 2:
            return domain_features
        return domain_features.mean(dim=1)

    def forward_all_slots(
        self,
        domain_features: torch.Tensor,
        domain_id: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate factors for all slots.

        Returns U, V grouped by input width, optional gamma, bucket_slot_ids,
        and trunk representations.
        """

        if domain_features.dim() == 2:
            domain_features = domain_features.unsqueeze(1)
        B = domain_features.shape[0]
        device = domain_features.device
        dtype = next(self.U_head.parameters()).dtype
        slot_tokens = self._build_slot_tokens(B, device, dtype, domain_id=domain_id)

        if self.arch == "mlp_additive":
            trunk_out = self.mlp_trunk(domain_features)
            trunk_out = trunk_out.to(device=device, dtype=dtype)
        else:
            kv = self.kv_in(domain_features.to(dtype=dtype))
            cond = self._pooled_cond(domain_features).to(dtype=dtype)
            x = slot_tokens
            for block in self.trunk_blocks:
                if isinstance(block, _CrossAttnBlock):
                    x = block(x, kv, cond)
                else:
                    x = block(x, cond)
            trunk_out = x

        U_flat = self.U_head(trunk_out)
        U = U_flat.reshape(B, self.num_total_slots, self.r, self.h_head)

        bucket_slot_ids = self._bucket_slot_ids()
        V_by_bucket: Dict[str, torch.Tensor] = {}
        for bucket, slot_ids in bucket_slot_ids.items():
            if not slot_ids:
                V_by_bucket[str(bucket)] = torch.zeros(
                    B, 0, self.h_head, bucket, device=device, dtype=dtype
                )
                continue
            slot_idx_tensor = torch.tensor(slot_ids, device=device, dtype=torch.long)
            subset = trunk_out.index_select(dim=1, index=slot_idx_tensor)
            V_flat = self.V_heads[str(bucket)](subset)
            V_by_bucket[str(bucket)] = V_flat.reshape(
                B, len(slot_ids), self.h_head, bucket
            )

        gamma_out = None
        if self.gamma_head is not None:
            raw = self.gamma_head(trunk_out)
            gamma_out = F.softplus(raw)

        return {
            "U": U,
            "V": V_by_bucket,
            "gamma": gamma_out,
            "bucket_slot_ids": bucket_slot_ids,
            "trunk": trunk_out,
        }

    def forward_for_layer(
        self,
        layer_idx: int,
        domain_features: torch.Tensor,
        domain_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate U/V for one decoder layer.

        Returns U_layer, V_4096, V_11008, their target-index lists, and gamma_layer.
        Matches the corresponding all-slot outputs when dropout is disabled.
        """
        if domain_features.dim() == 2:
            domain_features = domain_features.unsqueeze(1)
        B = domain_features.shape[0]
        device = domain_features.device
        dtype = next(self.U_head.parameters()).dtype

        if self.rope_layer_enabled:
            cos = self.rope_layer_cos[layer_idx].to(device=device, dtype=dtype)
            sin = self.rope_layer_sin[layer_idx].to(device=device, dtype=dtype)
            target_e = self.target_embed.to(device=device, dtype=dtype)
            cos_b = cos.unsqueeze(0).expand(self.num_targets, -1)
            sin_b = sin.unsqueeze(0).expand(self.num_targets, -1)
            slot_token_rows = self._apply_rope_half_split(target_e, cos_b, sin_b)
        else:
            slot_token_rows = (
                self.layer_embed[layer_idx].unsqueeze(0) + self.target_embed
            ).to(device=device, dtype=dtype)

        if (
            self.domain_embed_enabled
            and self.domain_embed is not None
            and domain_id is not None
        ):
            d = int(domain_id)
            if 0 <= d < self.domain_embed.shape[0]:
                slot_token_rows = slot_token_rows + self.domain_embed[d].to(
                    device=device, dtype=dtype
                ).unsqueeze(0)
        slot_tokens = slot_token_rows.unsqueeze(0).expand(B, -1, -1).contiguous()

        if self.arch == "mlp_additive":
            h = self.mlp_trunk.trunk(domain_features.mean(dim=1).to(dtype=dtype))
            slot_embed_subset = self.mlp_trunk.slot_embed[
                _llm_slot_id(layer_idx, 0) : _llm_slot_id(layer_idx, 0)
                + self.num_targets
            ]
            trunk_out = h.unsqueeze(1) + slot_embed_subset.unsqueeze(0)
        else:
            kv = self.kv_in(domain_features.to(dtype=dtype))
            cond = self._pooled_cond(domain_features).to(dtype=dtype)
            x = slot_tokens
            for block in self.trunk_blocks:
                if isinstance(block, _CrossAttnBlock):
                    x = block(x, kv, cond)
                else:
                    x = block(x, cond)
            trunk_out = x

        U_flat = self.U_head(trunk_out)
        U_layer = U_flat.reshape(B, self.num_targets, self.r, self.h_head)

        v_4096_targets: List[int] = []
        v_11008_targets: List[int] = []
        for t_idx, target in enumerate(DCAG_TARGET_MODULES):
            if target == "down_proj":
                v_11008_targets.append(t_idx)
            else:
                v_4096_targets.append(t_idx)

        if v_4096_targets:
            t_idx_t = torch.tensor(v_4096_targets, device=device, dtype=torch.long)
            subset = trunk_out.index_select(dim=1, index=t_idx_t)
            V_4096 = self.V_heads["4096"](subset).reshape(
                B, len(v_4096_targets), self.h_head, 4096
            )
        else:
            V_4096 = torch.zeros(B, 0, self.h_head, 4096, device=device, dtype=dtype)

        if v_11008_targets:
            t_idx_t = torch.tensor(v_11008_targets, device=device, dtype=torch.long)
            subset = trunk_out.index_select(dim=1, index=t_idx_t)
            V_11008 = self.V_heads["11008"](subset).reshape(
                B, len(v_11008_targets), self.h_head, 11008
            )
        else:
            V_11008 = torch.zeros(B, 0, self.h_head, 11008, device=device, dtype=dtype)

        gamma_layer = None
        if self.gamma_head is not None:
            gamma_layer = F.softplus(self.gamma_head(trunk_out))

        return {
            "U_layer": U_layer,
            "V_4096": V_4096,
            "V_4096_targets": v_4096_targets,
            "V_11008": V_11008,
            "V_11008_targets": v_11008_targets,
            "gamma_layer": gamma_layer,
        }

    def forward_for_projector(
        self,
        domain_features: torch.Tensor,
        domain_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate factors for projector slots 224 and 225 (input widths 1024 and 4096)."""
        if domain_features.dim() == 2:
            domain_features = domain_features.unsqueeze(1)
        B = domain_features.shape[0]
        device = domain_features.device
        dtype = next(self.U_head.parameters()).dtype

        proj_e = self.projector_slot_embed.to(device=device, dtype=dtype)
        if (
            self.domain_embed_enabled
            and self.domain_embed is not None
            and domain_id is not None
        ):
            d = int(domain_id)
            if 0 <= d < self.domain_embed.shape[0]:
                proj_e = proj_e + self.domain_embed[d].to(
                    device=device, dtype=dtype
                ).unsqueeze(0)
        slot_tokens = proj_e.unsqueeze(0).expand(B, -1, -1).contiguous()

        if self.arch == "mlp_additive":
            h = self.mlp_trunk.trunk(domain_features.mean(dim=1).to(dtype=dtype))
            slot_embed_subset = self.mlp_trunk.slot_embed[PROJECTOR_SLOT_OFFSET:]
            trunk_out = h.unsqueeze(1) + slot_embed_subset.unsqueeze(0)
        else:
            kv = self.kv_in(domain_features.to(dtype=dtype))
            cond = self._pooled_cond(domain_features).to(dtype=dtype)
            x = slot_tokens
            for block in self.trunk_blocks:
                if isinstance(block, _CrossAttnBlock):
                    x = block(x, kv, cond)
                else:
                    x = block(x, cond)
            trunk_out = x

        U_flat = self.U_head(trunk_out)
        U_proj = U_flat.reshape(B, NUM_PROJECTOR_SLOTS, self.r, self.h_head)

        bucket_slot_ids = self._bucket_slot_ids()
        v_1024_local: List[int] = []
        v_4096_local: List[int] = []
        for proj_local in range(NUM_PROJECTOR_SLOTS):
            slot_id = _projector_slot_id(proj_local)
            for bucket, slot_list in bucket_slot_ids.items():
                if slot_id in slot_list:
                    if bucket == 1024:
                        v_1024_local.append(proj_local)
                    elif bucket == 4096:
                        v_4096_local.append(proj_local)
                    break

        def _proj_V(local_idxs: List[int], bucket: int) -> torch.Tensor:
            if not local_idxs:
                return torch.zeros(
                    B, 0, self.h_head, bucket, device=device, dtype=dtype
                )
            t_idx_t = torch.tensor(local_idxs, device=device, dtype=torch.long)
            subset = trunk_out.index_select(dim=1, index=t_idx_t)
            return self.V_heads[str(bucket)](subset).reshape(
                B, len(local_idxs), self.h_head, bucket
            )

        V_1024 = _proj_V(v_1024_local, 1024)
        V_4096 = _proj_V(v_4096_local, 4096)

        gamma_proj = None
        if self.gamma_head is not None:
            gamma_proj = F.softplus(self.gamma_head(trunk_out))

        return {
            "U_proj": U_proj,
            "V_1024": V_1024,
            "V_1024_local": v_1024_local,
            "V_4096": V_4096,
            "V_4096_local": v_4096_local,
            "gamma_proj": gamma_proj,
        }

    def _bucket_slot_ids(self) -> Dict[int, List[int]]:
        """Group slot indices by input width for the decoder and projector."""
        if (
            hasattr(self, "_bucket_slot_ids_cache")
            and self._bucket_slot_ids_cache is not None
        ):
            return self._bucket_slot_ids_cache
        buckets: Dict[int, List[int]] = {b: [] for b in self.D_IN_BUCKETS}

        for layer_idx in range(self.num_hidden_layers):
            for t_idx, target in enumerate(DCAG_TARGET_MODULES):
                slot_id = _llm_slot_id(layer_idx, t_idx)
                if target == "down_proj":
                    buckets[11008].append(slot_id)
                else:
                    buckets[4096].append(slot_id)

        buckets[1024].append(_projector_slot_id(0))
        buckets[4096].append(_projector_slot_id(1))
        self._bucket_slot_ids_cache = buckets
        return buckets

    def set_bucket_slot_ids(self, buckets: Dict[int, List[int]]):
        """Set slot groups from the wrapped projections' input widths."""
        self._bucket_slot_ids_cache = buckets


class _NonAttnAdaLNBlock(nn.Module):
    """AdaLN + MLP without cross-attention (for generator_arch='adaln_mlp')."""

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _AdaLN(hidden_dim, cond_dim)
        self.norm2 = _AdaLN(hidden_dim, cond_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.ffn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, slots: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm1(slots, cond)
        x = slots + h
        h = self.norm2(x, cond)
        return x + self.ffn_dropout(self.ffn(h))


class FisherSubspaceTracker(nn.Module):
    def __init__(
        self,
        d_in: int,
        r_fisher: int,
        mode: str = "fisher_oja_per_slot",
        lr_oja: float = 0.01,
        offline_buffer_size: int = 1024,
    ):
        super().__init__()
        self.d_in = d_in
        self.r_fisher = r_fisher
        self.mode = mode
        self.lr_oja = lr_oja
        self.offline_buffer_size = offline_buffer_size

        self.register_buffer("V", self._init_V(), persistent=False)
        self.register_buffer(
            "num_updates", torch.zeros(1, dtype=torch.long), persistent=False
        )
        if mode == "fisher_offline":
            self.register_buffer(
                "_x_ring", torch.zeros(offline_buffer_size, d_in), persistent=False
            )
            self.register_buffer(
                "_g_ring", torch.zeros(offline_buffer_size, d_in), persistent=False
            )
            self.register_buffer(
                "_ring_cursor", torch.zeros(1, dtype=torch.long), persistent=False
            )

    def _init_V(self) -> torch.Tensor:
        v = torch.randn(self.d_in, self.r_fisher) * (1.0 / math.sqrt(self.d_in))
        q = _safe_qr(v)
        return q

    @torch.no_grad()
    def update(self, x_slot: torch.Tensor, grad_x_slot: Optional[torch.Tensor]):
        """x_slot: (N, d_in); grad_x_slot: (N, d_in) or None."""
        if self.mode == "none_disable_A_projection":
            return
        if x_slot.dim() > 2:
            x_slot = x_slot.reshape(-1, self.d_in)
        if grad_x_slot is not None and grad_x_slot.dim() > 2:
            grad_x_slot = grad_x_slot.reshape(-1, self.d_in)
        device = self.V.device
        dtype = self.V.dtype
        x_slot = x_slot.detach().to(device=device, dtype=dtype)
        if grad_x_slot is not None:
            grad_x_slot = grad_x_slot.detach().to(device=device, dtype=dtype)

        if self.mode == "fisher_offline":
            N = x_slot.shape[0]
            start = int(self._ring_cursor.item())
            for i in range(N):
                pos = (start + i) % self.offline_buffer_size
                self._x_ring[pos] = x_slot[i]
                if grad_x_slot is not None:
                    self._g_ring[pos] = grad_x_slot[i]
            self._ring_cursor[0] = (start + N) % self.offline_buffer_size
            self.num_updates += 1
            return

        if self.mode == "activation_pca_per_slot":
            signal = x_slot.mean(dim=0)
        elif self.mode == "fisher_oja_per_slot":
            if grad_x_slot is None:
                signal = x_slot.mean(dim=0)
            else:
                signal = grad_x_slot.mean(dim=0)
        else:
            return

        if torch.isnan(signal).any() or torch.isinf(signal).any():
            return
        signal_norm = signal.norm()
        if signal_norm < 1e-12:
            return

        scores = signal @ self.V
        self.V.add_(self.lr_oja * torch.outer(signal, scores))
        try:
            q = _safe_qr(self.V)
            self.V.copy_(q)
        except RuntimeError:
            pass
        self.num_updates += 1

    @torch.no_grad()
    def update_signal(self, signal: torch.Tensor):
        """Update the online subspace from a reduced (d_in,) signal.

        Offline covariance tracking uses update() with full activation tensors.
        """
        if self.mode in ("none_disable_A_projection", "fisher_offline"):
            return
        signal = signal.detach().to(device=self.V.device, dtype=self.V.dtype)
        if signal.dim() != 1 or signal.shape[0] != self.d_in:
            return
        if torch.isnan(signal).any() or torch.isinf(signal).any():
            return
        if signal.norm() < 1e-12:
            return
        scores = signal @ self.V
        self.V.add_(self.lr_oja * torch.outer(signal, scores))
        try:
            q = _safe_qr(self.V)
            self.V.copy_(q)
        except RuntimeError:
            pass
        self.num_updates += 1

    @torch.no_grad()
    def extract(self, k_d: int, P_prev: Optional[torch.Tensor]) -> torch.Tensor:
        """Return up to k_d directions orthogonal to P_prev.

        Return an empty basis when no updates were observed or projection is disabled.
        """
        if self.mode == "none_disable_A_projection":
            return torch.zeros(self.d_in, 0, device=self.V.device, dtype=self.V.dtype)
        if int(self.num_updates.item()) == 0:
            return torch.zeros(self.d_in, 0, device=self.V.device, dtype=self.V.dtype)
        if self.mode == "fisher_offline":
            device = self.V.device
            dtype = self.V.dtype
            g = self._g_ring.to(device=device, dtype=dtype)
            cov = g.t() @ g
            try:
                U_svd = _safe_svd_U(cov)
                W = U_svd[:, :k_d]
            except RuntimeError:
                W = self.V[:, :k_d]
            return self._orthonormalize_against_prev(W, P_prev)

        W = self.V[:, :k_d]
        return self._orthonormalize_against_prev(W, P_prev)

    @torch.no_grad()
    def _orthonormalize_against_prev(
        self, W: torch.Tensor, P_prev: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if P_prev is None or P_prev.shape[1] == 0:
            return _safe_qr(W)
        combined = torch.cat([P_prev.to(W.dtype).to(W.device), W], dim=1)
        q = _safe_qr(combined)

        return q[:, P_prev.shape[1] : P_prev.shape[1] + W.shape[1]].contiguous()

    @torch.no_grad()
    def sync_across_ranks(self):
        """Merge per-rank subspaces through covariance eigendecomposition.

        Sum update counts across ranks to detect globally unobserved slots.
        """
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return

        if not self.V.is_cuda:
            return

        v_input = self.V.detach().contiguous()
        gathered = [
            torch.empty(self.V.shape, dtype=self.V.dtype, device=self.V.device)
            for _ in range(world_size)
        ]
        torch.distributed.all_gather(gathered, v_input)
        stacked = torch.cat(gathered, dim=1)

        try:
            stacked_f = stacked.float()
            cov = stacked_f.t() @ stacked_f
            evals, evecs = torch.linalg.eigh(cov)
            r = self.V.shape[1]
            top = torch.flip(
                torch.arange(cov.shape[1] - r, cov.shape[1], device=cov.device),
                dims=[0],
            )
            top_evecs = evecs.index_select(dim=1, index=top)
            new_V = stacked_f @ top_evecs
            new_V = _safe_qr(new_V)
            self.V.copy_(new_V.to(self.V.dtype))
        except RuntimeError:
            pass

        nu = self.num_updates.clone()
        torch.distributed.all_reduce(nu, op=torch.distributed.ReduceOp.SUM)
        self.num_updates.copy_(nu)

    @torch.no_grad()
    def reset(self):
        self.V.copy_(self._init_V().to(self.V.device, self.V.dtype))
        self.num_updates.zero_()
        if self.mode == "fisher_offline":
            self._x_ring.zero_()
            self._g_ring.zero_()
            self._ring_cursor.zero_()


class TaskSliceManager(nn.Module):
    def __init__(
        self,
        r: int,
        h_d_per_kind: Dict[str, int],
        kind_per_slot: Dict[int, str],
        d_out_per_slot: Dict[int, int],
        B_mode: str = "column_partitioned_CR",
        R_retraction: str = "qr",
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.r = r
        self.h_d_per_kind = dict(h_d_per_kind)
        self.kind_per_slot = dict(kind_per_slot)
        self.d_out_per_slot = dict(d_out_per_slot)
        self.B_mode = B_mode
        self.R_retraction = R_retraction
        self.slot_ids = sorted(self.kind_per_slot.keys())
        self.kinds = tuple(sorted(set(self.kind_per_slot.values())))

        for s in self.slot_ids:
            kind = self.kind_per_slot[s]
            h_d = self.h_d_per_kind[kind]
            d_out = self.d_out_per_slot[s]
            self.register_buffer(
                f"R_slot_{s}_frozen", torch.zeros(r, 0, device=device, dtype=dtype)
            )
            setattr(
                self,
                f"R_slot_{s}_current",
                nn.Parameter(torch.zeros(r, h_d, device=device, dtype=dtype)),
            )
            self.register_buffer(
                f"C_slot_{s}_frozen", torch.zeros(d_out, 0, device=device, dtype=dtype)
            )
            setattr(
                self,
                f"C_slot_{s}_current",
                nn.Parameter(torch.zeros(d_out, h_d, device=device, dtype=dtype)),
            )

        if B_mode == "plain_B":
            self.plain_B = nn.ParameterDict()
            for s in self.slot_ids:
                d_out = self.d_out_per_slot[s]
                self.plain_B[f"slot_{s}"] = nn.Parameter(
                    torch.zeros(d_out, r, device=device, dtype=dtype)
                )

        self.task_slice_map: Dict[int, Dict[str, Tuple[int, int]]] = {}

        self.domain_to_task: Dict[int, int] = {}
        self.task_to_domain: Dict[int, int] = {}

    def get_extra_state(self) -> Dict[str, Any]:
        return {
            "task_slice_map": self.task_slice_map,
            "domain_to_task": self.domain_to_task,
            "task_to_domain": self.task_to_domain,
        }

    def set_extra_state(self, state: Dict[str, Any]):
        raw = state.get("task_slice_map", {})

        self.task_slice_map = {
            int(k): {kk: tuple(vv) for kk, vv in v.items()} for k, v in raw.items()
        }
        d2t = state.get("domain_to_task", {})
        self.domain_to_task = {int(k): int(v) for k, v in d2t.items()}
        t2d = state.get("task_to_domain", {})
        self.task_to_domain = {int(k): int(v) for k, v in t2d.items()}

    def record_domain_for_task(self, task_idx: int, domain_id: int):
        self.domain_to_task[int(domain_id)] = int(task_idx)
        self.task_to_domain[int(task_idx)] = int(domain_id)

    def cursor_for(self, kind: str) -> int:

        for s in self.slot_ids:
            if self.kind_per_slot[s] == kind:
                return int(getattr(self, f"R_slot_{s}_frozen").shape[1])
        return 0

    def completed_task_indices(self) -> List[int]:
        return sorted(self.task_slice_map.keys())

    def infer_next_task_idx(self) -> int:
        if not self.task_slice_map:
            return 0
        return max(self.task_slice_map.keys()) + 1

    @torch.no_grad()
    def start_task(self, task_idx: int):
        """Initialize new slices orthogonal to frozen slices, after checkpoint loading."""
        self.task_slice_map[task_idx] = {}
        cursor_per_kind: Dict[str, int] = {}
        for s in self.slot_ids:
            kind = self.kind_per_slot[s]
            h_d = self.h_d_per_kind[kind]
            R_frozen = getattr(self, f"R_slot_{s}_frozen")
            cursor = int(R_frozen.shape[1])
            cursor_per_kind[kind] = cursor
            device = R_frozen.device
            dtype = R_frozen.dtype
            rand_init = torch.randn(self.r, h_d, device=device, dtype=dtype)
            if cursor > 0:
                combined = torch.cat([R_frozen, rand_init], dim=1)
                q = _safe_qr(combined)
                R_cur_init = q[:, cursor : cursor + h_d].contiguous()
            else:
                q = _safe_qr(rand_init)
                R_cur_init = q
            param = getattr(self, f"R_slot_{s}_current")
            param.data.copy_(R_cur_init)
            param.requires_grad_(True)

            c_param = getattr(self, f"C_slot_{s}_current")
            c_param.data.zero_()
            c_param.requires_grad_(True)

        for kind, cursor in cursor_per_kind.items():
            h_d = self.h_d_per_kind[kind]
            self.task_slice_map[task_idx][kind] = (cursor, cursor + h_d)

    @torch.no_grad()
    def end_task(self, task_idx: int):
        """Append current slices to frozen buffers and disable their gradients."""
        for s in self.slot_ids:
            R_frozen = getattr(self, f"R_slot_{s}_frozen")
            R_cur = getattr(self, f"R_slot_{s}_current").data
            self._buffers[f"R_slot_{s}_frozen"] = torch.cat([R_frozen, R_cur], dim=1)
            getattr(self, f"R_slot_{s}_current").data.zero_()
            getattr(self, f"R_slot_{s}_current").requires_grad_(False)

            C_frozen = getattr(self, f"C_slot_{s}_frozen")
            C_cur = getattr(self, f"C_slot_{s}_current").data
            self._buffers[f"C_slot_{s}_frozen"] = torch.cat([C_frozen, C_cur], dim=1)
            getattr(self, f"C_slot_{s}_current").data.zero_()
            getattr(self, f"C_slot_{s}_current").requires_grad_(False)

    @torch.no_grad()
    def retract_R_current(self) -> Optional[Tuple[float, float]]:
        """Orthonormalize current slices against frozen slices.

        Return maximum cross-slice and within-slice orthogonality errors.
        """
        if self.R_retraction == "none":
            return None
        if self.R_retraction == "householder":
            _warn_once(
                "R_retraction_householder",
                "R_retraction='householder' is not implemented; falling back to QR retraction.",
            )
        max_off_diag_global = 0.0
        max_off_orth_global = 0.0
        for s in self.slot_ids:
            R_frozen = getattr(self, f"R_slot_{s}_frozen")
            R_cur_param = getattr(self, f"R_slot_{s}_current")
            if R_cur_param.shape[1] == 0 or not R_cur_param.requires_grad:
                continue
            R_cur = R_cur_param.data
            if R_frozen.shape[1] > 0:
                R_cur_proj = R_cur - R_frozen @ (R_frozen.t() @ R_cur)
            else:
                R_cur_proj = R_cur
            try:
                q = _safe_qr(R_cur_proj)
                R_cur_param.data.copy_(q)
            except RuntimeError:
                continue

            try:
                cur = R_cur_param.data
                if R_frozen.shape[1] > 0:
                    cross = R_frozen.t() @ cur
                    max_off_diag_global = max(
                        max_off_diag_global, float(cross.abs().max().item())
                    )
                gram = cur.t() @ cur
                eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
                max_off_orth_global = max(
                    max_off_orth_global, float((gram - eye).abs().max().item())
                )
            except RuntimeError:
                pass
        return (max_off_diag_global, max_off_orth_global)

    def B_effective(self, slot_id: int, counter: int) -> torch.Tensor:
        """Compose B = C_full @ R_full.T; the controller handles per-batch caching."""
        if self.B_mode == "plain_B":
            return self.plain_B[f"slot_{slot_id}"]
        R_frozen = getattr(self, f"R_slot_{slot_id}_frozen")
        R_cur = getattr(self, f"R_slot_{slot_id}_current")
        R_full = torch.cat([R_frozen, R_cur], dim=1)
        C_frozen = getattr(self, f"C_slot_{slot_id}_frozen")
        C_cur = getattr(self, f"C_slot_{slot_id}_current")
        C_full = torch.cat([C_frozen, C_cur], dim=1)
        return C_full @ R_full.t()

    def B_current_slice(self, slot_id: int) -> Optional[torch.Tensor]:
        if self.B_mode == "plain_B":
            return None
        R_cur = getattr(self, f"R_slot_{slot_id}_current", None)
        C_cur = getattr(self, f"C_slot_{slot_id}_current", None)
        if R_cur is None or C_cur is None:
            return None
        if R_cur.shape[1] == 0 or C_cur.shape[1] == 0:
            return None
        if not R_cur.requires_grad and not C_cur.requires_grad:
            return None
        return C_cur @ R_cur.t()

    def _B_effective_prefix(self, slot_id: int, end_col: int) -> torch.Tensor:
        """Compose B from frozen columns before end_col."""
        if self.B_mode == "plain_B":
            return self.plain_B[f"slot_{slot_id}"]

        R_frozen = getattr(self, f"R_slot_{slot_id}_frozen")
        C_frozen = getattr(self, f"C_slot_{slot_id}_frozen")
        end_col = max(
            0, min(int(end_col), int(R_frozen.shape[1]), int(C_frozen.shape[1]))
        )
        if end_col <= 0:
            return torch.zeros(
                C_frozen.shape[0],
                self.r,
                device=C_frozen.device,
                dtype=C_frozen.dtype,
            )
        return C_frozen[:, :end_col] @ R_frozen[:, :end_col].t()

    def B_effective_for_domain(
        self, slot_id: int, domain_id: Optional[int]
    ) -> torch.Tensor:
        if self.B_mode == "plain_B" or domain_id is None:
            return self.B_effective(slot_id, 0)

        task_idx = self.domain_to_task.get(int(domain_id), int(domain_id))
        task_map = self.task_slice_map.get(int(task_idx), None)
        kind = self.kind_per_slot[slot_id]
        if not task_map or kind not in task_map:
            _warn_once(
                "B_prefix_missing_task_slice_map",
                "DCAG domain_prefix composition requested, but task_slice_map is "
                "missing the target domain/kind; falling back to full B_eff.",
            )
            return self.B_effective(slot_id, 0)
        _, end_col = task_map[kind]
        return self._B_effective_prefix(slot_id, end_col)


class ResidualLoRAManager(nn.Module):
    def __init__(
        self,
        r_res: int,
        alpha: float,
        d_in_per_slot: Dict[int, int],
        d_out_per_slot: Dict[int, int],
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.r_res = int(r_res)

        self.scaling = float(alpha) / max(self.r_res, 1)
        self.slot_ids = sorted(d_in_per_slot.keys())

        for s in self.slot_ids:
            a_init = torch.empty(
                self.r_res, d_in_per_slot[s], device=device, dtype=dtype
            )
            nn.init.normal_(a_init, mean=0.0, std=0.01)
            setattr(self, f"residual_A_slot_{s}_current", nn.Parameter(a_init))
            setattr(
                self,
                f"residual_B_slot_{s}_current",
                nn.Parameter(
                    torch.zeros(
                        d_out_per_slot[s], self.r_res, device=device, dtype=dtype
                    )
                ),
            )

    @torch.no_grad()
    def start_task(self, task_idx: int):
        """Initialize the current residual with Gaussian A and zero B; enable gradients."""
        for s in self.slot_ids:
            a = getattr(self, f"residual_A_slot_{s}_current")
            a.data.normal_(mean=0.0, std=0.01)
            a.requires_grad_(True)
            b = getattr(self, f"residual_B_slot_{s}_current")
            b.data.zero_()
            b.requires_grad_(True)

    @torch.no_grad()
    def end_task(self, task_idx: int):
        """Save the current residual as frozen task buffers and clear active parameters."""
        task_idx = int(task_idx)
        for s in self.slot_ids:
            a = getattr(self, f"residual_A_slot_{s}_current")
            self.register_buffer(f"residual_A_slot_{s}_task_{task_idx}", a.data.clone())
            a.data.zero_()
            a.requires_grad_(False)
            b = getattr(self, f"residual_B_slot_{s}_current")
            self.register_buffer(f"residual_B_slot_{s}_task_{task_idx}", b.data.clone())
            b.data.zero_()
            b.requires_grad_(False)

    def current_AB(self, slot_id: int) -> Tuple[nn.Parameter, nn.Parameter]:
        """Trainable pair of the task currently being trained."""
        return (
            getattr(self, f"residual_A_slot_{slot_id}_current"),
            getattr(self, f"residual_B_slot_{slot_id}_current"),
        )

    def task_AB(
        self, slot_id: int, task_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return a completed task's frozen residual pair, or None if unavailable."""
        a_name = f"residual_A_slot_{slot_id}_task_{int(task_idx)}"
        b_name = f"residual_B_slot_{slot_id}_task_{int(task_idx)}"
        if a_name not in self._buffers or b_name not in self._buffers:
            return None
        return self._buffers[a_name], self._buffers[b_name]


class DCAGLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        controller: "DCAGController",
        layer_idx: int,
        target_name: Optional[str],
        kind: str,
        projector_idx: Optional[int] = None,
        slot_id: Optional[int] = None,
        alpha: Optional[float] = None,
    ):
        super().__init__()
        self.base_layer = base_layer

        object.__setattr__(self, "controller", controller)
        self.layer_idx = layer_idx
        self.target_name = target_name
        self.kind = kind
        self.projector_idx = projector_idx
        if slot_id is None:
            if kind == "llm":
                t_idx = DCAG_TARGET_MODULES.index(target_name)
                self.target_idx = t_idx
                self.slot_id = _llm_slot_id(layer_idx, t_idx)
            else:
                self.target_idx = -1
                self.slot_id = _projector_slot_id(projector_idx)
        else:
            self.slot_id = slot_id
            self.target_idx = (
                -1 if kind == "projector" else DCAG_TARGET_MODULES.index(target_name)
            )

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features

        for p in base_layer.parameters():
            p.requires_grad = False

        effective_alpha = alpha if alpha is not None else float(2 * controller.config.r)
        self.scaling = effective_alpha / max(controller.config.r, 1)
        self._cached_batch_counter: int = -2

        self._x_slot_cache: Optional[torch.Tensor] = None

    @property
    def d_in_bucket(self) -> int:

        if self.in_features in (1024, 4096, 11008):
            return self.in_features

        for b in (1024, 4096, 11008):
            if b == self.in_features:
                return b
        raise ValueError(f"Unsupported d_in {self.in_features} at slot {self.slot_id}")

    @property
    def d_out_bucket(self) -> int:
        return self.out_features

    def _apply_projection(self, A_tilde: torch.Tensor) -> torch.Tensor:
        """Project A_tilde: (B, r, d_in) onto the complement of P: (d_in, k)."""
        ctrl = self.controller
        if ctrl.config.Wd_mode == "none_disable_A_projection":
            return A_tilde

        P = ctrl.get_P_for_domain(self.slot_id, ctrl.current_domain_id)
        if P is None or P.shape[1] == 0:
            return A_tilde

        P_local = P.to(device=A_tilde.device, dtype=A_tilde.dtype)
        A_proj = A_tilde @ P_local
        return A_tilde - A_proj @ P_local.t()

    def _on_y_grad(self, grad_y: torch.Tensor):
        """Update the Fisher tracker from output-gradient-weighted input activations.

        The online signal is the token mean of ||grad_y||_2 * x.
        """
        tracker = self.controller.get_fisher_tracker(self.slot_id, self.d_in_bucket)
        if tracker is None or self._x_slot_cache is None:
            self._x_slot_cache = None
            return
        try:
            x = self._x_slot_cache
            g = grad_y.detach()

            norm = g.norm(dim=-1).to(dtype=x.dtype, device=x.device)
            x_flat = x.reshape(-1, x.shape[-1])
            norm_flat = norm.reshape(-1)
            if tracker.mode == "fisher_offline":
                weighted_flat = x_flat * norm_flat.unsqueeze(-1)
                tracker.update(x_flat, weighted_flat)
            else:
                N = max(x_flat.shape[0], 1)
                signal = (norm_flat.unsqueeze(0) @ x_flat).squeeze(0) / N
                tracker.update_signal(signal)
        finally:
            self._x_slot_cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        ctrl = self.controller
        if not ctrl.has_batch_context():
            return base_out

        if self.training and ctrl.config.Wd_mode != "none_disable_A_projection":
            self._x_slot_cache = x.detach()
        else:
            self._x_slot_cache = None

        if not self.training and self.slot_id in ctrl._a_proj_cache:
            A = ctrl._a_proj_cache[self.slot_id].to(device=x.device, dtype=x.dtype)
        else:
            U, V = ctrl.get_slot_UV(self.slot_id, self.d_in_bucket, x.device, x.dtype)
            if U is None or V is None:
                return base_out

            A_tilde = torch.einsum("brh,bhd->brd", U, V)
            A = self._apply_projection(A_tilde)

        B_eff = ctrl.get_B_effective(self.slot_id, x.device, x.dtype)

        if (
            ctrl.config.dynamic_current_B
            and self.kind == "llm"
            and ctrl.config.B_mode == "column_partitioned_CR"
        ):
            gamma = ctrl.get_slot_gamma(self.slot_id, x.device, x.dtype)
            if gamma is not None:
                current_range = ctrl.current_B_column_range(self.kind)
                if current_range is not None:
                    B_eff = self._apply_gamma_to_current(
                        B_eff, gamma, current_range, ctrl
                    )

        if x.dim() == 2:
            delta = torch.einsum("bi,bri->br", x, A)
            delta = torch.einsum("br,or->bo", delta, B_eff)
        else:
            delta = torch.einsum("bsi,bri->bsr", x, A)
            delta = torch.einsum("bsr,or->bso", delta, B_eff)

        if (
            self.training
            and self.kind == "projector"
            and ctrl.config.projector_warmup_steps > 0
            and getattr(ctrl, "current_task_idx", None) == 0
        ):
            step = ctrl.global_step
            ramp = min(1.0, float(step + 1) / float(ctrl.config.projector_warmup_steps))
            delta = delta * ramp

        res_pair = ctrl.get_residual_AB(self.slot_id, x.device, x.dtype)
        res_delta = None
        if res_pair is not None:
            A_res, B_res = res_pair
            if x.dim() == 2:
                res_delta = torch.einsum("bi,ri->br", x, A_res)
                res_delta = torch.einsum("br,or->bo", res_delta, B_res)
            else:
                res_delta = torch.einsum("bsi,ri->bsr", x, A_res)
                res_delta = torch.einsum("bsr,or->bso", res_delta, B_res)

        output = base_out + delta * self.scaling
        if res_delta is not None:
            output = output + res_delta * ctrl.residual_lora_manager.scaling

        if self.training and ctrl.config.Wd_mode != "none_disable_A_projection":
            if output.requires_grad:
                output.register_hook(self._on_y_grad)

        return output

    def _apply_gamma_to_current(
        self,
        B_eff: torch.Tensor,
        gamma: torch.Tensor,
        current_range: Tuple[int, int],
        ctrl: "DCAGController",
    ) -> torch.Tensor:
        """Scale current output-slice columns by batch-mean gamma, then compose B."""

        tsm = ctrl.task_slice_manager
        s = self.slot_id
        C_frozen = getattr(tsm, f"C_slot_{s}_frozen").to(
            device=B_eff.device, dtype=B_eff.dtype
        )
        C_cur = getattr(tsm, f"C_slot_{s}_current").to(
            device=B_eff.device, dtype=B_eff.dtype
        )
        R_frozen = getattr(tsm, f"R_slot_{s}_frozen").to(
            device=B_eff.device, dtype=B_eff.dtype
        )
        R_cur = getattr(tsm, f"R_slot_{s}_current").to(
            device=B_eff.device, dtype=B_eff.dtype
        )

        gamma_mean = gamma.mean(dim=0)
        C_cur_scaled = C_cur * gamma_mean.unsqueeze(0)
        return C_frozen @ R_frozen.t() + C_cur_scaled @ R_cur.t()


class DCAGController(nn.Module):
    """Manage domain features, generated factors, continual state, and per-batch caches."""

    def __init__(self, config: DCAGConfig, num_hidden_layers: int):
        super().__init__()
        self.config = config
        self.num_hidden_layers = num_hidden_layers
        self.num_targets = len(DCAG_TARGET_MODULES)
        self.num_total_slots = (
            num_hidden_layers * self.num_targets + NUM_PROJECTOR_SLOTS
        )
        self.anti_forgetting_weight = float(config.anti_forgetting_weight)

        self.feature_encoder = DCAGDomainFeatureEncoder(config)

        self.generator = DCAGGenerator(config, num_hidden_layers)

        self.alignment_head = nn.Linear(config.feature_dim, 1024)
        self._alignment_student_pooled: Optional[torch.Tensor] = None

        self.instruction_conditioning = bool(
            getattr(config, "instruction_conditioning", False)
        )
        if self.instruction_conditioning:
            self.instruction_proj = nn.Linear(
                config.instruction_feature_dim, config.feature_dim
            )
            self.instruction_gate = nn.Parameter(torch.zeros(1))
        else:
            self.instruction_proj = None
            self.instruction_gate = None
        self.current_instruction_token: Optional[torch.Tensor] = None

        if self.generator.domain_embed is not None:
            ctrl_self = self

            def _domain_embed_grad_mask(grad: torch.Tensor) -> torch.Tensor:
                d = ctrl_self.current_domain_id
                if d is None:
                    return torch.zeros_like(grad)
                mask = torch.zeros_like(grad)
                mask[int(d)] = 1.0
                return grad * mask

            self.generator.domain_embed.register_hook(_domain_embed_grad_mask)

        kind_per_slot: Dict[int, str] = {}
        d_out_per_slot: Dict[int, int] = {}
        for slot_id in range(self.num_total_slots):
            if slot_id >= NUM_LLM_SLOTS:
                kind_per_slot[slot_id] = "projector"
                d_out_per_slot[slot_id] = _LLAMA_7B_HIDDEN
            else:
                kind_per_slot[slot_id] = "llm"
                t_idx = slot_id % len(DCAG_TARGET_MODULES)
                target = DCAG_TARGET_MODULES[t_idx]
                d_out_per_slot[slot_id] = (
                    _LLAMA_7B_INTERMEDIATE
                    if target in ("gate_proj", "up_proj")
                    else _LLAMA_7B_HIDDEN
                )
        self.task_slice_manager = TaskSliceManager(
            r=config.r,
            h_d_per_kind={"llm": config.h_d_llm, "projector": config.h_d_projector},
            kind_per_slot=kind_per_slot,
            d_out_per_slot=d_out_per_slot,
            B_mode=config.B_mode,
            R_retraction=config.R_retraction,
        )

        self.residual_lora_manager: Optional[ResidualLoRAManager] = None
        if int(getattr(config, "residual_rank", 0) or 0) > 0:
            self.residual_lora_manager = ResidualLoRAManager(
                r_res=int(config.residual_rank),
                alpha=float(config.residual_alpha),
                d_in_per_slot={
                    s: self._slot_d_in(s) for s in range(self.num_total_slots)
                },
                d_out_per_slot=d_out_per_slot,
            )

        self.centroid_manager = PrototypeCentroidManager(
            num_domains=len(DCAG_DOMAIN_TO_ID),
            num_centroids=config.num_centroids,
            feature_dim=config.feature_dim,
            num_centroids_per_domain=config.num_centroids_per_domain,
        )

        self.fisher_trackers = nn.ModuleDict()
        if config.Wd_mode == "none_disable_A_projection":
            pass
        elif config.Wd_mode == "shared_across_slots_fisher":
            for bucket in (1024, 4096, 11008):
                self.fisher_trackers[f"shared_{bucket}"] = FisherSubspaceTracker(
                    bucket, config.r_fisher, mode="fisher_oja_per_slot"
                )
        else:
            for layer_idx in range(num_hidden_layers):
                for t_idx, target in enumerate(DCAG_TARGET_MODULES):
                    d_in = (
                        _LLAMA_7B_INTERMEDIATE
                        if target == "down_proj"
                        else _LLAMA_7B_HIDDEN
                    )
                    slot_id = _llm_slot_id(layer_idx, t_idx)
                    self.fisher_trackers[str(slot_id)] = FisherSubspaceTracker(
                        d_in, config.r_fisher, mode=config.Wd_mode
                    )

            self.fisher_trackers[str(_projector_slot_id(0))] = FisherSubspaceTracker(
                1024, config.r_fisher, mode=config.Wd_mode
            )
            self.fisher_trackers[str(_projector_slot_id(1))] = FisherSubspaceTracker(
                4096, config.r_fisher, mode=config.Wd_mode
            )

        for slot_id in range(self.num_total_slots):
            d_in = self._slot_d_in(slot_id)
            name = f"P_slot_{slot_id}"
            self.register_buffer(name, torch.zeros(d_in, 0))

        self.preservation_snapshots_U: Dict[str, Dict[int, torch.Tensor]] = {}
        self.preservation_snapshots_V: Dict[str, Dict[int, torch.Tensor]] = {}

        for bucket in DCAGGenerator.D_IN_BUCKETS:
            n_slots = len(self.generator._bucket_slot_ids()[bucket])
            for domain_id in range(len(DCAG_DOMAIN_TO_ID)):
                name_U = f"snap_U_{bucket}_{domain_id}"
                name_V = f"snap_V_{bucket}_{domain_id}"
                self.register_buffer(
                    name_U,
                    torch.zeros(
                        config.num_centroids,
                        n_slots,
                        config.r,
                        config.factored_head_dim,
                    ),
                )
                self.register_buffer(
                    name_V,
                    torch.zeros(
                        config.num_centroids, n_slots, config.factored_head_dim, bucket
                    ),
                )

        self.register_buffer("snapshot_mask", torch.zeros(len(DCAG_DOMAIN_TO_ID)))

        sp_size = sum(
            p.numel() for p in self.feature_encoder.shared_projection.parameters()
        )
        self._shared_projection_param_count = sp_size
        for domain_id in range(len(DCAG_DOMAIN_TO_ID)):
            self.register_buffer(
                f"shared_proj_snapshot_{domain_id}", torch.zeros(sp_size)
            )

        self._shared_proj_live_backup: Optional[torch.Tensor] = None

        for domain_id in range(len(DCAG_DOMAIN_TO_ID)):
            self.register_buffer(
                f"snap_trunk_{domain_id}",
                torch.zeros(
                    config.num_centroids,
                    self.num_total_slots,
                    config.generator_hidden_dim,
                ),
            )

        slot_weight = self._build_slot_weight()
        self.register_buffer("preservation_slot_weight", slot_weight)

        self.current_features: Optional[torch.Tensor] = None
        self.current_clean_features: Optional[torch.Tensor] = None
        self.current_domain_ids: Optional[torch.Tensor] = None
        self._generator_cache: Optional[Dict[str, Any]] = None

        self._beff_cache: Dict[int, torch.Tensor] = {}

        self._a_proj_cache: Dict[int, torch.Tensor] = {}

        self._layer_local_cache: Dict[int, Dict[str, Any]] = {}

        self._projector_local_cache: Optional[Dict[str, Any]] = None

        self._old_domain_gen_cache: Dict[int, Dict[str, Any]] = {}

        self._a_l2_value: Optional[torch.Tensor] = None
        self.latest_preservation_loss: Optional[torch.Tensor] = None
        self.latest_new_slice_suppression_loss: Optional[torch.Tensor] = None
        self.latest_functional_preservation_loss: Optional[torch.Tensor] = None
        self.latest_functional_preservation_llm_loss: Optional[torch.Tensor] = None
        self.latest_functional_preservation_projector_loss: Optional[torch.Tensor] = (
            None
        )

        self.latest_trunk_loss: Optional[torch.Tensor] = None
        self.latest_uv_loss: Optional[torch.Tensor] = None
        self.batch_counter: int = 0
        self.global_step: int = 0

        self.current_task_idx: Optional[int] = None

        self.current_domain_id: Optional[int] = None

        self._b_prefix_log_seen: set = set()

    @torch.no_grad()
    def _flatten_shared_projection(self) -> torch.Tensor:
        """Flatten shared-projection parameters into one vector."""
        return torch.cat(
            [
                p.detach().flatten()
                for p in self.feature_encoder.shared_projection.parameters()
            ]
        )

    @torch.no_grad()
    def _unflatten_into_shared_projection(self, flat: torch.Tensor) -> None:
        """Restore shared-projection parameters from a flat vector."""
        offset = 0
        for p in self.feature_encoder.shared_projection.parameters():
            n = p.numel()
            p.data.copy_(flat[offset : offset + n].view_as(p).to(p.device, p.dtype))
            offset += n

    def _swap_shared_projection_for_domain(self, domain_id: int) -> bool:
        """Load a domain projection snapshot for evaluation while retaining the live state.

        Return True if the snapshot was loaded.
        """
        if not getattr(self.config, "shared_projection_snapshot_enabled", True):
            return False
        snap_name = f"shared_proj_snapshot_{domain_id}"
        if not hasattr(self, snap_name):
            return False

        if float(self.snapshot_mask[domain_id].item()) <= 0:
            return False
        snap = getattr(self, snap_name)
        if self._shared_proj_live_backup is None:
            self._shared_proj_live_backup = self._flatten_shared_projection()
        self._unflatten_into_shared_projection(snap)
        return True

    def _restore_shared_projection(self) -> None:
        """Restore the live projection after domain-specific evaluation."""
        if self._shared_proj_live_backup is None:
            return
        self._unflatten_into_shared_projection(self._shared_proj_live_backup)
        self._shared_proj_live_backup = None

    def _slot_d_in(self, slot_id: int) -> int:
        if slot_id >= NUM_LLM_SLOTS:
            projector_idx = slot_id - NUM_LLM_SLOTS
            return 1024 if projector_idx == 0 else 4096
        t_idx = slot_id % len(DCAG_TARGET_MODULES)
        target = DCAG_TARGET_MODULES[t_idx]
        return _LLAMA_7B_INTERMEDIATE if target == "down_proj" else _LLAMA_7B_HIDDEN

    def _slot_d_out(self, slot_id: int) -> int:
        if slot_id >= NUM_LLM_SLOTS:
            return _LLAMA_7B_HIDDEN
        t_idx = slot_id % len(DCAG_TARGET_MODULES)
        target = DCAG_TARGET_MODULES[t_idx]

        return (
            _LLAMA_7B_INTERMEDIATE
            if target in ("gate_proj", "up_proj")
            else _LLAMA_7B_HIDDEN
        )

    def _slot_kind(self, slot_id: int) -> str:
        return "projector" if slot_id >= NUM_LLM_SLOTS else "llm"

    def _build_slot_weight(self) -> torch.Tensor:
        w = torch.ones(self.num_total_slots)
        mode = self.config.slot_weight_mode
        if mode == "projector_multiplier":
            w[PROJECTOR_SLOT_OFFSET:] = self.config.projector_preservation_multiplier
        elif mode == "uniform":
            pass
        elif mode == "layerwise_schedule":
            alpha = 1.0
            for layer_idx in range(self.num_hidden_layers):
                layer_weight = 1.0 + alpha * (
                    layer_idx / max(self.num_hidden_layers - 1, 1)
                )
                for t_idx in range(len(DCAG_TARGET_MODULES)):
                    w[_llm_slot_id(layer_idx, t_idx)] = layer_weight
            w[PROJECTOR_SLOT_OFFSET:] = self.config.projector_preservation_multiplier
        else:
            raise ValueError(f"Unknown slot_weight_mode {mode!r}")
        return w

    def _preservation_enabled(self) -> bool:
        per_domain = getattr(self.config, "anti_forgetting_weight_per_domain", None)
        if per_domain is not None:
            return any(float(x) > 0.0 for x in per_domain)
        return self.anti_forgetting_weight > 0.0

    def _get_old_domain_generator_output(
        self, domain_id: int, proto_features: torch.Tensor
    ) -> Dict[str, Any]:
        """Reuse generated factors for completed-domain prototypes within a batch."""
        domain_id = int(domain_id)
        cached = self._old_domain_gen_cache.get(domain_id)
        if cached is not None:
            return cached
        was_gen_training = self.generator.training
        self.generator.eval()
        try:
            gen_out = self.generator.forward_all_slots(
                proto_features, domain_id=domain_id
            )
        finally:
            self.generator.train(was_gen_training)
        self._old_domain_gen_cache[domain_id] = gen_out
        return gen_out

    def has_batch_context(self) -> bool:
        return self.current_features is not None

    def _regularize_features_for_generator(
        self, features: torch.Tensor
    ) -> torch.Tensor:
        """Apply training feature dropout and noise.

        With feature_regularize_generator_only, stored prototypes use clean features.
        """
        if not self.training:
            return features

        token_dropout = float(getattr(self.config, "feature_token_dropout", 0.0) or 0.0)
        noise_std = float(getattr(self.config, "feature_noise_std", 0.0) or 0.0)
        if token_dropout <= 0.0 and noise_std <= 0.0:
            return features

        regularized = features
        if token_dropout > 0.0:
            keep_prob = 1.0 - token_dropout
            if regularized.dim() == 3 and regularized.shape[1] > 1:
                layer_tokens = regularized[:, :-1, :]
                summary_token = regularized[:, -1:, :]
                mask = (
                    torch.rand(
                        layer_tokens.shape[0],
                        layer_tokens.shape[1],
                        1,
                        device=layer_tokens.device,
                        dtype=layer_tokens.dtype,
                    )
                    < keep_prob
                ).to(layer_tokens.dtype)
                layer_tokens = layer_tokens * mask / keep_prob
                regularized = torch.cat([layer_tokens, summary_token], dim=1)
            else:
                regularized = F.dropout(regularized, p=token_dropout, training=True)

        if noise_std > 0.0:
            regularized = regularized + torch.randn_like(regularized) * noise_std
        return regularized

    def _generator_input(self, features: torch.Tensor) -> torch.Tensor:
        if self.current_instruction_token is None:
            return features
        if features.dim() == 2:
            features = features.unsqueeze(1)
        tok = self.current_instruction_token
        if tok.shape[0] != features.shape[0]:
            return features
        return torch.cat(
            [features, tok.to(device=features.device, dtype=features.dtype)], dim=1
        )

    def set_batch_context(
        self,
        images,
        domain_ids: torch.Tensor,
        instruction_features: Optional[torch.Tensor] = None,
    ):

        self._beff_cache = {}
        self._a_proj_cache = {}
        self._layer_local_cache = {}
        self._projector_local_cache = None
        self._old_domain_gen_cache = {}
        self._a_l2_value = None
        self.latest_new_slice_suppression_loss = None
        self.latest_functional_preservation_loss = None
        self.latest_functional_preservation_llm_loss = None
        self.latest_functional_preservation_projector_loss = None

        self._restore_shared_projection()

        if not self.training and domain_ids.numel() > 0:
            unique = torch.unique(domain_ids)
            if unique.numel() == 1:
                d_for_swap = int(unique.item())
                self._swap_shared_projection_for_domain(d_for_swap)

        clean_features = self.feature_encoder(images, domain_ids)

        if (
            self.training
            and float(getattr(self.config, "alignment_coeff", 0.0) or 0.0) > 0.0
            and isinstance(clean_features, torch.Tensor)
            and clean_features.dim() == 3
        ):
            self._alignment_student_pooled = clean_features.mean(dim=1)
        else:
            self._alignment_student_pooled = None

        features = self._regularize_features_for_generator(clean_features)

        if (
            self.instruction_conditioning
            and instruction_features is not None
            and self.instruction_proj is not None
        ):
            p = next(self.instruction_proj.parameters())
            tok_in = instruction_features.to(device=p.device, dtype=p.dtype)
            if tok_in.dim() == 1:
                tok_in = tok_in.unsqueeze(0)
            self.current_instruction_token = (
                self.instruction_gate * self.instruction_proj(tok_in)
            ).unsqueeze(1)
        else:
            self.current_instruction_token = None
        gen_features = self._generator_input(features)
        self.current_clean_features = clean_features
        self.current_features = features
        self.current_domain_ids = domain_ids
        if domain_ids.numel() > 0:
            unique = torch.unique(domain_ids)
            if unique.numel() == 1:
                self.current_domain_id = int(unique.item())
            else:
                self.current_domain_id = None

        layer_recompute = self.config.generator_cache_mode == "layer_recompute"

        if not layer_recompute:
            self._generator_cache = self.generator.forward_all_slots(
                gen_features, domain_id=self.current_domain_id
            )
        elif not self.training:
            self._generator_cache = self.generator.forward_all_slots(
                gen_features, domain_id=self.current_domain_id
            )
        else:
            self._generator_cache = None
            if float(self.config.a_l2_penalty or 0.0) > 0.0:
                gen_full = self.generator.forward_all_slots(
                    gen_features, domain_id=self.current_domain_id
                )
                self._a_l2_value = self._compute_a_l2_from_gen_out(gen_full)

                del gen_full

        self.batch_counter += 1

        if self.training:
            self.global_step += 1

        if not self.training:
            self._populate_eval_caches()

        if (
            self.training
            and self._preservation_enabled()
            and float(self.snapshot_mask.sum().item()) > 0
        ):
            self.latest_preservation_loss = self._compute_preservation_loss()
        else:
            self.latest_preservation_loss = None
            self.latest_trunk_loss = None
            self.latest_uv_loss = None
        if self.training and float(self.snapshot_mask.sum().item()) > 0:
            self.latest_new_slice_suppression_loss = (
                self.compute_new_slice_suppression_loss()
            )
        else:
            self.latest_new_slice_suppression_loss = None
        if self.training and float(self.snapshot_mask.sum().item()) > 0:
            self.latest_functional_preservation_loss = (
                self.compute_functional_preservation_loss()
            )
        else:
            self.latest_functional_preservation_loss = None
            self.latest_functional_preservation_llm_loss = None
            self.latest_functional_preservation_projector_loss = None

        self._old_domain_gen_cache = {}

    def clear_batch_context(self):

        self._restore_shared_projection()
        self.current_features = None
        self.current_clean_features = None
        self.current_domain_ids = None
        self.current_instruction_token = None
        self._generator_cache = None
        self._beff_cache = {}
        self._a_proj_cache = {}
        self._layer_local_cache = {}
        self._projector_local_cache = None
        self._old_domain_gen_cache = {}
        self._a_l2_value = None
        self.latest_preservation_loss = None
        self.latest_new_slice_suppression_loss = None
        self.latest_functional_preservation_loss = None
        self.latest_functional_preservation_llm_loss = None
        self.latest_functional_preservation_projector_loss = None
        self.latest_trunk_loss = None
        self.latest_uv_loss = None

    @contextmanager
    def layer_context(self, layer_idx: int):
        """Cache factors for one decoder layer during training; clear them on exit.

        Evaluation uses the batch-level factor cache.
        """
        if (
            self.config.generator_cache_mode != "layer_recompute"
            or not self.training
            or self.current_features is None
        ):
            yield
            return
        gen_out = self.generator.forward_for_layer(
            layer_idx,
            self._generator_input(self.current_features),
            domain_id=self.current_domain_id,
        )
        self._layer_local_cache[layer_idx] = gen_out
        try:
            yield
        finally:
            self._layer_local_cache.pop(layer_idx, None)

    @contextmanager
    def projector_context(self):
        """Cache factors for the two projector slots during training; clear them on exit."""
        if (
            self.config.generator_cache_mode != "layer_recompute"
            or not self.training
            or self.current_features is None
        ):
            yield
            return
        self._projector_local_cache = self.generator.forward_for_projector(
            self._generator_input(self.current_features),
            domain_id=self.current_domain_id,
        )
        try:
            yield
        finally:
            self._projector_local_cache = None

    def finalize_after_backward(self):
        """Clear batch state after backward recomputation has completed."""
        self.current_features = None
        self.current_clean_features = None
        self.current_domain_ids = None
        self.current_instruction_token = None
        self._layer_local_cache = {}
        self._projector_local_cache = None

    @torch.no_grad()
    def _populate_eval_caches(self):
        """Cache projected low-rank input factors once per evaluation batch."""
        if self._generator_cache is None:
            return
        U_all = self._generator_cache["U"]
        V_dict = self._generator_cache["V"]
        bucket_slot_ids = self._generator_cache["bucket_slot_ids"]
        for bucket_int, slot_list in bucket_slot_ids.items():
            bucket_str = str(bucket_int)
            if bucket_str not in V_dict:
                continue
            V_bucket = V_dict[bucket_str]
            for local_idx, slot_id in enumerate(slot_list):
                U = U_all[:, slot_id, :, :]
                V = V_bucket[:, local_idx, :, :]
                A_tilde = torch.einsum("brh,bhd->brd", U, V)

                P = self.get_P_for_domain(slot_id, self.current_domain_id)
                if P is not None and P.shape[1] > 0:
                    P_local = P.to(device=A_tilde.device, dtype=A_tilde.dtype)
                    A = A_tilde - (A_tilde @ P_local) @ P_local.t()
                else:
                    A = A_tilde
                self._a_proj_cache[slot_id] = A

    def _compute_a_l2_from_gen_out(
        self, gen_out: Dict[str, Any]
    ) -> Optional[torch.Tensor]:
        """Compute factor regularization from all-slot generator outputs."""
        U_all = gen_out.get("U")
        V_dict = gen_out.get("V")
        bucket_slot_ids = gen_out.get("bucket_slot_ids")
        if U_all is None or V_dict is None or bucket_slot_ids is None:
            return None
        total = None
        n_slots = 0
        for bucket_int, slot_list in bucket_slot_ids.items():
            bucket_str = str(bucket_int)
            if bucket_str not in V_dict:
                continue
            V_bucket = V_dict[bucket_str]
            d_in_slot = V_bucket.shape[-1]
            r_slot = U_all.shape[-2]
            for local_idx, slot_id in enumerate(slot_list):
                U = U_all[:, slot_id, :, :]
                V = V_bucket[:, local_idx, :, :]
                UTU = torch.einsum("brh,brk->bhk", U, U)
                VVT = torch.einsum("bhd,bkd->bhk", V, V)
                fro2 = (UTU * VVT).sum(dim=(-2, -1))
                slot_term = fro2.mean() / float(r_slot * d_in_slot)
                total = slot_term if total is None else total + slot_term
                n_slots += 1
        if total is None or n_slots == 0:
            return None
        return total / float(n_slots)

    def _current_B_gram(
        self, slot_id: int, device: torch.device
    ) -> Optional[torch.Tensor]:
        B_cur = self.task_slice_manager.B_current_slice(slot_id)
        if B_cur is None:
            return None
        return B_cur.to(device=device, dtype=torch.float32).t() @ B_cur.to(
            device=device, dtype=torch.float32
        )

    def _effective_B_for_domain_snapshot(
        self, slot_id: int, domain_id: int, device: torch.device
    ) -> torch.Tensor:
        if self._rolling_function_active_for_current_task():
            return self._effective_B_for_current_task_start(slot_id, device)
        return self.task_slice_manager.B_effective_for_domain(slot_id, domain_id).to(
            device=device, dtype=torch.float32
        )

    def _effective_B_for_current_task_start(
        self, slot_id: int, device: torch.device
    ) -> torch.Tensor:
        if self.config.B_mode == "plain_B":
            return self.task_slice_manager.B_effective(slot_id, self.batch_counter).to(
                device=device, dtype=torch.float32
            )
        if self.current_task_idx is None:
            _warn_once(
                "rolling_B_missing_current_task",
                "DCAG rolling functional preservation requested before "
                "current_task_idx was set; falling back to full B.",
            )
            return self._effective_B_full_current(slot_id, device)
        kind = self.task_slice_manager.kind_per_slot[slot_id]
        task_map = self.task_slice_manager.task_slice_map.get(
            int(self.current_task_idx), {}
        )
        if kind not in task_map:
            _warn_once(
                "rolling_B_missing_task_slice",
                "DCAG rolling functional preservation could not find the "
                "current task slice boundary; falling back to full B.",
            )
            return self._effective_B_full_current(slot_id, device)
        start_col, _ = task_map[kind]
        return self.task_slice_manager._B_effective_prefix(slot_id, start_col).to(
            device=device, dtype=torch.float32
        )

    def _effective_B_full_current(
        self, slot_id: int, device: torch.device
    ) -> torch.Tensor:
        return self.task_slice_manager.B_effective(slot_id, self.batch_counter).to(
            device=device, dtype=torch.float32
        )

    def _project_A_for_domain(
        self, slot_id: int, domain_id: int, A_tilde: torch.Tensor
    ) -> torch.Tensor:
        P = self.get_P_for_domain(slot_id, domain_id)
        if P is None or P.shape[1] == 0:
            return A_tilde
        P_local = P.to(device=A_tilde.device, dtype=A_tilde.dtype)
        return A_tilde - (A_tilde @ P_local) @ P_local.t()

    def _projected_A_gram_from_factors(
        self,
        slot_id: int,
        domain_id: int,
        U: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (U V Q) (U V Q)^T without materializing A=UV.

        U: (K, r, h), V: (K, h, d), Q = I - P_<domain>P_<domain>^T.
        Returns: (K, r, r).
        """
        U = U.to(dtype=torch.float32)
        V = V.to(dtype=torch.float32)
        vqvt = torch.einsum("khd,kjd->khj", V, V)
        P = self.get_P_for_domain(slot_id, domain_id)
        if P is not None and P.shape[1] > 0:
            P_local = P.to(device=V.device, dtype=V.dtype)
            vp = torch.einsum("khd,dp->khp", V, P_local)
            vqvt = vqvt - torch.einsum("khp,kjp->khj", vp, vp)
        return torch.einsum("krh,khj,ksj->krs", U, vqvt, U)

    def _projected_A_cross_gram_from_factors(
        self,
        slot_id: int,
        domain_id: int,
        U_left: torch.Tensor,
        V_left: torch.Tensor,
        U_right: torch.Tensor,
        V_right: torch.Tensor,
    ) -> torch.Tensor:
        """Compute (U_l V_l Q) (U_r V_r Q)^T without materializing A."""
        U_left = U_left.to(dtype=torch.float32)
        V_left = V_left.to(dtype=torch.float32)
        U_right = U_right.to(dtype=torch.float32)
        V_right = V_right.to(dtype=torch.float32)
        vqvt = torch.einsum("khd,kjd->khj", V_left, V_right)
        P = self.get_P_for_domain(slot_id, domain_id)
        if P is not None and P.shape[1] > 0:
            P_local = P.to(device=V_left.device, dtype=V_left.dtype)
            left_p = torch.einsum("khd,dp->khp", V_left, P_local)
            right_p = torch.einsum("khd,dp->khp", V_right, P_local)
            vqvt = vqvt - torch.einsum("khp,kjp->khj", left_p, right_p)
        return torch.einsum("krh,khj,ksj->krs", U_left, vqvt, U_right)

    @staticmethod
    def _snapshot_A_gram(snap_U: torch.Tensor, snap_V: torch.Tensor) -> torch.Tensor:
        """Compute snapshot A @ A.T from U and V.

        Supports individual slots (K, r, h) or slot groups (K, n_slots, r, h).
        """
        snap_U = snap_U.to(dtype=torch.float32)
        snap_V = snap_V.to(dtype=torch.float32)
        if snap_U.dim() == 3:
            vvt = torch.einsum("khd,kjd->khj", snap_V, snap_V)
            return torch.einsum("krh,khj,ksj->krs", snap_U, vvt, snap_U)
        vvt = torch.einsum("knhd,knjd->knhj", snap_V, snap_V)
        return torch.einsum("knrh,knhj,knsj->knrs", snap_U, vvt, snap_U)

    def compute_new_slice_suppression_loss(self) -> Optional[torch.Tensor]:
        """Penalize ||B_current @ A_snapshot||_F^2 on completed-domain prototypes.

        Gradients update only the active output slices.
        """
        if not self.training:
            return None
        coef = float(getattr(self.config, "new_slice_suppression_coeff", 0.0) or 0.0)
        if coef <= 0.0:
            return None
        if self.current_domain_id is None:
            return None
        if self.config.B_mode != "column_partitioned_CR":
            return None

        completed = [
            d
            for d in range(len(DCAG_DOMAIN_TO_ID))
            if float(self.snapshot_mask[d].item()) > 0 and d != self.current_domain_id
        ]
        if not completed:
            return None

        gen_param = next(self.generator.parameters())
        device = gen_param.device
        total = torch.zeros((), device=device, dtype=torch.float32)
        n_terms = 0
        bucket_slot_ids = self.generator._bucket_slot_ids()
        slot_weight_all = self.preservation_slot_weight.to(
            device=device, dtype=torch.float32
        )
        b_gram_cache: Dict[int, Optional[torch.Tensor]] = {}

        for domain_id in completed:
            counts = self.centroid_manager.prototype_counts[domain_id]
            k_d = int(self.centroid_manager.num_centroids_per_domain[domain_id])
            valid_k = (counts[:k_d] > 0).nonzero(as_tuple=True)[0]
            if valid_k.numel() == 0:
                continue
            for bucket in DCAGGenerator.D_IN_BUCKETS:
                slot_ids = bucket_slot_ids[bucket]
                if not slot_ids:
                    continue
                snap_U_buf = getattr(self, f"snap_U_{bucket}_{domain_id}")
                snap_V_buf = getattr(self, f"snap_V_{bucket}_{domain_id}")
                for local_idx, slot_id in enumerate(slot_ids):
                    if slot_id not in b_gram_cache:
                        b_gram_cache[slot_id] = self._current_B_gram(slot_id, device)
                    b_gram = b_gram_cache[slot_id]
                    if b_gram is None:
                        continue

                    snap_U_slot = snap_U_buf[valid_k, local_idx, :, :].to(device=device)
                    snap_V_slot = snap_V_buf[valid_k, local_idx, :, :].to(device=device)
                    slot_a_gram = self._snapshot_A_gram(snap_U_slot, snap_V_slot)

                    denom = float(max(1, self._slot_d_out(slot_id) * bucket))
                    slot_term = (slot_a_gram * b_gram.unsqueeze(0)).sum(
                        dim=(-2, -1)
                    ).mean() / denom
                    total = total + slot_term * slot_weight_all[slot_id]
                    n_terms += 1

        if n_terms == 0:
            return None
        return total / float(n_terms)

    def compute_functional_preservation_loss(self) -> Optional[torch.Tensor]:
        if not self.training:
            return None
        coef = float(getattr(self.config, "functional_preservation_coeff", 0.0) or 0.0)
        if coef <= 0.0:
            return None
        if self.current_domain_id is None:
            return None

        completed = [
            d
            for d in range(len(DCAG_DOMAIN_TO_ID))
            if float(self.snapshot_mask[d].item()) > 0 and d != self.current_domain_id
        ]
        if not completed:
            return None

        gen_param = next(self.generator.parameters())
        device = gen_param.device
        dtype = gen_param.dtype
        bucket_slot_ids = self.generator._bucket_slot_ids()

        total = torch.zeros((), device=device, dtype=torch.float32)
        llm_total = torch.zeros((), device=device, dtype=torch.float32)
        proj_total = torch.zeros((), device=device, dtype=torch.float32)
        n_terms = 0
        llm_terms = 0
        proj_terms = 0
        slot_weight_all = self.preservation_slot_weight.to(
            device=device, dtype=torch.float32
        )
        b_full_cache: Dict[int, torch.Tensor] = {}
        b_snap_cache: Dict[Tuple[int, int], torch.Tensor] = {}

        was_gen_training = self.generator.training
        self.generator.eval()
        try:
            for domain_id in completed:
                counts = self.centroid_manager.prototype_counts[domain_id]
                k_d = int(self.centroid_manager.num_centroids_per_domain[domain_id])
                valid_k = (counts[:k_d] > 0).nonzero(as_tuple=True)[0]
                if valid_k.numel() == 0:
                    continue
                proto = self.centroid_manager.prototype_features[domain_id][valid_k].to(
                    device=device, dtype=dtype
                )
                gen_out_curr = self._get_old_domain_generator_output(domain_id, proto)
                curr_U_full = gen_out_curr["U"]
                curr_V_dict = gen_out_curr["V"]
                for bucket in DCAGGenerator.D_IN_BUCKETS:
                    slot_ids = bucket_slot_ids[bucket]
                    if not slot_ids:
                        continue
                    curr_V_bucket = curr_V_dict[str(bucket)]
                    snap_U_buf = getattr(self, f"snap_U_{bucket}_{domain_id}")
                    snap_V_buf = getattr(self, f"snap_V_{bucket}_{domain_id}")
                    for local_idx, slot_id in enumerate(slot_ids):
                        if slot_id not in b_full_cache:
                            b_full_cache[slot_id] = self._effective_B_full_current(
                                slot_id, device
                            )
                        snap_key = (domain_id, slot_id)
                        if snap_key not in b_snap_cache:
                            b_snap_cache[snap_key] = (
                                self._effective_B_for_domain_snapshot(
                                    slot_id, domain_id, device
                                )
                            )
                        B_full = b_full_cache[slot_id]
                        B_snap = b_snap_cache[snap_key]

                        U_cur = curr_U_full[:, slot_id, :, :]
                        V_cur = curr_V_bucket[:, local_idx, :, :]
                        snap_U = snap_U_buf[valid_k, local_idx, :, :].to(device=device)
                        snap_V = snap_V_buf[valid_k, local_idx, :, :].to(device=device)

                        B_full_gram = B_full.t() @ B_full
                        B_snap_gram = B_snap.t() @ B_snap
                        B_cross = B_full.t() @ B_snap
                        A_cur_gram = self._projected_A_gram_from_factors(
                            slot_id, domain_id, U_cur, V_cur
                        )
                        A_snap_gram = self._projected_A_gram_from_factors(
                            slot_id, domain_id, snap_U, snap_V
                        )
                        A_cross = self._projected_A_cross_gram_from_factors(
                            slot_id, domain_id, U_cur, V_cur, snap_U, snap_V
                        )

                        term_cur = (A_cur_gram * B_full_gram.unsqueeze(0)).sum(
                            dim=(-2, -1)
                        )
                        term_snap = (A_snap_gram * B_snap_gram.unsqueeze(0)).sum(
                            dim=(-2, -1)
                        )
                        term_cross = (A_cross * B_cross.unsqueeze(0)).sum(dim=(-2, -1))
                        per_proto = torch.clamp(
                            term_cur + term_snap - 2.0 * term_cross, min=0.0
                        )
                        denom = float(max(1, self.config.r * bucket))
                        slot_term = per_proto.mean() / denom
                        slot_term = slot_term * slot_weight_all[slot_id]

                        total = total + slot_term
                        n_terms += 1
                        if slot_id < NUM_LLM_SLOTS:
                            llm_total = llm_total + slot_term
                            llm_terms += 1
                        else:
                            proj_total = proj_total + slot_term
                            proj_terms += 1
        finally:
            self.generator.train(was_gen_training)

        if n_terms == 0:
            self.latest_functional_preservation_llm_loss = None
            self.latest_functional_preservation_projector_loss = None
            return None
        llm_loss = (
            llm_total / float(llm_terms)
            if llm_terms
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        proj_loss = (
            proj_total / float(proj_terms)
            if proj_terms
            else torch.zeros((), device=device, dtype=torch.float32)
        )
        self.latest_functional_preservation_llm_loss = llm_loss.detach()
        self.latest_functional_preservation_projector_loss = proj_loss.detach()
        return (total / float(n_terms)).to(dtype=dtype)

    def compute_spectral_balance_penalty(self) -> Optional[torch.Tensor]:
        if not self.training:
            return None
        coef = float(getattr(self.config, "spectral_balance_coeff", 0.0) or 0.0)
        if coef <= 0.0:
            return None
        every_n = max(
            1, int(getattr(self.config, "spectral_balance_every_n_steps", 50))
        )
        if (self.global_step % every_n) != 0:
            return None

        tsm = self.task_slice_manager

        device = next(self.generator.parameters()).device
        dtype = next(self.generator.parameters()).dtype
        per_slot_var = []
        for slot_id in range(self.num_total_slots):
            try:
                C_cur = getattr(tsm, f"C_slot_{slot_id}_current", None)
                R_cur = getattr(tsm, f"R_slot_{slot_id}_current", None)
            except Exception:
                continue
            if C_cur is None or R_cur is None:
                continue
            if not C_cur.requires_grad or not R_cur.requires_grad:
                continue

            h_d_kind = R_cur.shape[1]
            if h_d_kind < 2:
                continue
            B_cur = (
                C_cur.to(dtype=torch.float32, device=device)
                @ R_cur.to(dtype=torch.float32, device=device).t()
            )
            if B_cur.numel() == 0:
                continue
            try:
                svs = torch.linalg.svdvals(B_cur)
            except Exception:
                continue
            svs_top = svs[:h_d_kind]
            if svs_top.numel() < 2:
                continue

            total_e = svs_top.sum() + 1e-8
            svs_norm = svs_top / total_e
            target = 1.0 / float(h_d_kind)
            per_slot_var.append(((svs_norm - target) ** 2).sum())
        if not per_slot_var:
            return None
        total = torch.stack(per_slot_var).mean().to(dtype=dtype)
        return coef * total

    def compute_a_l2_penalty(self) -> Optional[torch.Tensor]:
        """Compute mean normalized ||U @ V||_F^2 across slots using factor Gram matrices.

        Returns None outside training or when generator outputs are unavailable.
        """
        if not self.training:
            return None
        if self.config.generator_cache_mode == "layer_recompute":
            return self._a_l2_value
        if self._generator_cache is None:
            return None
        return self._compute_a_l2_from_gen_out(self._generator_cache)

    def compute_alignment_loss(
        self, teacher: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Compute cosine alignment loss against detached CLIP features during training.

        Return None when disabled or when either representation is unavailable.
        """
        if not self.training:
            return None
        if float(getattr(self.config, "alignment_coeff", 0.0) or 0.0) <= 0.0:
            return None
        student_pooled = getattr(self, "_alignment_student_pooled", None)
        if student_pooled is None or teacher is None:
            return None
        if student_pooled.shape[0] != teacher.shape[0]:
            return None
        student = self.alignment_head(student_pooled)
        teacher = teacher.to(device=student.device, dtype=student.dtype)
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean()

    def finalize_batch(self):
        if not self.training:
            return
        if self.current_features is not None and self.current_domain_ids is not None:
            if (
                getattr(self.config, "feature_regularize_generator_only", True)
                and self.current_clean_features is not None
            ):
                self.centroid_manager.update(
                    self.current_clean_features, self.current_domain_ids
                )
            else:
                self.centroid_manager.update(
                    self.current_features, self.current_domain_ids
                )
        self.current_clean_features = None

        self._generator_cache = None
        self._beff_cache = {}
        self._old_domain_gen_cache = {}
        self.latest_preservation_loss = None
        self.latest_new_slice_suppression_loss = None
        self.latest_functional_preservation_loss = None
        self.latest_functional_preservation_llm_loss = None
        self.latest_functional_preservation_projector_loss = None
        self.latest_trunk_loss = None
        self.latest_uv_loss = None
        self._a_l2_value = None
        if self.config.generator_cache_mode != "layer_recompute":
            self.current_features = None
            self.current_domain_ids = None

    def get_slot_UV(
        self, slot_id: int, d_in_bucket: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        if (
            self.config.generator_cache_mode == "layer_recompute"
            and self._generator_cache is None
        ):
            if slot_id < NUM_LLM_SLOTS:
                layer_idx = slot_id // len(DCAG_TARGET_MODULES)
                target_idx = slot_id % len(DCAG_TARGET_MODULES)
                cache = self._layer_local_cache.get(layer_idx)
                if cache is None:
                    return None, None
                U = cache["U_layer"][:, target_idx, :, :].to(device=device, dtype=dtype)

                if d_in_bucket == 11008:
                    targets = cache["V_11008_targets"]
                    V_subset = cache["V_11008"]
                elif d_in_bucket == 4096:
                    targets = cache["V_4096_targets"]
                    V_subset = cache["V_4096"]
                else:
                    return None, None
                try:
                    local_idx = targets.index(target_idx)
                except ValueError:
                    return None, None
                V = V_subset[:, local_idx, :, :].to(device=device, dtype=dtype)
                return U, V
            else:
                proj_local = slot_id - NUM_LLM_SLOTS
                cache = self._projector_local_cache
                if cache is None:
                    return None, None
                U = cache["U_proj"][:, proj_local, :, :].to(device=device, dtype=dtype)
                if d_in_bucket == 1024:
                    targets = cache["V_1024_local"]
                    V_subset = cache["V_1024"]
                elif d_in_bucket == 4096:
                    targets = cache["V_4096_local"]
                    V_subset = cache["V_4096"]
                else:
                    return None, None
                try:
                    local_idx = targets.index(proj_local)
                except ValueError:
                    return None, None
                V = V_subset[:, local_idx, :, :].to(device=device, dtype=dtype)
                return U, V

        if self._generator_cache is None:
            return None, None
        U_all = self._generator_cache["U"]
        U = U_all[:, slot_id, :, :].to(device=device, dtype=dtype)
        bucket_slot_ids = self._generator_cache["bucket_slot_ids"]
        try:
            local_idx = bucket_slot_ids[d_in_bucket].index(slot_id)
        except ValueError:
            return None, None
        V_bucket = self._generator_cache["V"][str(d_in_bucket)]
        V = V_bucket[:, local_idx, :, :].to(device=device, dtype=dtype)
        return U, V

    def get_slot_gamma(
        self, slot_id: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:

        if (
            self.config.generator_cache_mode == "layer_recompute"
            and self._generator_cache is None
        ):
            if slot_id < NUM_LLM_SLOTS:
                layer_idx = slot_id // len(DCAG_TARGET_MODULES)
                target_idx = slot_id % len(DCAG_TARGET_MODULES)
                cache = self._layer_local_cache.get(layer_idx)
                if cache is None or cache.get("gamma_layer") is None:
                    return None
                return cache["gamma_layer"][:, target_idx, :].to(
                    device=device, dtype=dtype
                )
            else:
                proj_local = slot_id - NUM_LLM_SLOTS
                cache = self._projector_local_cache
                if cache is None or cache.get("gamma_proj") is None:
                    return None
                return cache["gamma_proj"][:, proj_local, :].to(
                    device=device, dtype=dtype
                )

        if self._generator_cache is None:
            return None
        gamma = self._generator_cache["gamma"]
        if gamma is None:
            return None
        return gamma[:, slot_id, :].to(device=device, dtype=dtype)

    def get_B_effective(
        self, slot_id: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Cache the composed output factor per slot and batch."""
        if slot_id in self._beff_cache:
            return self._beff_cache[slot_id].to(device=device, dtype=dtype)
        if (
            (not self.training)
            and getattr(self.config, "b_composition_mode", "full") == "domain_prefix"
            and self.current_domain_id is not None
        ):
            self._log_b_prefix_boundary_once(slot_id)
            B_eff = self.task_slice_manager.B_effective_for_domain(
                slot_id, self.current_domain_id
            )
        else:
            B_eff = self.task_slice_manager.B_effective(slot_id, self.batch_counter)
        self._beff_cache[slot_id] = B_eff
        return B_eff.to(device=device, dtype=dtype)

    def _log_b_prefix_boundary_once(self, slot_id: int) -> None:
        kind = self.task_slice_manager.kind_per_slot.get(slot_id)
        if kind is None:
            return

        if kind == "llm" and slot_id != 0:
            return
        if kind == "projector" and slot_id != NUM_LLM_SLOTS:
            return
        domain_id = int(self.current_domain_id)
        key = (domain_id, kind)
        if key in self._b_prefix_log_seen:
            return
        tsm = self.task_slice_manager
        task_idx = tsm.domain_to_task.get(domain_id, domain_id)
        task_map = tsm.task_slice_map.get(int(task_idx), {})
        start_col, end_col = task_map.get(kind, (None, None))
        domain_name = DCAG_ID_TO_DOMAIN.get(domain_id, str(domain_id))
        print(
            "[DCAG][B] "
            f"composition=domain_prefix domain={domain_name} domain_id={domain_id} "
            f"task_idx={task_idx} kind={kind} cols={start_col}:{end_col} "
            f"domain_id_source={getattr(self.config, 'domain_id_source', 'oracle')}",
            flush=True,
        )
        self._b_prefix_log_seen.add(key)

    def current_B_column_range(self, kind: str) -> Optional[Tuple[int, int]]:

        if self.current_task_idx is None:
            return None
        return self.task_slice_manager.task_slice_map.get(
            self.current_task_idx, {}
        ).get(kind, None)

    def get_residual_AB(
        self, slot_id: int, device: torch.device, dtype: torch.dtype
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        mgr = self.residual_lora_manager
        if mgr is None:
            return None
        if self.training:
            A_res, B_res = mgr.current_AB(slot_id)
        else:
            if self.current_domain_id is None:
                return None
            tsm = self.task_slice_manager
            task_idx = tsm.domain_to_task.get(
                int(self.current_domain_id), int(self.current_domain_id)
            )
            pair = mgr.task_AB(slot_id, task_idx)
            if pair is None:
                return None
            A_res, B_res = pair
        return (
            A_res.to(device=device, dtype=dtype),
            B_res.to(device=device, dtype=dtype),
        )

    def get_P_cumulative(self, slot_id: int) -> Optional[torch.Tensor]:

        name = f"P_slot_{slot_id}"
        if name in self._buffers:
            return self._buffers[name]
        return None

    def get_P_for_domain(
        self, slot_id: int, domain_id: Optional[int]
    ) -> Optional[torch.Tensor]:
        if domain_id is None:
            return self.get_P_cumulative(slot_id)

        tsm = self.task_slice_manager
        task_idx = tsm.domain_to_task.get(int(domain_id), None)
        if task_idx is None:
            task_idx = int(domain_id)

        if task_idx <= 0:
            return None

        pieces: List[torch.Tensor] = []
        per_task_present = False
        for t in range(int(task_idx)):
            name = f"W_slot_{slot_id}_task_{t}"
            if name in self._buffers:
                per_task_present = True
                buf = self._buffers[name]
                if buf is not None and buf.numel() > 0:
                    pieces.append(buf)

        if per_task_present:
            if not pieces:
                return None
            if len(pieces) == 1:
                return pieces[0]
            return torch.cat(pieces, dim=1)

        _warn_once(
            "P_for_domain_legacy_fallback",
            "DCAG: per-task W_slot_*_task_* buffers not found; falling "
            "back to cumulative P_slot_{i}. Eval on past domains may project "
            "against later-task W's. Re-run task training with the current "
            "code to get per-domain P snapshots in non_lora_trainables.bin.",
        )
        return self.get_P_cumulative(slot_id)

    def get_fisher_tracker(
        self, slot_id: int, d_in_bucket: int
    ) -> Optional[FisherSubspaceTracker]:
        if self.config.Wd_mode == "none_disable_A_projection":
            return None

        if self.config.Wd_mode == "shared_across_slots_fisher":
            key = f"shared_{d_in_bucket}"
        else:
            key = str(slot_id)
        if key in self.fisher_trackers:
            return self.fisher_trackers[key]
        return None

    def _compute_preservation_loss(self) -> torch.Tensor:
        gen_param = next(self.generator.parameters())
        device = gen_param.device
        dtype = gen_param.dtype
        target = self.config.preservation_target

        completed = [
            d
            for d in range(len(DCAG_DOMAIN_TO_ID))
            if float(self.snapshot_mask[d].item()) > 0 and d != self.current_domain_id
        ]
        if not completed:
            self.latest_trunk_loss = None
            self.latest_uv_loss = None
            return torch.zeros((), device=device, dtype=dtype)

        afw_per_domain = getattr(self.config, "anti_forgetting_weight_per_domain", None)

        def _afw_for_domain(domain_id: int) -> float:
            if afw_per_domain is None:
                return 1.0
            return float(afw_per_domain[domain_id])

        bucket_slot_ids = self.generator._bucket_slot_ids()

        trunk_sum = torch.zeros((), device=device, dtype=dtype)
        trunk_count = 0
        uv_sum = torch.zeros((), device=device, dtype=dtype)
        uv_count = 0
        for d in completed:
            afw_d = _afw_for_domain(d)

            proto = self.centroid_manager.prototype_features[d]
            counts = self.centroid_manager.prototype_counts[d]
            valid_k = (counts > 0).nonzero(as_tuple=True)[0]
            if valid_k.numel() == 0:
                continue
            proto_valid = proto[valid_k].to(device=device, dtype=dtype)
            gen_out_curr = self._get_old_domain_generator_output(d, proto_valid)
            curr_U_full = gen_out_curr["U"]
            curr_V_dict = gen_out_curr["V"]

            if getattr(self.config, "hidden_state_consistency_enabled", True):
                trunk_buf = getattr(self, f"snap_trunk_{d}", None)
                trunk_curr = gen_out_curr.get("trunk")
                if trunk_buf is not None and trunk_curr is not None:
                    trunk_snap = trunk_buf[valid_k].to(device=device, dtype=dtype)

                    diff = (trunk_curr - trunk_snap).pow(2).mean(dim=-1)
                    weighted_trunk = diff * self.preservation_slot_weight.to(
                        device=device, dtype=dtype
                    ).unsqueeze(0)
                    trunk_sum = trunk_sum + afw_d * weighted_trunk.mean()
                    trunk_count += 1
            for bucket in DCAGGenerator.D_IN_BUCKETS:
                bucket_str = str(bucket)
                slot_ids_in_bucket = bucket_slot_ids[bucket]
                if not slot_ids_in_bucket:
                    continue

                snap_U = getattr(self, f"snap_U_{bucket}_{d}").to(
                    device=device, dtype=dtype
                )
                snap_V = getattr(self, f"snap_V_{bucket}_{d}").to(
                    device=device, dtype=dtype
                )
                curr_V = curr_V_dict[bucket_str]
                slot_idx_t = torch.tensor(
                    slot_ids_in_bucket, device=device, dtype=torch.long
                )
                curr_U = curr_U_full.index_select(dim=1, index=slot_idx_t)

                slot_weights = self.preservation_slot_weight[slot_idx_t].to(
                    device=device, dtype=dtype
                )

                snap_U_v = snap_U[valid_k]
                snap_V_v = snap_V[valid_k]
                if target == "UV_factor_mse":
                    u_diff = (curr_U - snap_U_v).pow(2).mean(dim=(-1, -2))
                    v_diff = (curr_V - snap_V_v).pow(2).mean(dim=(-1, -2))
                    per_slot = u_diff + v_diff
                    weighted = per_slot * slot_weights.unsqueeze(0)
                    uv_sum = uv_sum + afw_d * weighted.mean()
                    uv_count += 1
                elif target == "A_reconstruction_mse":
                    A_curr = torch.einsum("knrh,knhd->knrd", curr_U, curr_V)
                    A_snap = torch.einsum("knrh,knhd->knrd", snap_U_v, snap_V_v)
                    diff = (A_curr - A_snap).pow(2).mean(dim=(-1, -2))
                    weighted = diff * slot_weights.unsqueeze(0)
                    uv_sum = uv_sum + afw_d * weighted.mean()
                    uv_count += 1
                else:
                    raise ValueError(target)

        if trunk_count == 0 and uv_count == 0:
            self.latest_trunk_loss = None
            self.latest_uv_loss = None
            return torch.zeros((), device=device, dtype=dtype)
        if trunk_count > 0:
            trunk_term = trunk_sum / trunk_count
        else:
            trunk_term = torch.zeros((), device=device, dtype=dtype)
        if uv_count > 0:
            uv_term = uv_sum / uv_count
        else:
            uv_term = torch.zeros((), device=device, dtype=dtype)

        self.latest_trunk_loss = trunk_term.detach()
        self.latest_uv_loss = uv_term.detach()
        trunk_coeff = float(getattr(self.config, "preservation_trunk_coeff", 1.0))
        uv_coeff = float(getattr(self.config, "preservation_uv_coeff", 1.0))
        return trunk_coeff * trunk_term + uv_coeff * uv_term

    def start_task(self, task_idx: int):
        self.current_task_idx = task_idx
        self.task_slice_manager.start_task(task_idx)

        if self.residual_lora_manager is not None:
            self.residual_lora_manager.start_task(task_idx)

        for tracker in self.fisher_trackers.values():
            tracker.reset()

        self.centroid_manager.sync_completed_domains_only(None)

    def _rolling_start_task_idx(self) -> int:
        return int(getattr(self.config, "rolling_start_task_idx", 4))

    def _rolling_function_active_for_current_task(self) -> bool:
        if not getattr(self.config, "rolling_function_preservation", True):
            return False
        if self.current_task_idx is None:
            return False
        return int(self.current_task_idx) >= self._rolling_start_task_idx()

    def _rolling_snapshot_refresh_active_for_task_end(self) -> bool:
        if not getattr(self.config, "rolling_function_preservation", True):
            return False
        if self.current_task_idx is None:
            return False

        return int(self.current_task_idx) + 1 >= self._rolling_start_task_idx()

    @torch.no_grad()
    def _snapshot_generator_outputs_for_domain(self, domain_id: int) -> bool:
        domain_id = int(domain_id)
        proto = self.centroid_manager.prototype_features[domain_id]
        counts = self.centroid_manager.prototype_counts[domain_id]
        if float(counts.sum().item()) <= 0.0:
            return False

        gen_param = next(self.generator.parameters())
        proto_on_dev = proto.to(device=gen_param.device, dtype=gen_param.dtype)
        gen_out = self.generator.forward_all_slots(proto_on_dev, domain_id=domain_id)
        bucket_slot_ids = self.generator._bucket_slot_ids()
        for bucket in DCAGGenerator.D_IN_BUCKETS:
            slot_ids = bucket_slot_ids[bucket]
            if not slot_ids:
                continue
            snap_U = getattr(self, f"snap_U_{bucket}_{domain_id}")
            snap_V = getattr(self, f"snap_V_{bucket}_{domain_id}")
            slot_idx_t = torch.tensor(
                slot_ids, device=gen_out["U"].device, dtype=torch.long
            )
            U_subset = gen_out["U"].index_select(dim=1, index=slot_idx_t)
            V_subset = gen_out["V"][str(bucket)]
            snap_U.copy_(U_subset.to(snap_U.dtype).to(snap_U.device))
            snap_V.copy_(V_subset.to(snap_V.dtype).to(snap_V.device))

        if getattr(self.config, "hidden_state_consistency_enabled", True):
            trunk_name = f"snap_trunk_{domain_id}"
            trunk_buf = getattr(self, trunk_name, None)
            if trunk_buf is not None and "trunk" in gen_out:
                trunk_buf.copy_(
                    gen_out["trunk"].to(trunk_buf.dtype).to(trunk_buf.device)
                )

        self.snapshot_mask[domain_id] = 1.0
        del gen_out
        return True

    @torch.no_grad()
    def snapshot_and_end_task(self):
        """Snapshot generator responses, extend historical subspaces, and freeze task slices."""
        if self.current_task_idx is None:
            return
        was_training = self.training
        self.eval()
        try:
            final_metrics = self.task_slice_manager.retract_R_current()
            if final_metrics is not None:
                max_off_diag, max_off_orth = final_metrics
                print(
                    f"[DCAG][snapshot] final R orthogonality: "
                    f"max|R_old^T R_cur|={max_off_diag:.2e} "
                    f"max|R_cur^T R_cur - I|={max_off_orth:.2e}"
                )

            self.centroid_manager.sync_current_domain_only(self.current_domain_id)
            for tracker in self.fisher_trackers.values():
                tracker.sync_across_ranks()

            if self.current_domain_id is not None:
                domain_id = self.current_domain_id
                if self._rolling_snapshot_refresh_active_for_task_end():
                    seen_domains = [
                        d
                        for d in range(len(DCAG_DOMAIN_TO_ID))
                        if d == domain_id or float(self.snapshot_mask[d].item()) > 0
                    ]
                    refreshed = [
                        d
                        for d in seen_domains
                        if self._snapshot_generator_outputs_for_domain(d)
                    ]
                    if refreshed:
                        names = ",".join(
                            DCAG_ID_TO_DOMAIN.get(d, str(d)) for d in refreshed
                        )
                        print(
                            f"[DCAG][rolling] refreshed full-function snapshots for domains={names}",
                            flush=True,
                        )
                else:
                    self._snapshot_generator_outputs_for_domain(domain_id)

                if getattr(self.config, "shared_projection_snapshot_enabled", True):
                    snap_name = f"shared_proj_snapshot_{domain_id}"
                    snap_buf = getattr(self, snap_name, None)
                    if snap_buf is not None:
                        flat = self._flatten_shared_projection().to(
                            snap_buf.device, snap_buf.dtype
                        )
                        snap_buf.copy_(flat)

            if self.config.Wd_mode != "none_disable_A_projection":
                task_idx = int(self.current_task_idx)
                for slot_id in range(self.num_total_slots):
                    d_in = self._slot_d_in(slot_id)
                    tracker = self.get_fisher_tracker(slot_id, d_in)
                    if tracker is None:
                        continue
                    P_prev = self.get_P_cumulative(slot_id)
                    if P_prev is None:
                        continue
                    W_d = tracker.extract(self.config.k_d, P_prev)
                    if W_d.shape[1] == 0:
                        continue
                    W_d = W_d.to(device=P_prev.device, dtype=P_prev.dtype)

                    self.register_buffer(
                        f"W_slot_{slot_id}_task_{task_idx}", W_d.detach().clone()
                    )
                    new_P = torch.cat([P_prev, W_d], dim=1)

                    self._buffers[f"P_slot_{slot_id}"] = new_P

            self.task_slice_manager.end_task(self.current_task_idx)

            if self.residual_lora_manager is not None:
                self.residual_lora_manager.end_task(int(self.current_task_idx))

            if self.current_domain_id is not None:
                self.task_slice_manager.record_domain_for_task(
                    int(self.current_task_idx), int(self.current_domain_id)
                )

        finally:
            if was_training:
                self.train()

    def update_task_idx_from_state(self):
        """After loading state_dict, infer the next task idx from task_slice_map."""
        return self.task_slice_manager.infer_next_task_idx()

    def write_dcag_config(self, output_dir: str):
        try:
            os.makedirs(output_dir, exist_ok=True)
            snapshot = {
                k: getattr(self.config, k, None)
                for k in [
                    "B_mode",
                    "R_retraction",
                    "Wd_mode",
                    "num_centroids",
                    "slot_weight_mode",
                    "projector_mode",
                    "dynamic_current_B",
                    "generator_arch",
                    "domain_feature_mode",
                    "preservation_target",
                    "anti_forgetting_weight",
                    "anti_forgetting_weight_per_domain",
                    "num_centroids_per_domain",
                    "shared_projection_snapshot_enabled",
                    "hidden_state_consistency_enabled",
                    "rope_layer_enabled",
                    "domain_embed_enabled",
                    "spectral_balance_coeff",
                    "spectral_balance_every_n_steps",
                    "preservation_trunk_coeff",
                    "preservation_uv_coeff",
                    "new_slice_suppression_coeff",
                    "functional_preservation_coeff",
                    "rolling_function_preservation",
                    "rolling_start_task_idx",
                    "generator_hidden_dim",
                    "generator_layers",
                    "generator_heads",
                    "factored_head_dim",
                    "r",
                    "h_d_llm",
                    "h_d_projector",
                    "k_d",
                    "r_fisher",
                    "generator_dropout",
                    "a_l2_penalty",
                    "feature_token_dropout",
                    "feature_noise_std",
                    "feature_regularize_generator_only",
                    "projector_preservation_multiplier",
                    "projector_warmup_steps",
                    "projector_cr_lr_multiplier",
                    "projector_disable_dynamic_B",
                    "generator_cache_mode",
                    "b_composition_mode",
                    "domain_id_source",
                    "domain_image_resize_mode",
                    "alignment_coeff",
                    "instruction_conditioning",
                    "instruction_feature_dim",
                    "attn_pool_queries",
                    "residual_rank",
                    "residual_lr",
                    "residual_alpha",
                ]
            }
            with open(os.path.join(output_dir, "dcag_config.json"), "w") as f:
                json.dump(snapshot, f, indent=2)
        except OSError:
            pass


class DCAGModelMixin:
    """Methods mixed into LlavaLlamaForCausalLM to own the DCAG controller."""

    def initialize_dcag(self, dcag_config: Optional[Dict] = None):
        cfg_dict = dcag_config or {}
        cfg = DCAGConfig(**cfg_dict)
        self.dcag_enabled = cfg.enabled
        self.dcag_controller = None
        if not cfg.enabled:
            return

        num_hidden_layers = getattr(
            self.config, "num_hidden_layers", len(self.model.layers)
        )
        self.dcag_controller = DCAGController(cfg, num_hidden_layers)
        self.config.dcag = cfg_dict
        self._inject_dcag_layers(cfg)

    def inject_dcag_projector_if_ready(self):
        """Wrap initialized projector linears when integrated adaptation is enabled."""
        if self.dcag_controller is None:
            return
        cfg = self.dcag_controller.config
        if cfg.projector_mode not in ("integrated_with_policy", "integrated_no_policy"):
            return
        self._inject_dcag_projector_layers(cfg)

    def _inject_dcag_layers(self, cfg: DCAGConfig):
        attn_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
        for layer_idx, layer in enumerate(self.model.layers):
            for target_name in DCAG_TARGET_MODULES:
                parent = layer.self_attn if target_name in attn_targets else layer.mlp
                module = getattr(parent, target_name)
                if isinstance(module, DCAGLinear):
                    continue
                wrapped = DCAGLinear(
                    base_layer=module,
                    controller=self.dcag_controller,
                    layer_idx=layer_idx,
                    target_name=target_name,
                    kind="llm",
                    alpha=cfg.alpha,
                )
                setattr(parent, target_name, wrapped)

        self._wrap_decoder_layer_forwards()

    def _wrap_decoder_layer_forwards(self):
        """Wrap decoder layers with factor contexts and optional non-reentrant checkpointing."""
        import torch.utils.checkpoint as _ckpt

        ctrl = self.dcag_controller
        if ctrl is None:
            return
        for layer_idx, layer in enumerate(self.model.layers):
            if getattr(layer, "_dcag_forward_wrapped", False):
                continue
            layer.dcag_layer_idx = layer_idx
            original_forward = layer.forward

            def make_wrapped(orig, idx):
                def wrapped(hidden_states, *args, **kwargs):
                    def fn(hs, *a, **kw):
                        with ctrl.layer_context(idx):
                            return orig(hs, *a, **kw)

                    do_ckpt = (
                        ctrl.config.generator_cache_mode == "layer_recompute"
                        and ctrl.training
                        and isinstance(hidden_states, torch.Tensor)
                        and hidden_states.requires_grad
                    )
                    if do_ckpt:
                        return _ckpt.checkpoint(
                            fn, hidden_states, *args, use_reentrant=False, **kwargs
                        )
                    return fn(hidden_states, *args, **kwargs)

                return wrapped

            layer.forward = make_wrapped(original_forward, layer_idx)
            layer._dcag_forward_wrapped = True

    def _inject_dcag_projector_layers(self, cfg: DCAGConfig):
        """Wrap mm_projector.0 and mm_projector.2 as DCAGLinear slots 224, 225."""
        mm_projector = getattr(self.model, "mm_projector", None)
        if mm_projector is None:
            return

        if not isinstance(mm_projector, nn.Sequential):
            return
        for projector_idx, module_idx in enumerate((0, 2)):
            if module_idx >= len(mm_projector):
                continue
            module = mm_projector[module_idx]
            if not isinstance(module, nn.Linear):
                continue
            if isinstance(module, DCAGLinear):
                continue
            wrapped = DCAGLinear(
                base_layer=module,
                controller=self.dcag_controller,
                layer_idx=-1,
                target_name=None,
                kind="projector",
                projector_idx=projector_idx,
                alpha=cfg.alpha,
            )
            mm_projector[module_idx] = wrapped

        self._wrap_projector_forward()

    def _wrap_projector_forward(self):
        ctrl = self.dcag_controller
        if ctrl is None:
            return
        mm_projector = getattr(self.model, "mm_projector", None)
        if mm_projector is None:
            return
        if getattr(mm_projector, "_dcag_proj_wrapped", False):
            return
        original_forward = mm_projector.forward

        def proj_wrapped(*args, **kwargs):
            with ctrl.projector_context():
                return original_forward(*args, **kwargs)

        mm_projector.forward = proj_wrapped
        mm_projector._dcag_proj_wrapped = True

    def set_dcag_context(
        self,
        images,
        domain_ids: Optional[torch.Tensor],
        instruction_features: Optional[torch.Tensor] = None,
    ):
        if self.dcag_controller is None or images is None or domain_ids is None:
            return
        self.dcag_controller.set_batch_context(
            images, domain_ids, instruction_features=instruction_features
        )

    def clear_dcag_context(self):
        if self.dcag_controller is not None:
            self.dcag_controller.clear_batch_context()

    def finalize_dcag_batch(self):
        if self.dcag_controller is not None:
            self.dcag_controller.finalize_batch()

    def snapshot_dcag_coefficients(self):
        """End-of-task hook: snapshot generator outputs, extract W_d, freeze R/C slices."""
        if self.dcag_controller is not None:
            self.dcag_controller.snapshot_and_end_task()

    def start_dcag_task(self, task_idx: int):
        if self.dcag_controller is not None:
            self.dcag_controller.start_task(task_idx)

    def get_dcag_preservation_loss(self) -> Optional[torch.Tensor]:
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.latest_preservation_loss

    def get_dcag_new_slice_suppression_loss(self) -> Optional[torch.Tensor]:
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.latest_new_slice_suppression_loss

    def get_dcag_functional_preservation_loss(self) -> Optional[torch.Tensor]:
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.latest_functional_preservation_loss

    def get_dcag_a_l2_penalty(self) -> Optional[torch.Tensor]:
        """Return the training factor penalty before applying config.a_l2_penalty."""
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.compute_a_l2_penalty()

    def get_dcag_alignment_loss(
        self, teacher: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Return the training alignment loss before applying config.alignment_coeff."""
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.compute_alignment_loss(teacher)

    def get_dcag_spectral_balance_penalty(self) -> Optional[torch.Tensor]:
        if self.dcag_controller is None:
            return None
        return self.dcag_controller.compute_spectral_balance_penalty()

    def get_dcag_config(self) -> Optional[Dict]:
        return getattr(self.config, "dcag", None)

    def has_dcag(self) -> bool:
        return self.dcag_controller is not None
