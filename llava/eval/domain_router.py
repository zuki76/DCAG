import os
from pathlib import Path

import torch
import torch.nn.functional as F

_DOMAIN_ID = {"RS": 0, "Med": 1, "AD": 2, "Sci": 3, "Fin": 4}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PROTOTYPES = _REPO_ROOT / "checkpoints" / "router_prototypes.pt"
_DEFAULT_CLIP = _REPO_ROOT / "checkpoints" / "clip-vit-large-patch14-336"
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)


class _Router:
    def __init__(self):
        proto_path = os.environ.get("DCAG_ROUTER_PROTOTYPES", str(_DEFAULT_PROTOTYPES))
        clip_path = os.environ.get("DCAG_ROUTER_CLIP", str(_DEFAULT_CLIP))
        from transformers import CLIPModel

        bank = torch.load(proto_path, map_location="cpu")
        self.domains = bank["domains"]
        if self.domains != list(_DOMAIN_ID):
            raise ValueError(
                "Router prototypes must follow RS, Med, AD, Sci, Fin order"
            )
        self.centroids = bank["centroids"].float()

        self.device = torch.device("cpu")
        self.model = (
            CLIPModel.from_pretrained(clip_path).vision_model.eval().to(self.device)
        )
        for p in self.model.parameters():
            p.requires_grad = False
        self.mean = torch.tensor(_MEAN).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor(_STD).view(1, 3, 1, 1).to(self.device)
        print(
            f"[domain-router] prototypes={proto_path} clip={clip_path} device={self.device}",
            flush=True,
        )

    @torch.no_grad()
    def route(self, pil_image) -> int:
        import numpy as np
        from PIL import Image

        im = pil_image.convert("RGB").resize((336, 336), Image.BICUBIC)
        x = (
            torch.from_numpy(np.array(im))
            .permute(2, 0, 1)
            .float()
            .div_(255.0)
            .unsqueeze(0)
            .to(self.device)
        )
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        pooled = (
            out.pooler_output
            if getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state[:, 0]
        )
        feat = F.normalize(pooled.float(), dim=-1).cpu()
        sims = feat @ self.centroids.T
        return int(sims.argmax(dim=-1).item())


_INSTANCE = None


def _get_router() -> _Router:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _Router()
    return _INSTANCE


def resolve_domain_id(oracle_domain_name: str, pil_image) -> int:
    """Infer the domain from CLIP prototypes, or use an explicitly selected oracle mode."""
    mode = os.environ.get("DCAG_DOMAIN_ROUTER", "prototype").strip().lower()
    if mode == "oracle":
        return _DOMAIN_ID[oracle_domain_name]
    if mode != "prototype":
        raise ValueError(f"Unknown routing mode: {mode}")
    return _get_router().route(pil_image)
