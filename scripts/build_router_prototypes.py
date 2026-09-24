"""Build nearest-centroid CLIP prototypes from training images only."""

import argparse
import random
from _common import DOMAINS, path, read_json, require_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--output", default="checkpoints/router_prototypes.pt")
    parser.add_argument("--samples-per-domain", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.samples_per_domain < 1 or args.batch_size < 1:
        parser.error("sample and batch counts must be positive")
    model_cfg = read_json(args.model_config)
    require_paths([model_cfg["vision_tower"]])
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel

    model = (
        CLIPModel.from_pretrained(
            str(path(model_cfg["vision_tower"])), local_files_only=True
        )
        .vision_model.eval()
        .to(args.device)
    )
    model.requires_grad_(False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=args.device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=args.device).view(
        1, 3, 1, 1
    )
    rng = random.Random(args.seed)
    centroids = []
    with torch.inference_mode():
        for domain in DOMAINS:
            cfg = read_json(f"configs/data/{domain.lower()}.json")
            rows = read_json(cfg["train_path"])
            if not rows:
                raise ValueError(f"Empty training split: {domain}")
            chosen = rng.sample(rows, min(args.samples_per_domain, len(rows)))
            features = []
            for start in range(0, len(chosen), args.batch_size):
                images = []
                for row in chosen[start : start + args.batch_size]:
                    filename = path(cfg["train_folder"]) / row["image"]
                    if not filename.exists() and not filename.suffix:
                        filename = next(
                            (
                                filename.with_suffix(ext)
                                for ext in (".png", ".jpg", ".jpeg")
                                if filename.with_suffix(ext).exists()
                            ),
                            filename,
                        )
                    with Image.open(filename) as image:
                        pixels = np.array(
                            image.convert("RGB").resize(
                                (336, 336), Image.Resampling.BICUBIC
                            )
                        )
                    images.append(
                        torch.from_numpy(pixels).permute(2, 0, 1).float() / 255
                    )
                inputs = (torch.stack(images).to(args.device) - mean) / std
                outputs = model(pixel_values=inputs)
                pooled = (
                    outputs.pooler_output
                    if outputs.pooler_output is not None
                    else outputs.last_hidden_state[:, 0]
                )
                features.append(F.normalize(pooled.float(), dim=-1).cpu())
            centroids.append(F.normalize(torch.cat(features).mean(0), dim=0))
            print(f"{domain}: {len(chosen)} training images", flush=True)
    output = path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "domains": list(DOMAINS),
            "centroids": torch.stack(centroids),
            "seed": args.seed,
            "samples_per_domain": args.samples_per_domain,
        },
        output,
    )
    print(f"Saved prototypes to {output}")


if __name__ == "__main__":
    main()
