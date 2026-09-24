"""Set local vision-tower paths in a downloaded LLaVA model configuration."""

import argparse
import json
from _common import path, read_json, require_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model.json")
    args = parser.parse_args()
    model = read_json(args.model_config)
    target = path(model["model_name"]) / "config.json"
    vision = path(model["vision_tower"])
    require_paths([target, vision / "config.json"])
    cfg = read_json(target)
    cfg.update(
        {
            "mm_vision_tower": str(vision),
            "mm_text_tower": str(vision),
            "mm_text_select_layer": -1,
        }
    )
    target.write_text(json.dumps(cfg, indent=2) + "\n")
    generation = target.parent / "generation_config.json"
    if generation.exists():
        cfg = read_json(generation)
        if not cfg.get("do_sample", False):
            cfg.pop("temperature", None)
            cfg.pop("top_p", None)
        generation.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Prepared {target}")


if __name__ == "__main__":
    main()
