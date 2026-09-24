"""Evaluate a DCAG checkpoint on the five MLLM-DCL domains."""

import argparse
from collections import Counter
import json
import sys
from _common import DOMAINS, environment, path, read_json, require_paths, run

MODELS = {
    "RS": "model_ai2d",
    "Med": "model_pvqa",
    "AD": "model_ai2d",
    "Sci": "model_ai2d",
    "Fin": "model_fin",
}
METRICS = {
    "RS": "eval_ai2d",
    "Med": "eval_pvqa",
    "AD": "eval_ai2d",
    "Sci": "eval_sci",
    "Fin": "eval_finvis",
}


def validate_predictions(annotation_file, result_file):
    annotations = read_json(annotation_file)
    expected = Counter(x["question_id"] for x in annotations)
    with path(result_file).open() as handle:
        actual = Counter(
            json.loads(line)["question_id"] for line in handle if line.strip()
        )
    if not expected or any(n != 1 for n in expected.values()) or actual != expected:
        raise ValueError(
            "Predictions must contain exactly one answer for every unique annotation question_id"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=list(DOMAINS))
    parser.add_argument(
        "--gpu", default="0", help="One GPU index; domains run sequentially"
    )
    parser.add_argument("--routing", choices=["prototype", "oracle"])
    parser.add_argument("--prototypes", default="checkpoints/router_prototypes.pt")
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--eval-config", default="configs/eval.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    model = read_json(args.model_config)
    cfg = read_json(args.eval_config)
    routing = args.routing or cfg["routing"]
    if cfg["batch_size"] != 1:
        parser.error(
            "This evaluation entrypoint requires batch_size=1 for sample-conditioned generation"
        )
    if not args.gpu.isdigit():
        parser.error("gpu must be one non-negative device index")
    env = environment()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "DCAG_DOMAIN_ROUTER": routing,
            "DCAG_ROUTER_PROTOTYPES": str(path(args.prototypes)),
            "DCAG_ROUTER_CLIP": str(path(model["vision_tower"])),
        }
    )
    if not args.dry_run:
        require_paths(
            [
                path(args.checkpoint) / "config.json",
                path(args.checkpoint) / "non_lora_trainables.bin",
                model["model_name"],
                model["vision_tower"],
                *model["domain_encoders"].values(),
            ]
        )
        if routing == "prototype":
            require_paths([args.prototypes])
    for domain in args.domains:
        data = read_json(f"configs/data/{domain.lower()}.json")
        output = (
            path(cfg["output_dir"]) / f"{path(args.checkpoint).name}_{routing}" / domain
        )
        answers = output / "predictions.jsonl"
        command = [
            sys.executable,
            "-m",
            "llava.eval." + MODELS[domain],
            "--model-path",
            str(path(args.checkpoint)),
            "--model-base",
            str(path(model["model_name"])),
            "--question-file",
            str(path(data["test_path"])),
            "--image-folder",
            str(path(data["test_folder"])),
            "--answers-file",
            str(answers),
            "--conv-mode",
            cfg["conv_mode"],
            "--temperature",
            "0",
            "--domain-name",
            domain,
            "--batch-size",
            "1",
        ]
        if not args.dry_run:
            require_paths([data["test_path"], data["test_folder"]])
            output.mkdir(parents=True, exist_ok=True)
        run(command, env, args.dry_run)
        if not args.dry_run:
            validate_predictions(data["test_path"], answers)
        run(
            [
                sys.executable,
                "-m",
                "llava.eval." + METRICS[domain],
                "--annotation-file",
                str(path(data["test_path"])),
                "--result-file",
                str(answers),
                "--output-dir",
                str(output),
            ],
            env,
            args.dry_run,
        )


if __name__ == "__main__":
    main()
