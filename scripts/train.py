"""Train the five-domain DCAG sequence or a selected contiguous stage range."""

import argparse
from _common import (
    DOMAINS,
    ROOT,
    arguments,
    environment,
    path,
    read_json,
    require_paths,
    run,
)


def training_command(task, config_dir, model_config, gpus, port):
    cfg = read_json(path(config_dir) / f"task{task}.json")
    model = read_json(model_config)
    domain = DOMAINS[task - 1]
    if cfg["domain"] != domain or cfg["dcag_task_idx"] != task - 1:
        raise ValueError(
            f"Task {task} must correspond to {domain} and index {task - 1}"
        )
    if len(gpus.split(",")) != cfg["gpu_num"]:
        raise ValueError("GPU count must match gpu_num in each training config")
    data = read_json(f"configs/data/{domain.lower()}.json")
    values = {
        "deepspeed": str(ROOT / "configs/zero2.json"),
        "model_name_or_path": str(path(model["model_name"])),
        "vision_tower": str(path(model["vision_tower"])),
        "version": "v1",
        "domain_name": domain,
        "data_path": str(path(data["train_path"])),
        "image_folder": str(path(data["train_folder"])),
        "lora_enable": False,
        "lora_r": cfg["rank"],
        "lora_alpha": cfg["rank"] * 2,
        "mm_projector_type": "mlp2x_gelu",
        "mm_vision_select_layer": -2,
        "mm_use_im_start_end": False,
        "mm_use_im_patch_token": False,
        "image_aspect_ratio": "pad",
        "group_by_modality_length": False,
        "bf16": True,
        "output_dir": str(path(cfg["output_dir"])),
        "num_train_epochs": cfg["epoch"],
        "per_device_train_batch_size": cfg["batch_size"],
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": cfg["grad_acc"],
        "evaluation_strategy": "no",
        "save_strategy": "steps",
        "save_steps": 50000,
        "learning_rate": cfg["lr"],
        "seed": cfg["seed"],
        "weight_decay": 0.05,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "logging_steps": 10,
        "tf32": True,
        "model_max_length": 2048,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 4,
        "lazy_preprocess": True,
        "report_to": "none",
    }
    values.update({k: v for k, v in cfg.items() if k.startswith("dcag_")})
    values.update({k: str(path(v)) for k, v in model["domain_encoders"].items()})
    if model.get("mm_projector"):
        values["pretrain_mm_mlp_adapter"] = str(path(model["mm_projector"]))
    if task > 1:
        values["previous_task_model_path"] = str(path(cfg["previous_model"]))
    command = [
        "deepspeed",
        "--include",
        f"localhost:{gpus}",
        "--master_port",
        str(port),
        str(ROOT / "llava/train/train_mem.py"),
        *arguments(values),
    ]
    inputs = [
        model["model_name"],
        model["vision_tower"],
        data["train_path"],
        data["train_folder"],
        *model["domain_encoders"].values(),
    ]
    if model.get("mm_projector"):
        inputs.append(model["mm_projector"])
    if task > 1:
        inputs += [
            path(cfg["previous_model"]) / "config.json",
            path(cfg["previous_model"]) / "non_lora_trainables.bin",
        ]
    return command, inputs, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-task", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--end-task", type=int, choices=range(1, 6), default=5)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--model-config", default="configs/model.json")
    parser.add_argument("--config-dir", default="configs/train")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without loading weights or launching GPUs",
    )
    args = parser.parse_args()
    if args.start_task > args.end_task:
        parser.error("start-task must not exceed end-task")
    ids = args.gpus.split(",")
    if not all(x.isdigit() for x in ids) or len(set(ids)) != len(ids):
        parser.error("gpus must be unique comma-separated device indices")
    env = environment()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    for task in range(args.start_task, args.end_task + 1):
        command, inputs, cfg = training_command(
            task, args.config_dir, args.model_config, args.gpus, args.master_port
        )
        if not args.dry_run:
            require_paths(inputs)
            output = path(cfg["output_dir"])
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Training output is not empty: {output}. Choose a new output_dir."
                )
        effective = cfg["gpu_num"] * cfg["batch_size"] * cfg["grad_acc"]
        print(
            f"Task {task}: {cfg['domain']}; effective batch size {effective}",
            flush=True,
        )
        run(command, env, args.dry_run)


if __name__ == "__main__":
    main()
