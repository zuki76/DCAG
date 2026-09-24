#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.model.dcag import (
    normalize_dcag_config_paths,
    resolve_dcag_repo_path,
)


def _load_non_lora_trainables(model_path):
    local_path = os.path.join(model_path, "non_lora_trainables.bin")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Expected local DCAG weights at {local_path}")
    non_lora_trainables = torch.load(local_path, map_location="cpu")
    non_lora_trainables = {
        (k[11:] if k.startswith("base_model.") else k): v
        for k, v in non_lora_trainables.items()
    }
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in non_lora_trainables.items()
        }
    return non_lora_trainables


def _is_dcag_config(cfg):
    dcag = getattr(cfg, "dcag", None)
    return isinstance(dcag, dict) and dcag.get("enabled", False)


def _repair_dcag_checkpoint_config_paths(cfg):
    dcag = getattr(cfg, "dcag", None)
    if isinstance(dcag, dict):
        cfg.dcag = normalize_dcag_config_paths(dcag)

    vision_tower = getattr(cfg, "mm_vision_tower", None)
    fixed_vision_tower = resolve_dcag_repo_path(
        vision_tower,
        ("checkpoints", "clip-vit-large-patch14-336"),
        "mm_vision_tower",
    )
    if fixed_vision_tower is not None and fixed_vision_tower != vision_tower:
        cfg.mm_vision_tower = fixed_vision_tower
    return cfg


def _apply_dcag_eval_overrides(cfg):
    """Apply adapter-composition overrides from environment variables."""
    dcag = getattr(cfg, "dcag", None)
    if not isinstance(dcag, dict):
        return cfg
    b_mode = os.environ.get("DCAG_B_COMPOSITION_MODE_OVERRIDE")
    if b_mode and b_mode != "None":
        dcag["b_composition_mode"] = b_mode
        print(f"[DCAG][eval override] b_composition_mode={b_mode}", flush=True)
    domain_id_source = os.environ.get("DCAG_DOMAIN_ID_SOURCE_OVERRIDE")
    if domain_id_source and domain_id_source != "None":
        dcag["domain_id_source"] = domain_id_source
        print(f"[DCAG][eval override] domain_id_source={domain_id_source}", flush=True)
    cfg.dcag = dcag
    return cfg


def _resolve_model_path(model_path, model_base):
    return model_base if model_base is not None else model_path


def _load_llava_model_with_cfg(model_path, model_base, cfg_pretrained, kwargs):
    base_path = _resolve_model_path(model_path, model_base)
    load_kwargs = dict(kwargs)
    load_kwargs.setdefault("local_files_only", True)
    if _is_dcag_config(cfg_pretrained):
        target_device = load_kwargs.pop("device_map", None)
        load_kwargs["low_cpu_mem_usage"] = False
        model = LlavaLlamaForCausalLM.from_pretrained(
            base_path, config=cfg_pretrained, **load_kwargs
        )
        if isinstance(target_device, dict):
            target_device = target_device.get("", None)
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(target_device, str):
            model = model.to(target_device)
        return model
    load_kwargs["low_cpu_mem_usage"] = True
    return LlavaLlamaForCausalLM.from_pretrained(
        base_path, config=cfg_pretrained, **load_kwargs
    )


def _load_tokenizer(model_path, model_base, use_fast=False):
    base_path = _resolve_model_path(model_path, model_base)
    return AutoTokenizer.from_pretrained(
        base_path, use_fast=use_fast, local_files_only=True
    )


def _is_base_llama_weight(key: str) -> bool:

    if key.endswith(".inv_freq"):
        return True
    if key.endswith(".base_layer.weight") or key.endswith(".base_layer.bias"):
        return True
    if key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        return True
    if key.endswith(".input_layernorm.weight") or key.endswith(
        ".post_attention_layernorm.weight"
    ):
        return True

    if ".vision_tower." in key or ".mm_projector." in key:
        return True

    if ".feature_encoder.pipelines." in key and ".extractor.extractor." in key:
        return True

    if "dcag_controller.task_slice_manager." in key and key.endswith("_current"):
        return True

    if "dcag_controller.residual_lora_manager." in key and key.endswith("_current"):
        return True

    if key.endswith("._extra_state"):
        return True
    return False


def _resize_dcag_variable_buffers(model, state_dict):
    """Register and resize task-dependent buffers before checkpoint loading.

    Restores historical subspaces, task slices, and domain residuals.
    """
    variable_patterns = ("P_slot_", "W_slot_", "_frozen", "residual_")
    for key, val in state_dict.items():
        if not any(pat in key for pat in variable_patterns):
            continue
        parts = key.split(".")
        module = model
        for part in parts[:-1]:
            module = getattr(module, part, None)
            if module is None:
                break
        if module is None:
            continue
        buf_name = parts[-1]
        buffers = getattr(module, "_buffers", {})
        if buf_name not in buffers:
            if (
                buf_name.startswith("W_slot_") or buf_name.startswith("residual_")
            ) and "_task_" in buf_name:
                ref_dev = None
                for sib in buffers.values():
                    if sib is not None:
                        ref_dev = sib.device
                        break
                if ref_dev is None:
                    ref_dev = torch.device("cpu")
                module.register_buffer(
                    buf_name,
                    torch.zeros(*val.shape, dtype=val.dtype, device=ref_dev),
                )
            continue
        existing = buffers[buf_name]
        if existing is None:
            continue
        if existing.shape == val.shape:
            continue
        module._buffers[buf_name] = torch.zeros(
            *val.shape, dtype=existing.dtype, device=existing.device
        )


def _load_dcag_weights(model, model_path):
    non_lora_trainables = _load_non_lora_trainables(model_path)

    _resize_dcag_variable_buffers(model, non_lora_trainables)

    extra_task_slice_map_key = (
        "dcag_controller.task_slice_manager._extra_state_task_slice_map"
    )
    extra_task_slice_map_val = non_lora_trainables.pop(extra_task_slice_map_key, None)
    missing, unexpected = model.load_state_dict(non_lora_trainables, strict=False)

    if (
        extra_task_slice_map_val is not None
        and getattr(model, "dcag_controller", None) is not None
    ):
        try:
            import json as _json

            raw_bytes = bytes(extra_task_slice_map_val.tolist())
            payload = _json.loads(raw_bytes.decode("utf-8"))

            if "task_slice_map" not in payload:
                payload = {"task_slice_map": payload}
            model.dcag_controller.task_slice_manager.set_extra_state(payload)
            tsm = model.dcag_controller.task_slice_manager
            print(
                f"DCAG task_slice_map restored: tasks={sorted(tsm.task_slice_map.keys())} "
                f"domain_to_task={tsm.domain_to_task}"
            )
        except Exception as exc:
            print(f"DCAG task_slice_map restore warning: {exc}")
    filtered_missing = [k for k in missing if not _is_base_llama_weight(k)]
    if filtered_missing or unexpected:
        print(
            f"DCAG load_state_dict missing={len(filtered_missing)} unexpected={len(unexpected)}"
        )
        if filtered_missing:
            print("DCAG missing sample:", filtered_missing[:20])
        if unexpected:
            print("DCAG unexpected sample:", unexpected[:20])
    return model


def _load_peft_lora_weights(model, model_path):
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()
    return model


def _move_dcag_to_model_device(model):
    """Move the controller to the base model device and dtype.

    Lazy domain encoders use their separately configured device on first access.
    """
    ctrl = getattr(model, "dcag_controller", None)
    if ctrl is None:
        return
    target_device = None
    target_dtype = None
    for p in model.parameters():
        if p.is_floating_point():
            target_device = p.device
            target_dtype = p.dtype
            break
    if target_device is None:
        return
    ctrl.to(device=target_device, dtype=target_dtype)


def _load_llava_variant(model_path, model_base, model_name, kwargs):
    cfg_pretrained = AutoConfig.from_pretrained(model_path, local_files_only=True)
    cfg_pretrained = _repair_dcag_checkpoint_config_paths(cfg_pretrained)
    cfg_pretrained = _apply_dcag_eval_overrides(cfg_pretrained)
    tokenizer = _load_tokenizer(model_path, model_base, use_fast=False)
    model = _load_llava_model_with_cfg(model_path, model_base, cfg_pretrained, kwargs)
    if _is_dcag_config(cfg_pretrained):
        model.initialize_dcag(cfg_pretrained.dcag)

        if hasattr(model, "inject_dcag_projector_if_ready"):
            model.inject_dcag_projector_if_ready()
        print("Loading DCAG weights...")
        model = _load_dcag_weights(model, model_path)

        _move_dcag_to_model_device(model)

        model.eval()
        print("DCAG model is loaded...")
        return tokenizer, model

    if "lora" in model_name.lower() and model_base is not None:
        print("Loading LLaVA from base model...")
        model = _load_peft_lora_weights(model, model_path)
        print("Model is loaded...")
        return tokenizer, model

    if model_base is not None:
        print("Loading LLaVA from base model...")
        mm_projector_weights = torch.load(
            os.path.join(model_path, "mm_projector.bin"), map_location="cpu"
        )
        mm_projector_weights = {
            k: v.to(torch.float16) for k, v in mm_projector_weights.items()
        }
        model.load_state_dict(mm_projector_weights, strict=False)
    return tokenizer, model


def _is_llava_checkpoint(model_path, model_name):
    if "llava" in model_name.lower():
        return True
    try:
        cfg = AutoConfig.from_pretrained(model_path, local_files_only=True)
    except Exception:
        return False

    if getattr(cfg, "model_type", None) == "llava":
        return True

    architectures = getattr(cfg, "architectures", None) or []
    return any("llava" in arch.lower() for arch in architectures)


def _prepare_vision_modules(model, tokenizer, device):
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    return vision_tower.image_processor


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    is_llava_checkpoint = _is_llava_checkpoint(model_path, model_name)

    if not is_llava_checkpoint:
        raise ValueError("This implementation supports LLaVA-1.5 checkpoints.")
    tokenizer, model = _load_llava_variant(model_path, model_base, model_name, kwargs)

    image_processor = None
    if is_llava_checkpoint:
        image_processor = _prepare_vision_modules(model, tokenizer, device)

        if hasattr(model, "inject_dcag_projector_if_ready"):
            model.inject_dcag_projector_if_ready()

        _move_dcag_to_model_device(model)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
