# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import copy
from dataclasses import dataclass, field
import json
import logging
import random
from typing import Dict, Optional, Sequence

import torch
import numpy as np
import transformers
import subprocess

from llava.model.dcag import DCAG_DOMAIN_TO_ID, normalize_dcag_config_paths
from llava.eval._ad_split import apply_ad_split, should_split_ad_mosaic

from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from peft.utils import WEIGHTS_NAME, set_peft_model_state_dict
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.mm_utils import tokenizer_image_token

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def parse_optional_sequence_arg(value, cast):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [cast(x) for x in value]
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parsed = [item.strip() for item in text.split(",") if item.strip()]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Expected a sequence argument, got {value!r}")
    return [cast(x) for x in parsed]


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    previous_task_model_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    domain_name: Optional[str] = field(default=None)
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    memory_data_path: str = field(
        default=None, metadata={"help": "Path to the memory data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    dcag_enable: bool = field(default=False)
    dcag_feature_dim: int = field(default=512)
    dcag_num_bases: int = field(default=4)
    dcag_basis_rank: int = field(default=8)
    dcag_alpha: Optional[float] = field(default=None)
    dcag_generator_hidden_dim: int = field(default=512)
    dcag_basis_lr: Optional[float] = field(default=None)
    dcag_generator_lr: Optional[float] = field(default=None)
    dcag_anti_forgetting_weight: float = field(default=0.1)
    dcag_anti_forgetting_weight_per_domain: Optional[str] = field(default=None)
    dcag_feature_layers: str = field(default="-3,-2,-1")
    dcag_train_domain_backbones: bool = field(default=False)
    dcag_domain_model_device: str = field(default="cuda")
    dcag_domain_model_dtype: str = field(default="float16")
    rs_domain_model_path: Optional[str] = field(default=None)
    dcag_rs_extractor_type: str = field(default="rvsa")
    med_domain_model_path: Optional[str] = field(default=None)
    dcag_med_extractor_type: str = field(default="pubmedclip")
    ad_domain_model_path: Optional[str] = field(default=None)
    dcag_ad_extractor_type: str = field(default="geomim")
    sci_domain_model_path: Optional[str] = field(default=None)
    dcag_sci_extractor_type: str = field(default="pix2struct")
    fin_domain_model_path: Optional[str] = field(default=None)
    dcag_fin_extractor_type: str = field(default="candlefusion")

    dcag_r: int = field(default=32)
    dcag_h_d_llm: int = field(default=4)
    dcag_h_d_projector: int = field(default=2)
    dcag_k_d: int = field(default=4)
    dcag_r_fisher: int = field(default=32)
    dcag_num_centroids: int = field(default=4)
    dcag_num_centroids_per_domain: Optional[str] = field(default=None)
    dcag_generator_layers: int = field(default=4)
    dcag_generator_heads: int = field(default=8)
    dcag_factored_head_dim: int = field(default=8)

    dcag_generator_dropout: float = field(default=0.1)
    dcag_a_l2_penalty: float = field(default=1e-3)
    dcag_feature_token_dropout: float = field(default=0.05)
    dcag_feature_noise_std: float = field(default=0.01)
    dcag_feature_regularize_generator_only: bool = field(default=True)

    dcag_projector_preservation_multiplier: float = field(default=3.0)
    dcag_projector_warmup_steps: int = field(default=500)
    dcag_projector_cr_lr_multiplier: float = field(default=0.5)
    dcag_projector_disable_dynamic_B: bool = field(default=True)

    dcag_dynamic_current_B: bool = field(default=False)

    dcag_domain_image_resize_mode: str = field(default="squash")

    dcag_alignment_coeff: float = field(default=0.0)

    dcag_instruction_conditioning: bool = field(default=False)
    dcag_instruction_feature_dim: int = field(default=4096)

    dcag_attn_pool_queries: int = field(default=1)

    dcag_residual_rank: int = field(default=0)
    dcag_residual_lr: float = field(default=2e-4)
    dcag_residual_alpha: Optional[float] = field(default=None)

    dcag_task_idx: Optional[int] = field(default=None)

    dcag_B_mode: str = field(default="column_partitioned_CR")
    dcag_R_retraction: str = field(default="qr")
    dcag_Wd_mode: str = field(default="fisher_oja_per_slot")
    dcag_slot_weight_mode: str = field(default="projector_multiplier")
    dcag_projector_mode: str = field(default="integrated_with_policy")
    dcag_generator_arch: str = field(default="crossattn_adaln")
    dcag_domain_feature_mode: str = field(default="all_layers_crossattn")
    dcag_preservation_target: str = field(default="UV_factor_mse")

    dcag_generator_cache_mode: str = field(default="global_all_slots")

    dcag_b_composition_mode: str = field(default="full")
    dcag_domain_id_source: str = field(default="oracle")

    dcag_new_slice_suppression_coeff: float = field(default=0.0)

    dcag_functional_preservation_coeff: float = field(default=1.0)

    dcag_rolling_function_preservation: bool = field(default=True)
    dcag_rolling_start_task_idx: int = field(default=4)

    dcag_shared_projection_snapshot_enabled: bool = field(default=True)
    dcag_hidden_state_consistency_enabled: bool = field(default=True)
    dcag_rope_layer_enabled: bool = field(default=True)
    dcag_domain_embed_enabled: bool = field(default=True)
    dcag_spectral_balance_coeff: float = field(default=1e-4)
    dcag_spectral_balance_every_n_steps: int = field(default=50)

    dcag_preservation_trunk_coeff: float = field(default=3.0)
    dcag_preservation_uv_coeff: float = field(default=1.0)
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    is_SAT: bool = field(
        default=False,
        metadata={"help": "whether SAT data (different group_by_modality_length)"},
    )


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collect model state and save it to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        keys_to_match = ["mm_projector"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize token embeddings and initialize new rows from the existing mean."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):

    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_llama_2(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = torch.stack(
        [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ],
        dim=0,
    )
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(
                tokenizer_image_token(conv.sep, tokenizer)
            )
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:

    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)

    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """Format and tokenize conversations, masking user tokens in the training labels."""
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.PLAIN
    ):
        return preprocess_plain(sources, tokenizer)
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer)

    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
            for prompt in conversations
        ]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn(
                [header] + [s["value"] for s in source], tokenizer
            )["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        if data_args.memory_data_path is not None:
            rank0_print("Adding memory data... {}".format(data_args.memory_data_path))
            list_memory_data_dict = json.load(open(data_args.memory_data_path, "r"))

            list_data_dict = list_data_dict + list_memory_data_dict

            random.shuffle(list_data_dict)

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    @property
    def modality_lengths_for_SAT(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = cur_len if isinstance(sample["image"], list) else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"

        def pil_to_unit_tensor(pil_img):
            return (
                torch.from_numpy(np.array(pil_img, dtype=np.float32)).permute(2, 0, 1)
                / 255.0
            )

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor

            if isinstance(image_file, list):
                pil_images = [
                    Image.open(os.path.join(image_folder, img_file)).convert("RGB")
                    for img_file in image_file
                ]
            else:
                pil_images = Image.open(os.path.join(image_folder, image_file)).convert(
                    "RGB"
                )

            if self.data_args.image_aspect_ratio == "pad":

                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(
                            pil_img.mode, (width, width), background_color
                        )
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(
                            pil_img.mode, (height, height), background_color
                        )
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result

                if isinstance(pil_images, list):
                    processed_images = [
                        expand2square(
                            img, tuple(int(x * 255) for x in processor.image_mean)
                        )
                        for img in pil_images
                    ]
                    image = [
                        processor.preprocess(img, return_tensors="pt")["pixel_values"][
                            0
                        ]
                        for img in processed_images
                    ]
                else:
                    processed_image = expand2square(
                        pil_images, tuple(int(x * 255) for x in processor.image_mean)
                    )
                    try:
                        image = processor.preprocess(
                            processed_image, return_tensors="pt"
                        )["pixel_values"][0]
                    except Exception as e:
                        print("Wrong image", image_folder, image_file)
                        print(e)
                        raise e
            else:
                if isinstance(pil_images, list):
                    image = [
                        processor.preprocess(img, return_tensors="pt")["pixel_values"][
                            0
                        ]
                        for img in pil_images
                    ]
                else:
                    image = processor.preprocess(pil_images, return_tensors="pt")[
                        "pixel_values"
                    ][0]

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )

            if isinstance(pil_images, list):
                domain_image = torch.stack(
                    [pil_to_unit_tensor(img) for img in pil_images], dim=0
                )
            else:
                domain_image = pil_to_unit_tensor(pil_images)

                if should_split_ad_mosaic(image_file, self.data_args.image_folder, ""):
                    domain_image = apply_ad_split(domain_image)

            domain_image = domain_image.contiguous().float()
            if isinstance(image, list):
                image = [img.contiguous() for img in image]
            else:
                image = image.contiguous()
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources, self.tokenizer, has_image=("image" in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
            data_dict["domain_image"] = domain_image
        elif self.data_args.is_multimodal:
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])
            data_dict["domain_image"] = data_dict["image"].clone()

        domain_name = (
            self.data_args.domain_name
            or self.list_data_dict[i].get("domain")
            or self.list_data_dict[i].get("domain_name")
        )
        if domain_name is None:
            image_folder = self.data_args.image_folder or ""
            for candidate in DCAG_DOMAIN_TO_ID:
                if candidate.lower() in image_folder.lower():
                    domain_name = candidate
                    break
        if domain_name is None:
            raise KeyError("Unable to determine domain for sample")
        data_dict["domain_id"] = torch.tensor(
            DCAG_DOMAIN_TO_ID[domain_name], dtype=torch.long
        )
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            if isinstance(images[0], list):
                images = torch.stack([torch.stack(img, dim=0) for img in images], dim=0)
                batch["images"] = images
            else:
                if all(x is not None and x.shape == images[0].shape for x in images):
                    batch["images"] = torch.stack(images)
                else:
                    batch["images"] = images

        if "domain_image" in instances[0]:
            domain_images = [instance["domain_image"] for instance in instances]
            if all(
                x is not None and x.shape == domain_images[0].shape
                for x in domain_images
            ):
                batch["domain_images"] = torch.stack(domain_images)
            else:
                batch["domain_images"] = domain_images

        if "domain_id" in instances[0]:
            batch["domain_ids"] = torch.stack(
                [instance["domain_id"] for instance in instances]
            )

        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def _resize_dcag_variable_buffers(model, state_dict):
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


def load_model_from_previous_task(model, previous_task_model_path):

    print("Loading additional LLaVA weights...")
    non_lora_path = os.path.join(previous_task_model_path, "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        raise FileNotFoundError(
            f"Expected local previous-task weights at {non_lora_path}"
        )
    non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
    non_lora_trainables = {
        (k[11:] if k.startswith("base_model.") else k): v
        for k, v in non_lora_trainables.items()
    }
    if any(k.startswith("model.model.") for k in non_lora_trainables):
        non_lora_trainables = {
            (k[6:] if k.startswith("model.") else k): v
            for k, v in non_lora_trainables.items()
        }

    if getattr(model.config, "dcag", None):
        extra_task_slice_map_key = (
            "dcag_controller.task_slice_manager._extra_state_task_slice_map"
        )
        extra_task_slice_map_val = non_lora_trainables.pop(
            extra_task_slice_map_key, None
        )
        _resize_dcag_variable_buffers(model, non_lora_trainables)

        model_shapes = {
            k: tuple(v.shape)
            for k, v in model.state_dict().items()
            if isinstance(v, torch.Tensor)
        }
        dropped_shape_mismatch = []
        for k in list(non_lora_trainables.keys()):
            v = non_lora_trainables[k]
            tgt = model_shapes.get(k)
            if (
                tgt is not None
                and isinstance(v, torch.Tensor)
                and tuple(v.shape) != tgt
            ):
                dropped_shape_mismatch.append((k, tuple(v.shape), tgt))
                non_lora_trainables.pop(k)
        if dropped_shape_mismatch:
            print(
                f"[DCAG][load] dropping {len(dropped_shape_mismatch)} shape-mismatched "
                "checkpoint key(s) (kept current-model init; expected when a domain "
                "backbone hidden size changed):",
                flush=True,
            )
            for k, src_shape, tgt_shape in dropped_shape_mismatch:
                print(f"  - {k}: ckpt{src_shape} -> model{tgt_shape}", flush=True)
        missing, unexpected = model.load_state_dict(non_lora_trainables, strict=False)

        if (
            extra_task_slice_map_val is not None
            and getattr(model, "dcag_controller", None) is not None
        ):
            try:
                raw_bytes = bytes(extra_task_slice_map_val.tolist())
                payload = json.loads(raw_bytes.decode("utf-8"))

                if "task_slice_map" not in payload:
                    payload = {"task_slice_map": payload}
                model.dcag_controller.task_slice_manager.set_extra_state(payload)
                print(
                    f"DCAG task_slice_map restored: "
                    f"{sorted(model.dcag_controller.task_slice_manager.task_slice_map.keys())} "
                    f"domain_to_task={model.dcag_controller.task_slice_manager.domain_to_task}"
                )
            except Exception as exc:
                print(f"DCAG task_slice_map restore warning: {exc}")

        def _is_expected_absence(key: str) -> bool:
            if key.endswith(".inv_freq"):
                return True
            if key.endswith(".base_layer.weight") or key.endswith(".base_layer.bias"):
                return True
            if key in (
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
            ):
                return True
            if key.endswith(".input_layernorm.weight") or key.endswith(
                ".post_attention_layernorm.weight"
            ):
                return True
            if ".vision_tower." in key or ".mm_projector." in key:
                return True
            if ".feature_encoder.pipelines." in key and ".extractor.extractor." in key:
                return True

            if "dcag_controller.task_slice_manager." in key and key.endswith(
                "_current"
            ):
                return True

            if "dcag_controller.residual_lora_manager." in key and key.endswith(
                "_current"
            ):
                return True

            if key.endswith("._extra_state"):
                return True
            return False

        filtered_missing = [k for k in missing if not _is_expected_absence(k)]
        filtered_unexpected = [k for k in unexpected if not _is_expected_absence(k)]
        if filtered_missing or filtered_unexpected:
            print(
                f"DCAG previous-task load missing={len(filtered_missing)} unexpected={len(filtered_unexpected)}"
            )
            if filtered_missing:
                print("DCAG previous-task missing sample:", filtered_missing[:20])
            if filtered_unexpected:
                print("DCAG previous-task unexpected sample:", filtered_unexpected[:20])
        print("DCAG model is loaded...")
    else:
        model.base_model.model.load_state_dict(non_lora_trainables, strict=False)
        print("Loading LoRA weights...")
        filename = os.path.join(previous_task_model_path, WEIGHTS_NAME)
        adapters_weights = torch.load(
            filename,
            map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        set_peft_model_state_dict(model, adapters_weights, adapter_name="default")
        print("Model is loaded...")


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    if training_args.gradient_checkpointing:
        raise ValueError(
            "DCAG does not support HF gradient_checkpointing=True. "
            "To enable memory savings, set "
            "dcag_generator_cache_mode='layer_recompute' in the train "
            "config and keep --gradient_checkpointing False; DCAG will "
            "wrap each decoder layer with non-reentrant checkpointing "
            "internally. Reentrant checkpointing fails on the shared "
            "current_features.grad_fn across decoder layer regions."
        )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,
                ),
            )
        )

    rank0_print("[DCAG][startup] begin model load")
    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            **bnb_model_from_pretrained_args,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            **bnb_model_from_pretrained_args,
        )
    rank0_print("[DCAG][startup] model load done")
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    needs_input_require_grad = (
        training_args.gradient_checkpointing
        or training_args.dcag_generator_cache_mode == "layer_recompute"
    )
    if needs_input_require_grad:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.dcag_enable:
        rank0_print("[DCAG][startup] configuring DCAG")
        feature_layers = [
            int(item) for item in training_args.dcag_feature_layers.split(",") if item
        ]
        model.config.dcag = {
            "enabled": True,
            "feature_dim": training_args.dcag_feature_dim,
            "alpha": training_args.dcag_alpha,
            "generator_hidden_dim": training_args.dcag_generator_hidden_dim,
            "basis_lr": training_args.dcag_basis_lr,
            "generator_lr": training_args.dcag_generator_lr,
            "anti_forgetting_weight": training_args.dcag_anti_forgetting_weight,
            "anti_forgetting_weight_per_domain": parse_optional_sequence_arg(
                training_args.dcag_anti_forgetting_weight_per_domain, float
            ),
            "feature_layers": feature_layers,
            "train_domain_backbones": training_args.dcag_train_domain_backbones,
            "domain_model_device": training_args.dcag_domain_model_device,
            "domain_model_dtype": training_args.dcag_domain_model_dtype,
            "rs_ckpt_path": training_args.rs_domain_model_path,
            "rs_extractor_type": training_args.dcag_rs_extractor_type,
            "med_ckpt_path": training_args.med_domain_model_path,
            "med_extractor_type": training_args.dcag_med_extractor_type,
            "ad_ckpt_path": training_args.ad_domain_model_path,
            "ad_extractor_type": training_args.dcag_ad_extractor_type,
            "sci_model_dir": training_args.sci_domain_model_path,
            "sci_extractor_type": training_args.dcag_sci_extractor_type,
            "fin_model_dir": training_args.fin_domain_model_path,
            "fin_extractor_type": training_args.dcag_fin_extractor_type,
            "r": training_args.dcag_r,
            "h_d_llm": training_args.dcag_h_d_llm,
            "h_d_projector": training_args.dcag_h_d_projector,
            "k_d": training_args.dcag_k_d,
            "r_fisher": training_args.dcag_r_fisher,
            "num_centroids": training_args.dcag_num_centroids,
            "num_centroids_per_domain": parse_optional_sequence_arg(
                training_args.dcag_num_centroids_per_domain, int
            ),
            "generator_layers": training_args.dcag_generator_layers,
            "generator_heads": training_args.dcag_generator_heads,
            "factored_head_dim": training_args.dcag_factored_head_dim,
            "generator_dropout": training_args.dcag_generator_dropout,
            "a_l2_penalty": training_args.dcag_a_l2_penalty,
            "feature_token_dropout": training_args.dcag_feature_token_dropout,
            "feature_noise_std": training_args.dcag_feature_noise_std,
            "feature_regularize_generator_only": training_args.dcag_feature_regularize_generator_only,
            "projector_preservation_multiplier": training_args.dcag_projector_preservation_multiplier,
            "projector_warmup_steps": training_args.dcag_projector_warmup_steps,
            "projector_cr_lr_multiplier": training_args.dcag_projector_cr_lr_multiplier,
            "projector_disable_dynamic_B": training_args.dcag_projector_disable_dynamic_B,
            "dynamic_current_B": training_args.dcag_dynamic_current_B,
            "domain_image_resize_mode": training_args.dcag_domain_image_resize_mode,
            "alignment_coeff": training_args.dcag_alignment_coeff,
            "instruction_conditioning": training_args.dcag_instruction_conditioning,
            "instruction_feature_dim": training_args.dcag_instruction_feature_dim,
            "attn_pool_queries": training_args.dcag_attn_pool_queries,
            "residual_rank": training_args.dcag_residual_rank,
            "residual_lr": training_args.dcag_residual_lr,
            "residual_alpha": training_args.dcag_residual_alpha,
            "task_idx": training_args.dcag_task_idx,
            "B_mode": training_args.dcag_B_mode,
            "R_retraction": training_args.dcag_R_retraction,
            "Wd_mode": training_args.dcag_Wd_mode,
            "slot_weight_mode": training_args.dcag_slot_weight_mode,
            "projector_mode": training_args.dcag_projector_mode,
            "generator_arch": training_args.dcag_generator_arch,
            "domain_feature_mode": training_args.dcag_domain_feature_mode,
            "preservation_target": training_args.dcag_preservation_target,
            "generator_cache_mode": training_args.dcag_generator_cache_mode,
            "b_composition_mode": training_args.dcag_b_composition_mode,
            "domain_id_source": training_args.dcag_domain_id_source,
            "new_slice_suppression_coeff": training_args.dcag_new_slice_suppression_coeff,
            "functional_preservation_coeff": training_args.dcag_functional_preservation_coeff,
            "rolling_function_preservation": training_args.dcag_rolling_function_preservation,
            "rolling_start_task_idx": training_args.dcag_rolling_start_task_idx,
            "shared_projection_snapshot_enabled": training_args.dcag_shared_projection_snapshot_enabled,
            "hidden_state_consistency_enabled": training_args.dcag_hidden_state_consistency_enabled,
            "rope_layer_enabled": training_args.dcag_rope_layer_enabled,
            "domain_embed_enabled": training_args.dcag_domain_embed_enabled,
            "spectral_balance_coeff": training_args.dcag_spectral_balance_coeff,
            "spectral_balance_every_n_steps": training_args.dcag_spectral_balance_every_n_steps,
            "preservation_trunk_coeff": training_args.dcag_preservation_trunk_coeff,
            "preservation_uv_coeff": training_args.dcag_preservation_uv_coeff,
        }
        normalize_dcag_config_paths(model.config.dcag)

        if getattr(training_args, "local_rank", -1) in (0, -1):
            print(f"[DCAG][config] runtime dcag={model.config.dcag}")
        if not getattr(model, "has_dcag", lambda: False)():
            model.initialize_dcag(model.config.dcag)
        if getattr(model, "dcag_controller", None) is not None:
            model.dcag_controller.feature_encoder.train_domain_backbones = (
                training_args.dcag_train_domain_backbones
            )
        rank0_print("[DCAG][startup] DCAG ready")

        for p in model.parameters():
            p.requires_grad = False
        for name, p in model.named_parameters():
            if "dcag_controller.generator." in name:
                p.requires_grad = True
            elif "dcag_controller.feature_encoder.shared_projection." in name:
                p.requires_grad = True
            elif (
                "dcag_controller.feature_encoder.pipelines." in name
                and ".aggregator." in name
            ):
                p.requires_grad = True

            elif "dcag_controller.task_slice_manager." in name and "_current" in name:
                p.requires_grad = True
            elif "dcag_controller.task_slice_manager.plain_B." in name:
                p.requires_grad = True

            elif "dcag_controller.instruction_" in name:
                p.requires_grad = True

            elif "dcag_controller.residual_lora_manager." in name and name.endswith(
                "_current"
            ):
                p.requires_grad = True

        if training_args.dcag_projector_mode == "lora_ft_style":
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        if training_args.dcag_task_idx is not None:
            current_task_idx = int(training_args.dcag_task_idx)
        elif model_args.previous_task_model_path is None:
            current_task_idx = 0
        else:
            current_task_idx = None
        model.config._dcag_current_task_idx = current_task_idx

        trainable_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        rank0_print(
            f"[DCAG][freeze] trainable tensors={trainable_count}, "
            f"total params={trainable_numel / 1e6:.2f} M"
        )
    elif training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        local_files_only=True,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        if hasattr(model, "inject_dcag_projector_if_ready"):
            model.inject_dcag_projector_if_ready()

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
            model_args.tune_mm_mlp_adapter
        )
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if model_args.previous_task_model_path is not None:
        load_model_from_previous_task(model, model_args.previous_task_model_path)

    if (
        training_args.dcag_enable
        and getattr(model, "dcag_controller", None) is not None
    ):
        current_task_idx = getattr(model.config, "_dcag_current_task_idx", None)
        if current_task_idx is None:
            current_task_idx = model.dcag_controller.update_task_idx_from_state()
        rank0_print(f"[DCAG][task] starting task_idx={current_task_idx}")
        if data_args.domain_name not in DCAG_DOMAIN_TO_ID:
            raise ValueError("DCAG training requires a valid --domain_name")
        model.start_dcag_task(
            int(current_task_idx), domain_id=DCAG_DOMAIN_TO_ID[data_args.domain_name]
        )
        model.dcag_controller.write_dcag_config(training_args.output_dir)

    rank0_print("[DCAG][startup] begin dataset build")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    rank0_print("[DCAG][startup] dataset build done")
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    rank0_print("[DCAG][startup] trainer build done")

    trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.dcag_enable:
        if hasattr(model, "snapshot_dcag_coefficients"):
            rank0_print(
                "[DCAG][save] snapshotting coefficients and extracting W_d for next-task preservation"
            )
            model.snapshot_dcag_coefficients()

        if training_args.local_rank == 0 or training_args.local_rank == -1:
            state_dict = {
                k: maybe_zero_3(v, ignore_status=True)
                for k, v in model.named_parameters()
                if v.requires_grad
            }
            buffer_state = {}
            for module_name, module in model.named_modules():
                non_persist = getattr(module, "_non_persistent_buffers_set", set())
                for buf_name, buf in module._buffers.items():
                    if buf is None or buf_name in non_persist:
                        continue
                    full_name = f"{module_name}.{buf_name}" if module_name else buf_name
                    if "dcag_controller" in full_name:
                        buffer_state[full_name] = buf.detach().cpu().clone()

            try:
                extra_state = model.dcag_controller.task_slice_manager.get_extra_state()
                buffer_state[
                    "dcag_controller.task_slice_manager._extra_state_task_slice_map"
                ] = torch.tensor(
                    bytearray(json.dumps(extra_state).encode("utf-8")),
                    dtype=torch.uint8,
                )
            except Exception as exc:
                rank0_print(
                    f"[DCAG][save] warning: could not serialize task_slice_map: {exc}"
                )
            combined_state = {**state_dict, **buffer_state}
            rank0_print(
                f"[DCAG][save] trainable params: {len(state_dict)} tensors, "
                f"DCAG buffers: {len(buffer_state)} tensors"
            )
            model.config.save_pretrained(training_args.output_dir)

            _final_path = os.path.join(
                training_args.output_dir, "non_lora_trainables.bin"
            )
            _tmp_path = _final_path + ".tmp"
            torch.save(combined_state, _tmp_path)
            with open(_tmp_path, "rb") as _fh:
                os.fsync(_fh.fileno())
            os.replace(_tmp_path, _final_path)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    elif training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)

            _final_path_lora = os.path.join(
                training_args.output_dir, "non_lora_trainables.bin"
            )
            _tmp_path_lora = _final_path_lora + ".tmp"
            torch.save(non_lora_state_dict, _tmp_path_lora)
            with open(_tmp_path_lora, "rb") as _fh:
                os.fsync(_fh.fileno())
            os.replace(_tmp_path_lora, _final_path_lora)
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )

    remove_dir = training_args.output_dir
    subprocess.run(
        f"find {remove_dir} -maxdepth 1 -type d -name 'checkpoint-*' -exec rm -rf {{}} +",
        shell=True,
    )


if __name__ == "__main__":
    train()
