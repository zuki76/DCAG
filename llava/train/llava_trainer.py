import os
import torch
import torch.nn as nn

from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    ShardedDDPOption,
    logger,
)
from typing import List, Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True, name=k).cpu()
        for k, v in to_return.items()
    }
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """Split indices into chunks with roughly equal cumulative lengths."""

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(
    lengths, batch_size, world_size, generator=None
):

    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        return get_length_grouped_indices(
            lengths, batch_size, world_size, generator=generator
        )
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [
        mm_indices[i]
        for i in get_length_grouped_indices(
            mm_lengths, batch_size, world_size, generator=None
        )
    ]
    lang_shuffle = [
        lang_indices[i]
        for i in get_length_grouped_indices(
            lang_lengths, batch_size, world_size, generator=None
        )
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [
        mm_shuffle[i : i + megabatch_size]
        for i in range(0, len(mm_shuffle), megabatch_size)
    ]
    lang_megabatches = [
        lang_shuffle[i : i + megabatch_size]
        for i in range(0, len(lang_shuffle), megabatch_size)
    ]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(
    lengths, batch_size, world_size, generator=None, merge=True
):

    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [
        indices[i : i + megabatch_size].tolist()
        for i in range(0, len(lengths), megabatch_size)
    ]
    megabatches = [
        sorted(megabatch, key=lambda i: lengths[i], reverse=True)
        for megabatch in megabatches
    ]
    megabatches = [
        split_to_even_chunks(megabatch, lengths, world_size)
        for megabatch in megabatches
    ]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    """Group similar sequence lengths while randomizing sample order."""

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class LLaVATrainer(Trainer):
    @staticmethod
    def _ddp_avg_scalar(value, device):
        if (
            (not torch.distributed.is_available())
            or (not torch.distributed.is_initialized())
            or torch.distributed.get_world_size() <= 1
        ):
            return float(value)
        try:
            t = torch.tensor(float(value), device=device)
            torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.AVG)
            return float(t)
        except Exception:
            return float(value)

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            if self.args.is_SAT:
                lengths = self.train_dataset.modality_lengths_for_SAT
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """Create optimizer groups for generation, output slices, and domain residuals.

        Projector output slices use the configured learning-rate multiplier.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            projector_base_parameters = {
                name
                for name, _ in opt_model.named_parameters()
                if "mm_projector" in name and "dcag_controller" not in name
            }

            dcag_generator_parameters = {
                name
                for name, p in opt_model.named_parameters()
                if (
                    ("dcag_controller.generator." in name)
                    or ("dcag_controller.feature_encoder.shared_projection." in name)
                    or ("dcag_controller.alignment_head." in name)
                    or (
                        "dcag_controller.feature_encoder.pipelines." in name
                        and ".aggregator." in name
                    )
                )
                and p.requires_grad
            }

            import re as _re

            _slot_re = _re.compile(
                r"task_slice_manager\.(?:R_slot_|C_slot_|plain_B\.slot_)(\d+)"
            )
            NUM_LLM_SLOTS_LOCAL = 224

            def _is_llm_cr(n: str) -> bool:
                if "dcag_controller.task_slice_manager." not in n:
                    return False
                if "_current" not in n and ".plain_B.slot_" not in n:
                    return False
                m = _slot_re.search(n)
                if m is None:
                    return False
                return int(m.group(1)) < NUM_LLM_SLOTS_LOCAL

            def _is_projector_cr(n: str) -> bool:
                if "dcag_controller.task_slice_manager." not in n:
                    return False
                if "_current" not in n and ".plain_B.slot_" not in n:
                    return False
                m = _slot_re.search(n)
                if m is None:
                    return False
                return int(m.group(1)) >= NUM_LLM_SLOTS_LOCAL

            def _is_R_slot(n: str) -> bool:

                return "task_slice_manager.R_slot_" in n

            dcag_cr_llm_parameters = {
                name
                for name, p in opt_model.named_parameters()
                if _is_llm_cr(name) and p.requires_grad
            }
            dcag_cr_projector_parameters = {
                name
                for name, p in opt_model.named_parameters()
                if _is_projector_cr(name) and p.requires_grad
            }

            dcag_cr_llm_R_parameters = {
                n for n in dcag_cr_llm_parameters if _is_R_slot(n)
            }
            dcag_cr_llm_C_parameters = dcag_cr_llm_parameters - dcag_cr_llm_R_parameters
            dcag_cr_projector_R_parameters = {
                n for n in dcag_cr_projector_parameters if _is_R_slot(n)
            }
            dcag_cr_projector_C_parameters = (
                dcag_cr_projector_parameters - dcag_cr_projector_R_parameters
            )

            dcag_residual_parameters = {
                name
                for name, p in opt_model.named_parameters()
                if "dcag_controller.residual_lora_manager." in name and p.requires_grad
            }

            special_parameters = (
                projector_base_parameters
                | dcag_generator_parameters
                | dcag_cr_llm_parameters
                | dcag_cr_projector_parameters
                | dcag_residual_parameters
            )

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n in decay_parameters
                            and n not in special_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (
                            n not in decay_parameters
                            and n not in special_parameters
                            and p.requires_grad
                        )
                    ],
                    "weight_decay": 0.0,
                },
            ]

            if self.args.mm_projector_lr is not None:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in projector_base_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in projector_base_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                )

            if getattr(self.args, "dcag_generator_lr", None) is not None:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in dcag_generator_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.dcag_generator_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in dcag_generator_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.dcag_generator_lr,
                        },
                    ]
                )

            if getattr(self.args, "dcag_residual_lr", None) is not None:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in dcag_residual_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.dcag_residual_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in dcag_residual_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.dcag_residual_lr,
                        },
                    ]
                )

            if getattr(self.args, "dcag_basis_lr", None) is not None:
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (n in dcag_cr_llm_R_parameters and p.requires_grad)
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.dcag_basis_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in dcag_cr_llm_C_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.dcag_basis_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in dcag_cr_llm_C_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.dcag_basis_lr,
                        },
                    ]
                )

                projector_cr_lr = self.args.dcag_basis_lr * getattr(
                    self.args, "dcag_projector_cr_lr_multiplier", 0.5
                )
                optimizer_grouped_parameters.extend(
                    [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in dcag_cr_projector_R_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": projector_cr_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n in decay_parameters
                                    and n in dcag_cr_projector_C_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": projector_cr_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                    n not in decay_parameters
                                    and n in dcag_cr_projector_C_parameters
                                    and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": projector_cr_lr,
                        },
                    ]
                )

            optimizer_grouped_parameters = [
                group for group in optimizer_grouped_parameters if group["params"]
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                raise ValueError(
                    "Use the supplied DeepSpeed ZeRO-2 configuration for DCAG training"
                )
            else:
                self.optimizer = optimizer_cls(
                    optimizer_grouped_parameters, **optimizer_kwargs
                )
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum(
                                {
                                    p.data_ptr(): p.numel() for p in module.parameters()
                                }.values()
                            )
                            logger.info(f"skipped {module}: {skipped / 2**20}M params")
                            manager.register_module_override(
                                module, "weight", {"optim_bits": 32}
                            )
                            logger.debug(
                                f"bitsandbytes: will optimize {module} in fp32"
                            )
                    logger.info(f"skipped: {skipped / 2**20}M params")

        return self.optimizer

    def training_step(self, model, inputs):
        """Run backward, aggregate loss metrics, and maintain adapter state."""
        loss = super().training_step(model, inputs)

        task_loss = getattr(model, "_dcag_last_task_loss", None)
        if task_loss is not None:
            preservation_loss = getattr(model, "_dcag_last_preservation_loss", 0.0)
            weight = getattr(model, "_dcag_last_anti_forgetting_weight", 0.0)
            a_l2_loss = getattr(model, "_dcag_last_a_l2_loss", 0.0)
            a_l2_coef = getattr(model, "_dcag_last_a_l2_coef", 0.0)
            spectral_balance_loss = getattr(
                model, "_dcag_last_spectral_balance_loss", 0.0
            )
            new_slice_suppression_loss = getattr(
                model, "_dcag_last_new_slice_suppression_loss", 0.0
            )
            new_slice_suppression_coef = getattr(
                model, "_dcag_last_new_slice_suppression_coef", 0.0
            )
            functional_preservation_loss = getattr(
                model, "_dcag_last_functional_preservation_loss", 0.0
            )
            functional_preservation_coef = getattr(
                model, "_dcag_last_functional_preservation_coef", 0.0
            )
            functional_preservation_llm_loss = getattr(
                model, "_dcag_last_functional_preservation_llm_loss", 0.0
            )
            functional_preservation_projector_loss = getattr(
                model, "_dcag_last_functional_preservation_projector_loss", 0.0
            )
            alignment_loss = getattr(model, "_dcag_last_alignment_loss", 0.0)
            alignment_coef = getattr(model, "_dcag_last_alignment_coef", 0.0)

            trunk_loss = getattr(model, "_dcag_last_trunk_loss", 0.0)
            uv_loss = getattr(model, "_dcag_last_uv_loss", 0.0)

            device = getattr(self.args, "device", None) or torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            task_loss = self._ddp_avg_scalar(task_loss, device)
            preservation_loss = self._ddp_avg_scalar(preservation_loss or 0.0, device)
            a_l2_loss = self._ddp_avg_scalar(a_l2_loss or 0.0, device)
            spectral_balance_loss = self._ddp_avg_scalar(
                spectral_balance_loss or 0.0, device
            )
            new_slice_suppression_loss = self._ddp_avg_scalar(
                new_slice_suppression_loss or 0.0, device
            )
            functional_preservation_loss = self._ddp_avg_scalar(
                functional_preservation_loss or 0.0, device
            )
            functional_preservation_llm_loss = self._ddp_avg_scalar(
                functional_preservation_llm_loss or 0.0, device
            )
            functional_preservation_projector_loss = self._ddp_avg_scalar(
                functional_preservation_projector_loss or 0.0, device
            )
            trunk_loss = self._ddp_avg_scalar(trunk_loss or 0.0, device)
            uv_loss = self._ddp_avg_scalar(uv_loss or 0.0, device)
            alignment_loss = self._ddp_avg_scalar(alignment_loss or 0.0, device)
            grad_acc = max(int(self.args.gradient_accumulation_steps), 1)
            self._dcag_task_loss_sum = (
                getattr(self, "_dcag_task_loss_sum", 0.0) + task_loss / grad_acc
            )
            self._dcag_preservation_loss_sum = (
                getattr(self, "_dcag_preservation_loss_sum", 0.0)
                + preservation_loss / grad_acc
            )
            self._dcag_preservation_loss_weighted_sum = (
                getattr(self, "_dcag_preservation_loss_weighted_sum", 0.0)
                + preservation_loss * weight / grad_acc
            )
            self._dcag_a_l2_loss_sum = (
                getattr(self, "_dcag_a_l2_loss_sum", 0.0) + a_l2_loss / grad_acc
            )
            self._dcag_a_l2_loss_weighted_sum = (
                getattr(self, "_dcag_a_l2_loss_weighted_sum", 0.0)
                + a_l2_loss * a_l2_coef / grad_acc
            )

            self._dcag_spectral_balance_loss_sum = (
                getattr(self, "_dcag_spectral_balance_loss_sum", 0.0)
                + spectral_balance_loss / grad_acc
            )
            self._dcag_new_slice_suppression_loss_sum = (
                getattr(self, "_dcag_new_slice_suppression_loss_sum", 0.0)
                + new_slice_suppression_loss / grad_acc
            )
            self._dcag_new_slice_suppression_loss_weighted_sum = (
                getattr(self, "_dcag_new_slice_suppression_loss_weighted_sum", 0.0)
                + new_slice_suppression_loss * new_slice_suppression_coef / grad_acc
            )
            self._dcag_functional_preservation_loss_sum = (
                getattr(self, "_dcag_functional_preservation_loss_sum", 0.0)
                + functional_preservation_loss / grad_acc
            )
            self._dcag_functional_preservation_loss_weighted_sum = (
                getattr(self, "_dcag_functional_preservation_loss_weighted_sum", 0.0)
                + functional_preservation_loss * functional_preservation_coef / grad_acc
            )
            self._dcag_functional_preservation_llm_loss_sum = (
                getattr(self, "_dcag_functional_preservation_llm_loss_sum", 0.0)
                + functional_preservation_llm_loss / grad_acc
            )
            self._dcag_functional_preservation_projector_loss_sum = (
                getattr(self, "_dcag_functional_preservation_projector_loss_sum", 0.0)
                + functional_preservation_projector_loss / grad_acc
            )

            self._dcag_trunk_loss_sum = (
                getattr(self, "_dcag_trunk_loss_sum", 0.0) + trunk_loss / grad_acc
            )
            self._dcag_uv_loss_sum = (
                getattr(self, "_dcag_uv_loss_sum", 0.0) + uv_loss / grad_acc
            )

            self._dcag_alignment_loss_sum = (
                getattr(self, "_dcag_alignment_loss_sum", 0.0)
                + alignment_loss / grad_acc
            )
            self._dcag_alignment_loss_weighted_sum = (
                getattr(self, "_dcag_alignment_loss_weighted_sum", 0.0)
                + alignment_loss * alignment_coef / grad_acc
            )
            self._dcag_loss_step_count = getattr(self, "_dcag_loss_step_count", 0) + 1

        if (
            (self.state.global_step > self._last_retract_step)
            and hasattr(model, "dcag_controller")
            and model.dcag_controller is not None
        ):
            try:
                metrics = model.dcag_controller.task_slice_manager.retract_R_current()
                self._last_retract_step = self.state.global_step

                if metrics is not None:
                    max_off_diag, max_off_orth = metrics

                    device = getattr(self.args, "device", None) or torch.device(
                        "cuda" if torch.cuda.is_available() else "cpu"
                    )

                    self._dcag_R_off_diag_max = self._ddp_avg_scalar(
                        float(max_off_diag), device
                    )
                    self._dcag_R_off_orth_max = self._ddp_avg_scalar(
                        float(max_off_orth), device
                    )
                    if (
                        self.args.local_rank in (0, -1)
                        and self.state.global_step > 0
                        and self.state.global_step % 200 == 0
                    ):
                        print(
                            f"[DCAG][retract@{self.state.global_step}] "
                            f"max|R_old^T R_cur|={max_off_diag:.2e} "
                            f"max|R_cur^T R_cur - I|={max_off_orth:.2e}"
                        )
            except Exception as exc:
                if self.args.local_rank in (0, -1):
                    print(f"[DCAG][retract] warning: {exc}")

        if hasattr(model, "dcag_controller") and model.dcag_controller is not None:
            model.dcag_controller.finalize_after_backward()
        return loss

    def log(self, logs):
        """Add averaged component losses to the training metrics and reset accumulators."""
        n = getattr(self, "_dcag_loss_step_count", 0)
        if n > 0:
            logs["task_loss"] = round(self._dcag_task_loss_sum / n, 4)
            logs["preservation_loss"] = round(self._dcag_preservation_loss_sum / n, 4)
            logs["preservation_loss_weighted"] = round(
                self._dcag_preservation_loss_weighted_sum / n, 4
            )
            logs["a_l2_loss"] = round(self._dcag_a_l2_loss_sum / n, 6)
            logs["a_l2_loss_weighted"] = round(self._dcag_a_l2_loss_weighted_sum / n, 6)

            logs["spectral_balance_loss"] = round(
                getattr(self, "_dcag_spectral_balance_loss_sum", 0.0) / n, 6
            )
            logs["new_slice_suppression_loss"] = round(
                getattr(self, "_dcag_new_slice_suppression_loss_sum", 0.0) / n, 6
            )
            logs["new_slice_suppression_loss_weighted"] = round(
                getattr(self, "_dcag_new_slice_suppression_loss_weighted_sum", 0.0) / n,
                6,
            )
            logs["functional_preservation_loss"] = round(
                getattr(self, "_dcag_functional_preservation_loss_sum", 0.0) / n, 6
            )
            logs["functional_preservation_loss_weighted"] = round(
                getattr(self, "_dcag_functional_preservation_loss_weighted_sum", 0.0)
                / n,
                6,
            )
            logs["functional_preservation_llm_loss"] = round(
                getattr(self, "_dcag_functional_preservation_llm_loss_sum", 0.0) / n, 6
            )
            logs["functional_preservation_projector_loss"] = round(
                getattr(self, "_dcag_functional_preservation_projector_loss_sum", 0.0)
                / n,
                6,
            )

            logs["trunk_loss"] = round(
                getattr(self, "_dcag_trunk_loss_sum", 0.0) / n, 6
            )
            logs["uv_loss"] = round(getattr(self, "_dcag_uv_loss_sum", 0.0) / n, 6)

            logs["alignment_loss"] = round(
                getattr(self, "_dcag_alignment_loss_sum", 0.0) / n, 6
            )
            logs["alignment_loss_weighted"] = round(
                getattr(self, "_dcag_alignment_loss_weighted_sum", 0.0) / n, 6
            )
            self._dcag_task_loss_sum = 0.0
            self._dcag_preservation_loss_sum = 0.0
            self._dcag_preservation_loss_weighted_sum = 0.0
            self._dcag_a_l2_loss_sum = 0.0
            self._dcag_a_l2_loss_weighted_sum = 0.0
            self._dcag_spectral_balance_loss_sum = 0.0
            self._dcag_new_slice_suppression_loss_sum = 0.0
            self._dcag_new_slice_suppression_loss_weighted_sum = 0.0
            self._dcag_functional_preservation_loss_sum = 0.0
            self._dcag_functional_preservation_loss_weighted_sum = 0.0
            self._dcag_functional_preservation_llm_loss_sum = 0.0
            self._dcag_functional_preservation_projector_loss_sum = 0.0
            self._dcag_trunk_loss_sum = 0.0
            self._dcag_uv_loss_sum = 0.0
            self._dcag_alignment_loss_sum = 0.0
            self._dcag_alignment_loss_weighted_sum = 0.0
            self._dcag_loss_step_count = 0

        if hasattr(self, "_dcag_R_off_diag_max"):
            logs["R_off_diagonal_max"] = round(self._dcag_R_off_diag_max, 8)
            logs["R_diagonal_max"] = round(self._dcag_R_off_orth_max, 8)
        super().log(logs)

    @property
    def _last_retract_step(self) -> int:
        return getattr(self, "__last_retract_step", -1)

    @_last_retract_step.setter
    def _last_retract_step(self, value: int):
        self.__last_retract_step = value

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            keys_to_match = ["mm_projector", "vision_resampler"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(["embed_tokens", "embed_in"])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match
            )

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, "mm_projector.bin"))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, "tune_mm_mlp_adapter", False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
