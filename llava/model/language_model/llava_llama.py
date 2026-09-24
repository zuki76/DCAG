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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..dcag import DCAGModelMixin


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(DCAGModelMixin, LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

        self.dcag_enabled = False
        self.dcag_controller = None

    def get_model(self):
        return self.model

    @torch.no_grad()
    def _compute_instruction_features(self, input_ids, attention_mask, labels):
        from llava.constants import IMAGE_TOKEN_INDEX

        safe_ids = input_ids.clamp(min=0)
        emb = self.model.embed_tokens(safe_ids)
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        mask = mask & (input_ids != IMAGE_TOKEN_INDEX) & (input_ids >= 0)
        if labels is not None:
            mask = mask & (labels == -100)
        denom = mask.sum(dim=1).clamp(min=1)
        feat = (emb * mask.unsqueeze(-1).to(emb.dtype)).sum(dim=1) / denom.unsqueeze(-1)
        return feat

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        domain_ids: Optional[torch.LongTensor] = None,
        domain_images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        new_sample = past_key_values is None
        if new_sample:
            instruction_features = None
            if (
                self.dcag_controller is not None
                and getattr(self.dcag_controller, "instruction_conditioning", False)
                and input_ids is not None
            ):
                instruction_features = self._compute_instruction_features(
                    input_ids, attention_mask, labels
                )
            self.set_dcag_context(
                domain_images if domain_images is not None else images,
                domain_ids,
                instruction_features=instruction_features,
            )
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        preservation_loss = self.get_dcag_preservation_loss()

        a_l2 = self.get_dcag_a_l2_penalty() if self.training else None
        a_l2_coef = float(
            getattr(self.config, "dcag", {}).get("a_l2_penalty", 0.0) or 0.0
        )

        spectral_balance = (
            self.get_dcag_spectral_balance_penalty() if self.training else None
        )

        new_slice_suppression = (
            self.get_dcag_new_slice_suppression_loss() if self.training else None
        )
        new_slice_suppression_coef = float(
            getattr(self.config, "dcag", {}).get("new_slice_suppression_coeff", 0.0)
            or 0.0
        )

        functional_preservation = (
            self.get_dcag_functional_preservation_loss() if self.training else None
        )
        functional_preservation_coef = float(
            getattr(self.config, "dcag", {}).get("functional_preservation_coeff", 0.0)
            or 0.0
        )

        alignment_teacher = None
        alignment_coef = float(
            getattr(self.config, "dcag", {}).get("alignment_coeff", 0.0) or 0.0
        )
        if (
            self.training
            and alignment_coef > 0.0
            and isinstance(images, torch.Tensor)
            and images.dim() == 4
        ):
            with torch.no_grad():
                vision_tower = self.get_model().get_vision_tower()
                alignment_teacher = vision_tower(images).mean(dim=1)
        alignment_loss = (
            self.get_dcag_alignment_loss(alignment_teacher) if self.training else None
        )

        anti_forgetting_weight = float(
            getattr(self.config, "dcag", {}).get("anti_forgetting_weight", 0.0) or 0.0
        )
        afw_per_domain = getattr(self.config, "dcag", {}).get(
            "anti_forgetting_weight_per_domain", None
        )
        use_per_domain_afw = afw_per_domain is not None
        preservation_outer_weight = (
            1.0 if use_per_domain_afw else anti_forgetting_weight
        )
        if getattr(outputs, "loss", None) is not None:
            self._dcag_last_task_loss = float(outputs.loss.detach())
        else:
            self._dcag_last_task_loss = None
        if preservation_loss is not None:
            self._dcag_last_preservation_loss = float(preservation_loss.detach())
        else:
            self._dcag_last_preservation_loss = 0.0

        ctrl = getattr(self, "dcag_controller", None)
        trunk_l = getattr(ctrl, "latest_trunk_loss", None) if ctrl is not None else None
        uv_l = getattr(ctrl, "latest_uv_loss", None) if ctrl is not None else None
        self._dcag_last_trunk_loss = float(trunk_l) if trunk_l is not None else 0.0
        self._dcag_last_uv_loss = float(uv_l) if uv_l is not None else 0.0
        self._dcag_last_anti_forgetting_weight = preservation_outer_weight
        if a_l2 is not None:
            self._dcag_last_a_l2_loss = float(a_l2.detach())
        else:
            self._dcag_last_a_l2_loss = 0.0
        self._dcag_last_a_l2_coef = a_l2_coef

        if spectral_balance is not None:
            self._dcag_last_spectral_balance_loss = float(spectral_balance.detach())
        else:
            self._dcag_last_spectral_balance_loss = 0.0
        if new_slice_suppression is not None:
            self._dcag_last_new_slice_suppression_loss = float(
                new_slice_suppression.detach()
            )
        else:
            self._dcag_last_new_slice_suppression_loss = 0.0
        self._dcag_last_new_slice_suppression_coef = new_slice_suppression_coef
        if functional_preservation is not None:
            self._dcag_last_functional_preservation_loss = float(
                functional_preservation.detach()
            )
        else:
            self._dcag_last_functional_preservation_loss = 0.0
        self._dcag_last_functional_preservation_coef = functional_preservation_coef
        f_llm = (
            getattr(ctrl, "latest_functional_preservation_llm_loss", None)
            if ctrl is not None
            else None
        )
        f_proj = (
            getattr(ctrl, "latest_functional_preservation_projector_loss", None)
            if ctrl is not None
            else None
        )
        self._dcag_last_functional_preservation_llm_loss = (
            float(f_llm) if f_llm is not None else 0.0
        )
        self._dcag_last_functional_preservation_projector_loss = (
            float(f_proj) if f_proj is not None else 0.0
        )
        if alignment_loss is not None:
            self._dcag_last_alignment_loss = float(alignment_loss.detach())
        else:
            self._dcag_last_alignment_loss = 0.0
        self._dcag_last_alignment_coef = alignment_coef
        if (
            preservation_loss is not None
            and (anti_forgetting_weight > 0 or use_per_domain_afw)
            and getattr(outputs, "loss", None) is not None
        ):
            outputs.loss = outputs.loss + preservation_loss * preservation_outer_weight
        if (
            a_l2 is not None
            and a_l2_coef > 0
            and getattr(outputs, "loss", None) is not None
        ):
            outputs.loss = outputs.loss + a_l2 * a_l2_coef

        if spectral_balance is not None and getattr(outputs, "loss", None) is not None:
            outputs.loss = outputs.loss + spectral_balance
        if (
            new_slice_suppression is not None
            and new_slice_suppression_coef > 0
            and getattr(outputs, "loss", None) is not None
        ):
            outputs.loss = (
                outputs.loss + new_slice_suppression * new_slice_suppression_coef
            )
        if (
            functional_preservation is not None
            and functional_preservation_coef > 0
            and getattr(outputs, "loss", None) is not None
        ):
            outputs.loss = (
                outputs.loss + functional_preservation * functional_preservation_coef
            )
        if (
            alignment_loss is not None
            and alignment_coef > 0
            and getattr(outputs, "loss", None) is not None
        ):
            outputs.loss = outputs.loss + alignment_loss * alignment_coef
        if new_sample:
            self.finalize_dcag_batch()

        return outputs

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        domain_ids = kwargs.pop("domain_ids", None)
        domain_images = kwargs.pop("domain_images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            _inputs["images"] = images
        if domain_ids is not None:
            _inputs["domain_ids"] = domain_ids
        if domain_images is not None:
            _inputs["domain_images"] = domain_images
        return _inputs


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
