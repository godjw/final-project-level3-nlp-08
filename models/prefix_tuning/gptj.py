# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch OpenAI GPT-2 model."""

from typing import Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.gptj.modeling_gptj import (
    GPTJModel,
    GPTJPreTrainedModel,
    GPTJ_START_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    DEPARALLELIZE_DOCSTRING,
    GPTJ_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
)

NUM_P = 5


@add_start_docstrings(
    """
    The GPT-J Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPTJ_START_DOCSTRING,
)
class GPTJPrefixTuningForCausalLM(GPTJPreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"h\.\d+\.attn\.masked_bias",
        r"h\.\d+\.attn\.bias",
        r"lm_head\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTJModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.init_weights()

        self.transformer.h = self.transformer.h[:8]
        self.freeze_transformer()

        # layers for prefix tuning
        self.prefixes = torch.LongTensor(list(range(NUM_P))).unsqueeze(0).to("cuda")
        self.prefix_tkn_embd = nn.Embedding(num_embeddings=NUM_P, embedding_dim=config.n_embd)
        self.prefix_embd_synchronizer = nn.LSTM(
            input_size=config.n_embd,
            hidden_size=config.n_embd // 2,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        self.prefix_embd_fc = nn.Sequential(
            nn.Linear(in_features=config.n_embd, out_features=config.n_embd),
            nn.GELU(),
            nn.Linear(in_features=config.n_embd, out_features=config.n_embd),
        )

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def freeze_transformer(self):
        for param in self.transformer.parameters():
            param.requires_grad_(False)

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        return

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(GPTJ_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if past_key_values is None:
            prefix_embeddings = self.prefix_tkn_embd(self.prefixes)
            prefix_embeddings, _ = self.prefix_embd_synchronizer(prefix_embeddings)
            prefix_embeddings = self.prefix_embd_fc(prefix_embeddings)
            prefix_embeddings = torch.cat([prefix_embeddings] * input_ids.shape[0], dim=0)

            if input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            else:
                raise ValueError("You have to specify input_ids")

            inputs_embeds = self.transformer.wte(input_ids)
            input_ids = None

            inputs_embeds = torch.cat([prefix_embeddings, inputs_embeds], dim=-2)
        else:
            inputs_embeds = self.transformer.wte(input_ids)
            input_ids = None

        attention_mask = torch.cat([attention_mask[..., :1]] * NUM_P + [attention_mask], dim=-1)
        token_type_ids = torch.cat([token_type_ids[..., :1]] * NUM_P + [token_type_ids], dim=-1)

        if isinstance(position_ids, torch.Tensor):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PretrainedModel.beam_search` or :meth:`~transformers.PretrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )
