# Copyright The KiLoRA Authors
# SPDX-License-Identifier: Apache-2.0
from contextlib import contextmanager

import torch


class LayoutProcessor:
    def __init__(self, original):
        self.original = original
        self.mode = None
        self.maps = []
        self.index = 0

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        **kwargs,
    ):
        if self.mode is None or encoder_hidden_states is not None:
            return self.original(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
                **kwargs,
            )
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query, key, value = (
            x.unflatten(2, (attn.heads, -1)).transpose(1, 2) for x in (query, key, value)
        )
        if getattr(attn, "norm_q", None) is not None:
            query = attn.norm_q(query)
        if getattr(attn, "norm_k", None) is not None:
            key = attn.norm_k(key)
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(
                query,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-2,
            )
            key = apply_rotary_emb(
                key,
                image_rotary_emb,
                use_real=True,
                use_real_unbind_dim=-2,
            )
        key = key.repeat_interleave(query.shape[-1] // key.shape[-1], dim=-1)
        value = value.repeat_interleave(query.shape[-1] // value.shape[-1], dim=-1)
        if self.mode == "record":
            scores = query.float() @ key.float().transpose(-1, -2)
            scores *= query.shape[-1] ** -0.5
            if attention_mask is not None:
                if attention_mask.dtype == torch.bool:
                    scores.masked_fill_(~attention_mask, float("-inf"))
                else:
                    scores += attention_mask.float()
            probabilities = scores.softmax(-1).nan_to_num()
            self.maps.append(probabilities.detach().cpu())
        else:
            if self.index >= len(self.maps):
                raise ValueError("target self-attention has no matching student map")
            probabilities = self.maps[self.index].to(value.device)
            self.index += 1
            if probabilities.shape != (*query.shape[:-1], key.shape[-2]):
                raise ValueError("target and student self-attention shapes differ")
        output = (probabilities @ value.float()).to(query.dtype)
        output = output.transpose(1, 2).reshape(
            hidden_states.shape[0],
            -1,
            query.shape[1] * value.shape[-1],
        )
        for layer in attn.to_out:
            output = layer(output)
        return output


class PlainLayout:
    def __init__(self, model):
        self.processors = []
        for name, module in model.named_modules():
            if name.startswith("transformer_blocks.") and name.endswith(".attn1"):
                if not hasattr(module, "processor") or not hasattr(module, "set_processor"):
                    raise ValueError("plain layout requires self-attention processor support")
                self.processors.append((module, LayoutProcessor(module.processor)))
        if not self.processors:
            raise ValueError("plain layout requires image self-attention blocks")

    def install(self):
        for module, processor in self.processors:
            module.set_processor(processor)

    @contextmanager
    def active(self, mode):
        for _, processor in self.processors:
            processor.mode = mode
            processor.index = 0
            if mode == "record":
                processor.maps.clear()
        try:
            yield
        finally:
            for _, processor in self.processors:
                processor.mode = None
            if mode == "replay":
                self.clear()

    def clear(self):
        for _, processor in self.processors:
            processor.maps.clear()

    def close(self):
        for module, processor in self.processors:
            module.set_processor(processor.original)
        self.clear()
