import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor
from executorch.examples.models.model_base import EagerModelBase
from executorch.examples.models.llama.model_args import ModelArgs
from executorch.examples.models.llama.llama_transformer import construct_transformer
from executorch.examples.models.llama.source_transformation.custom_kv_cache import (
    replace_kv_cache_with_custom_kv_cache,
)
from executorch.examples.models.llama.source_transformation.sdpa import (
    replace_sdpa_with_custom_op,
)
import json
import os
import math
from typing import Optional, Tuple

IMAGE_SIZE = 512
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
FIXED_H, FIXED_W = 32, 32  # Adjusted for 256 tokens
MERGE_SIZE = 1  # Adjusted to maintain 256 tokens (16*16)
NUM_TOKENS = (FIXED_H // MERGE_SIZE) * (FIXED_W // MERGE_SIZE)  # 16 * 16 = 256


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class SimpleVisionAttention(nn.Module):
    def __init__(self, orig_attn):
        super().__init__()
        self.qkv = orig_attn.qkv
        self.proj = orig_attn.proj
        self.num_heads = orig_attn.num_heads
        self.scaling = orig_attn.scaling

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None, **kwargs):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3)
        query_states, key_states, value_states = qkv.unbind(0)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # B=1, H=num_heads, L=seq_len, D=head_dim
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, is_causal=False, scale=self.scaling
        )

        attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)


class SimpleVisionPatchEmbed(nn.Module):
    def __init__(self, patch_embed):
        super().__init__()
        self.patch_size = patch_embed.patch_size
        self.temporal_patch_size = patch_embed.temporal_patch_size
        self.in_channels = patch_embed.in_channels
        self.embed_dim = patch_embed.embed_dim

        self.proj = nn.Conv2d(
            in_channels=self.in_channels * self.temporal_patch_size,
            out_channels=self.embed_dim,
            kernel_size=(self.patch_size, self.patch_size),
            stride=(self.patch_size, self.patch_size),
            bias=True,
        )
        # Weight translation: [768, 3, 2, 16, 16] -> [768, 6, 16, 16]
        with torch.no_grad():
            w = patch_embed.proj.weight.data  # [768, 3, 2, 16, 16]
            w = w.permute(0, 2, 1, 3, 4).reshape(self.embed_dim, -1, self.patch_size, self.patch_size)
            self.proj.weight.copy_(w)
            self.proj.bias.copy_(patch_embed.proj.bias.data)

    def forward(self, x):
        # x is [1, 6, 512, 512]
        return self.proj(x).flatten(2).transpose(1, 2).reshape(-1, self.embed_dim)


class SimpleVisionWrapper(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual
        # Replace the entire patch_embed module
        self.visual.patch_embed = SimpleVisionPatchEmbed(self.visual.patch_embed)

        # Replace attention in blocks
        for block in self.visual.blocks:
            block.attn = SimpleVisionAttention(block.attn)

        # Precompute pos_embeds for 32x32
        num_grid_per_side = getattr(self.visual, "num_grid_per_side", 48)
        orig_pos_embed = self.visual.pos_embed.weight.data  # [2304, 768]
        dim = orig_pos_embed.shape[-1]

        h_idxs = torch.linspace(0, num_grid_per_side - 1, FIXED_H)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, FIXED_W)
        h_idxs_floor = h_idxs.long()
        w_idxs_floor = w_idxs.long()
        h_idxs_ceil = (h_idxs.long() + 1).clamp(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.long() + 1).clamp(max=num_grid_per_side - 1)
        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        # [h, w]
        p00 = orig_pos_embed[h_idxs_floor[:, None] * num_grid_per_side + w_idxs_floor[None, :]]
        p01 = orig_pos_embed[h_idxs_floor[:, None] * num_grid_per_side + w_idxs_ceil[None, :]]
        p10 = orig_pos_embed[h_idxs_ceil[:, None] * num_grid_per_side + w_idxs_floor[None, :]]
        p11 = orig_pos_embed[h_idxs_ceil[:, None] * num_grid_per_side + w_idxs_ceil[None, :]]

        w00 = (1 - dh[:, None]) * (1 - dw[None, :])
        w01 = (1 - dh[:, None]) * dw[None, :]
        w10 = dh[:, None] * (1 - dw[None, :])
        w11 = dh[:, None] * dw[None, :]

        res = p00 * w00[:, :, None] + p01 * w01[:, :, None] + p10 * w10[:, :, None] + p11 * w11[:, :, None]
        self.register_buffer("precomputed_pos_embed", res.reshape(-1, dim))

        # Precompute rot_pos_emb for 32x32
        grid_thw = torch.tensor([[1, 32, 32]])
        # Use original rot_pos_emb before patching
        with torch.no_grad():
            self.register_buffer("precomputed_rot_pos_emb", self.visual.rot_pos_emb(grid_thw))

        # Patch the methods
        self.visual.fast_pos_embed_interpolate = self.patched_interpolate
        self.visual.rot_pos_emb = self.patched_rot_pos_emb

    def patched_interpolate(self, grid_thw):
        return self.precomputed_pos_embed

    def patched_rot_pos_emb(self, grid_thw):
        return self.precomputed_rot_pos_emb

    def forward(self, nchw_pixels):
        # 1. Normalize
        x = nchw_pixels / 255.0
        x = (x - 0.5) / 0.5

        # 2. Duplicate temporal dimension (T=2) and concat in C
        x = torch.cat([x, x], dim=1)  # [1, 6, 512, 512]

        # Use dynamic grid_thw calculation
        grid_h = x.shape[-2] // 16
        grid_w = x.shape[-1] // 16
        grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.int64)

        out = self.visual(x, grid_thw)
        return out.pooler_output.view(1, -1, out.pooler_output.shape[-1])


class Qwen3_5(nn.Module):
    def __init__(self, hf_model, params):
        super().__init__()
        self.hf_model = hf_model
        self.text_model_args = params
        self.text_model = construct_transformer(params)

        if params.use_sdpa_with_kv_cache_op:
            self.text_model = replace_kv_cache_with_custom_kv_cache(self.text_model)
            self.text_model = replace_sdpa_with_custom_op(self.text_model)

        self.text_model.load_state_dict(self._translate_weights(), strict=False, assign=True)

    def _translate_weights(self):
        from executorch.examples.models.qwen3_5.convert_weights import qwen_3_5_to_meta

        raw = {}
        for k, v in self.hf_model.model.language_model.state_dict().items():
            raw[f"model.language_model.{k}"] = v
        # Add lm_head weights
        for k, v in self.hf_model.lm_head.state_dict().items():
            raw[f"lm_head.{k}"] = v
        return qwen_3_5_to_meta(raw)

    def forward(self, embeddings, input_pos):
        return self.text_model.forward(None, {"input_pos": input_pos}, embeddings)


class Qwen3_5MultimodalModel(EagerModelBase):
    def __init__(
        self, model_id="Qwen/Qwen3.5-0.8B", max_seq_len=1024, max_context_len=1024, use_sdpa_with_kv_cache_op=False
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.max_context_len = max_context_len
        self.use_sdpa_with_kv_cache_op = use_sdpa_with_kv_cache_op

        self.hf_model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.float32, device_map="cpu"
        )
        self.processor = AutoProcessor.from_pretrained(model_id)

        text_config = self.hf_model.config.text_config
        self.text_model_args = ModelArgs(
            dim=text_config.hidden_size,
            n_layers=text_config.num_hidden_layers,
            n_heads=text_config.num_attention_heads,
            n_kv_heads=text_config.num_key_value_heads,
            vocab_size=text_config.vocab_size,
            multiple_of=1,
            ffn_dim_multiplier=None,
            norm_eps=text_config.rms_norm_eps,
            max_batch_size=1,
            max_seq_len=max_seq_len,
            max_context_len=max_context_len,
            use_kv_cache=True,
            use_sdpa_with_kv_cache_op=use_sdpa_with_kv_cache_op,
            head_dim=text_config.head_dim,
            hidden_dim=text_config.intermediate_size,
            use_q_gate=True,
            rms_norm_add_unit_offset=True,
            use_qk_norm=True,
            # Hybrid model params
            layer_types=text_config.layer_types,
            linear_conv_kernel_dim=text_config.linear_conv_kernel_dim,
            linear_key_head_dim=text_config.linear_key_head_dim,
            linear_value_head_dim=text_config.linear_value_head_dim,
            linear_num_key_heads=text_config.linear_num_key_heads,
            linear_num_value_heads=text_config.linear_num_value_heads,
            partial_rotary_factor=text_config.partial_rotary_factor,
            rope_theta=text_config.rope_parameters.get("rope_theta"),
        )

    def get_eager_model(self):
        return Qwen3_5(self.hf_model, self.text_model_args)

    def get_example_inputs(self):
        return (torch.randint(0, 256, (1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32),)

    def get_inputs_for_prefill(self):
        pixel_values = self.get_example_inputs()[0]
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        return (input_ids, pixel_values, torch.tensor([[1, 32, 32]]))
