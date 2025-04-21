import torch
import json
from pathlib import Path
from torch import nn
import torch.nn.functional as F
from typing import List, Union, Optional
from safetensors import safe_open
from transformers.activations import ACT2FN
from transformers.models.mistral import MistralConfig
from opentelemetry import trace
from text_embeddings_server.models import Model
from text_embeddings_server.models.types import FlashBatch, Embedding, PaddedBatch
from text_embeddings_server.utils.flash_attn import attention

tracer = trace.get_tracer(__name__)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def load_weight(model_path, weight_map, name, dtype, device):
    """
    Helper function to load a weight tensor from safetensors.
    """
    target_file = weight_map[name]
    with safe_open(f"{model_path}/{target_file}", framework="pt") as f:
        return f.get_tensor(name).to(dtype).to(device)


def compute_default_rope_parameters(
    config: MistralConfig,
    device: torch.device,
) -> tuple["torch.Tensor", float]:
    base = config.rope_theta
    partial_rotary_factor = (
        config.partial_rotary_factor
        if hasattr(config, "partial_rotary_factor")
        else 1.0
    )
    head_dim = (
        getattr(config, "head_dim", None)
        or config.hidden_size // config.num_attention_heads
    )
    dim = int(head_dim * partial_rotary_factor)
    attention_factor = 1.0

    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(0, dim, 2, dtype=torch.int64).to(
                device=device, dtype=torch.float
            )
            / dim
        )
    )
    return inv_freq, attention_factor


class MistralRMSNorm:
    def __init__(
        self,
        model_path,
        weight_map,
        name,
        device,
        dtype,
        eps=1e-6,
    ):
        self.weight = load_weight(model_path, weight_map, name, dtype, device)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        if hidden_states.device.type == "hpu":
            from habana_frameworks.torch.hpex.normalization import (
                FusedRMSNorm as FusedRMSNorm,
            )

            hidden_states = FusedRMSNorm.apply(
                hidden_states, self.weight, self.variance_epsilon
            )
            return hidden_states
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(
                variance + self.variance_epsilon
            )
            return self.weight * hidden_states.to(input_dtype)


class MistralRotaryEmbedding(nn.Module):
    def __init__(self, config: MistralConfig, device=None):
        super().__init__()
        inv_freq, self.attention_scaling = compute_default_rope_parameters(
            config, device
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class MistralAttention:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: MistralConfig,
        layer_idx: Optional[int] = None,
    ):
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.softmax_scale = self.head_dim**-0.5
        self.q_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.q_proj.weight",
            dtype,
            device,
        )
        self.k_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.k_proj.weight",
            dtype,
            device,
        )
        self.v_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.v_proj.weight",
            dtype,
            device,
        )
        self.o_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.self_attn.o_proj.weight",
            dtype,
            device,
        )

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = F.linear(hidden_states, self.q_proj_weight).view(hidden_shape)
        k = F.linear(hidden_states, self.k_proj_weight).view(hidden_shape)
        v = F.linear(hidden_states, self.v_proj_weight).view(hidden_shape)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)
        attn_output = torch.empty_like(q)
        attention(
            q,
            k,
            v,
            attn_output,
            cu_seqlens,
            max_s,
            self.softmax_scale,
            is_causal=True,
            attn_mask=attn_mask,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = F.linear(attn_output, self.o_proj_weight, bias=None)

        return attn_output


class MistralMLP:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: MistralConfig,
        layer_idx: Optional[int] = None,
    ):
        self.gate_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.gate_proj.weight",
            dtype,
            device,
        )
        self.up_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.up_proj.weight",
            dtype,
            device,
        )
        self.down_proj_weight = load_weight(
            model_path,
            weight_map,
            f"layers.{layer_idx}.mlp.down_proj.weight",
            dtype,
            device,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        gated_hidden_states = F.linear(hidden_state, self.gate_proj_weight)
        uped_hidden_states = F.linear(hidden_state, self.up_proj_weight)
        return F.linear(
            self.act_fn(gated_hidden_states) * uped_hidden_states,
            self.down_proj_weight,
        )


class MistralDecoderLayer:
    def __init__(
        self,
        model_path,
        weight_map,
        device,
        dtype,
        config: MistralConfig,
        layer_idx: Optional[int] = None,
    ):
        self.attention = MistralAttention(
            model_path, weight_map, device, dtype, config, layer_idx
        )
        self.mlp = MistralMLP(model_path, weight_map, device, dtype, config, layer_idx)
        self.input_layernorm = MistralRMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.input_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = MistralRMSNorm(
            model_path,
            weight_map,
            f"layers.{layer_idx}.post_attention_layernorm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self, hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask=None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states)
        # Self Attention
        hidden_states = self.attention.forward(
            hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm.forward(hidden_states)
        hidden_states = self.mlp.forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class FlashMistralModel:
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MistralDecoderLayer`]

    Args:
        config: MistralConfig
    """

    def __init__(
        self, model_path, weight_map_json, device, dtype, config: MistralConfig
    ):
        self.word_embeddings_weight = load_weight(
            model_path,
            weight_map_json["weight_map"],
            "embed_tokens.weight",
            dtype,
            device,
        )
        self.layers = [
            MistralDecoderLayer(
                model_path,
                weight_map_json["weight_map"],
                device,
                dtype,
                config,
                layer_idx,
            )
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.rotary_emb = MistralRotaryEmbedding(config=config, device=device)
        self.norm = MistralRMSNorm(
            model_path,
            weight_map_json["weight_map"],
            f"norm.weight",
            device,
            dtype,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        input_ids,
        position_ids,
        cu_seqlens,
        max_s,
        mask=None,
        attn_mask=None,
    ):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer.forward(
                hidden_states, position_embeddings, cu_seqlens, max_s, attn_mask
            )
        hidden_states = self.norm.forward(hidden_states)
        if mask is not None:
            outputs = hidden_states[mask]
            return outputs[cu_seqlens[:-1]]
        return hidden_states[cu_seqlens[:-1]]


class FlashMistral(Model):
    def __init__(
        self, model_path: Path, device: torch.device, dtype: torch.dtype, pool: str
    ):
        config = MistralConfig.from_pretrained(model_path)

        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        with open(model_path / "model.safetensors.index.json", "r") as f:
            index_data = json.load(f)

        model = FlashMistralModel(model_path, index_data, device, dtype, config)
        self.device = device
        self.dtype = dtype
        if device.type == "hpu":
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            model = wrap_in_hpu_graph(model, disable_tensor_cache=False)
        self.hidden_size = config.hidden_size

        super(FlashMistral, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Union[FlashBatch, PaddedBatch]:
        # for hpu devices, we use PaddedBatch as we do not have real varlen fwd yet
        return FlashBatch if self.device.type != "hpu" else PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: Union[FlashBatch, PaddedBatch]) -> List[Embedding]:
        if isinstance(batch, PaddedBatch):
            input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
            max_input_lens = 0
            cu_seqlens = torch.cat(
                (input_lens.new_tensor([0]), input_lens.cumsum(-1).int())
            )
            mask = batch.attention_mask.bool()
            bsz, tgt_len = mask.size()
            min_val = torch.finfo(self.dtype).min
            attn_mask = torch.full(
                [bsz, 1, tgt_len, tgt_len],
                fill_value=min_val,
                device=self.device,
                dtype=self.dtype,
            )
            expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, tgt_len)
            attn_mask = attn_mask.masked_fill(expanded_mask, 0.0)
        elif isinstance(batch, FlashBatch):
            cu_seqlens = batch.cu_seqlens
            mask = None
            attn_mask = None
            max_input_lens = batch.max_s

        embedding = self.model.forward(
            input_ids=batch.input_ids,
            position_ids=batch.position_ids,
            cu_seqlens=cu_seqlens,
            max_s=max_input_lens,
            mask=mask,
            attn_mask=attn_mask,
        )
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
