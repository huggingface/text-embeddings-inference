import torch

from pathlib import Path
from torch import nn
from typing import Type, List
from safetensors import safe_open
from transformers.activations import ACT2FN
from transformers.models.bert import BertConfig
from opentelemetry import trace

# Flash attention imports
import dropout_layer_norm

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import FlashBatch, Embedding
from text_embeddings_server.utils.flash_attn import attention

tracer = trace.get_tracer(__name__)


class FastLayerNorm:
    def __init__(self, prefix, handle, device, dtype, config: BertConfig):
        self.weight = handle.get_tensor(f"{prefix}.weight").to(dtype).to(device)
        self.bias = handle.get_tensor(f"{prefix}.bias").to(dtype).to(device)
        self.variance_epsilon = config.layer_norm_eps

    def forward(self, hidden_states, residual=None):
        normed_hidden_states, res, *rest = dropout_layer_norm.dropout_add_ln_fwd(
            hidden_states,
            residual,
            self.weight,
            self.bias,
            None,
            None,
            None,
            None,
            0.0,
            self.variance_epsilon,
            1.0,
            0,
            None,
            False,
            False,
        )
        if res is None:
            res = hidden_states

        return normed_hidden_states, res


class BertEmbeddings:
    def __init__(self, prefix, handle, device, dtype, config: BertConfig):
        self.word_embeddings_weight = (
            handle.get_tensor(f"{prefix}.word_embeddings.weight").to(dtype).to(device)
        )
        self.token_type_embeddings_weight = (
            handle.get_tensor(f"{prefix}.token_type_embeddings.weight")
            .to(dtype)
            .to(device)
        )

        if config.position_embedding_type == "absolute":
            self.position_embeddings_weight = (
                handle.get_tensor(f"{prefix}.position_embeddings.weight")
                .to(dtype)
                .to(device)
            )
        else:
            raise NotImplementedError(
                "FlashBert only supports absolute position embeddings"
            )

        self.layer_norm = FastLayerNorm(
            f"{prefix}.LayerNorm", handle, device, dtype, config
        )

    def forward(self, input_ids, token_type_ids, position_ids):
        inputs_embeds = nn.functional.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeds = nn.functional.embedding(
            token_type_ids, self.token_type_embeddings_weight
        )
        position_embeds = nn.functional.embedding(
            position_ids, self.position_embeddings_weight
        )

        inputs_embeds += position_embeds

        embeddings, _ = self.layer_norm.forward(inputs_embeds, token_type_embeds)
        return embeddings


class BertAttention:
    def __init__(self, prefix, handle, device, dtype, config: BertConfig):
        query_weight = handle.get_tensor(f"{prefix}.self.query.weight")
        query_bias = handle.get_tensor(f"{prefix}.self.query.bias")
        key_weight = handle.get_tensor(f"{prefix}.self.key.weight")
        key_bias = handle.get_tensor(f"{prefix}.self.key.bias")
        value_weight = handle.get_tensor(f"{prefix}.self.value.weight")
        value_bias = handle.get_tensor(f"{prefix}.self.value.bias")

        self.qkv_weight = (
            torch.cat([query_weight, key_weight, value_weight]).T.to(dtype).to(device)
        )
        self.qkv_bias = (
            torch.cat([query_bias, key_bias, value_bias]).to(dtype).to(device)
        )

        self.dense_weight = (
            handle.get_tensor(f"{prefix}.output.dense.weight").T.to(dtype).to(device)
        )
        self.dense_bias = (
            handle.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)
        )

        self.layer_norm = FastLayerNorm(
            f"{prefix}.output.LayerNorm", handle, device, dtype, config
        )

        self.head_size = config.hidden_size // config.num_attention_heads
        self.softmax_scale = self.head_size**-0.5
        self.num_heads = config.num_attention_heads

    def forward(self, hidden_states, cu_seqlens, max_s):
        residual = hidden_states

        qkv = torch.addmm(self.qkv_bias, hidden_states, self.qkv_weight)
        q, k, v = qkv.view(-1, self.num_heads * 3, self.head_size).split(
            self.num_heads, dim=1
        )

        attn_output = torch.empty_like(q)
        attention(q, k, v, attn_output, cu_seqlens, max_s, self.softmax_scale)

        hidden_states = torch.addmm(
            self.dense_bias,
            attn_output.view(-1, self.num_heads * self.head_size),
            self.dense_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)

        return hidden_states


class BertLayer:
    def __init__(self, prefix, handle, device, dtype, config: BertConfig):
        self.attention = BertAttention(
            f"{prefix}.attention", handle, device, dtype, config
        )

        self.intermediate_weight = (
            handle.get_tensor(f"{prefix}.intermediate.dense.weight")
            .T.to(dtype)
            .to(device)
        )
        self.intermediate_bias = (
            handle.get_tensor(f"{prefix}.intermediate.dense.bias").to(dtype).to(device)
        )

        act = config.hidden_act
        self.intermediate_act_fn = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        self.output_weight = (
            handle.get_tensor(f"{prefix}.output.dense.weight").T.to(dtype).to(device)
        )
        self.output_bias = (
            handle.get_tensor(f"{prefix}.output.dense.bias").to(dtype).to(device)
        )
        self.layer_norm = FastLayerNorm(
            f"{prefix}.output.LayerNorm", handle, device, dtype, config
        )

    def forward(self, hidden_states, cu_seqlens, max_s):
        hidden_states = self.attention.forward(hidden_states, cu_seqlens, max_s)
        residual = hidden_states

        hidden_states = torch.addmm(
            self.intermediate_bias, hidden_states, self.intermediate_weight
        )
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = torch.addmm(
            self.output_bias,
            hidden_states,
            self.output_weight,
        )
        hidden_states, _ = self.layer_norm.forward(hidden_states, residual)
        return hidden_states


class BertEncoder:
    def __init__(self, prefix, handle, device, dtype, config: BertConfig):
        self.layers = [
            BertLayer(f"{prefix}.layer.{i}", handle, device, dtype, config)
            for i in range(config.num_hidden_layers)
        ]

    def forward(self, hidden_states, cu_seqlens, max_s):
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cu_seqlens, max_s)
        return hidden_states


class FlashBertModel:
    def __init__(self, handle, device, dtype, config: BertConfig):
        self.embeddings = BertEmbeddings("embeddings", handle, device, dtype, config)
        self.encoder = BertEncoder("encoder", handle, device, dtype, config)

    def forward(self, input_ids, token_type_ids, position_ids, cu_seqlens, max_s):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, cu_seqlens, max_s)

        return encoder_outputs[cu_seqlens[:-1]]


class FlashBert(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        config = BertConfig.from_pretrained(model_path)
        with safe_open(model_path / "model.safetensors", framework="pt") as f:
            model = FlashBertModel(f, device, dtype, config)

        self.hidden_size = config.hidden_size

        super(FlashBert, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[FlashBatch]:
        return FlashBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: FlashBatch) -> List[Embedding]:
        embedding = self.model.forward(
            input_ids=batch.input_ids,
            token_type_ids=batch.token_type_ids,
            position_ids=batch.position_ids,
            cu_seqlens=batch.cu_seqlens,
            max_s=batch.max_s,
        )
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
