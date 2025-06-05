import torch
import math
from torch import nn
import torch.nn.functional as F
from pathlib import Path
from typing import Type, List, Optional, Union, Tuple
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from opentelemetry import trace
from safetensors import safe_open
from text_embeddings_server.models.pooling import DefaultPooling

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)


class JinaBertConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        feed_forward_type="original",
        emb_pooler=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.feed_forward_type = feed_forward_type
        self.emb_pooler = emb_pooler


class JinaBertEmbeddings:
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, handle, device, dtype, config: JinaBertConfig):
        self.word_embeddings_weight = (
            handle.get_tensor(f"embeddings.word_embeddings.weight").to(dtype).to(device)
        )
        self.token_type_embeddings_weight = (
            handle.get_tensor(f"embeddings.token_type_embeddings.weight")
            .to(dtype)
            .to(device)
        )
        self.layernorm_weight = (
            handle.get_tensor(f"embeddings.LayerNorm.weight").to(dtype).to(device)
        )
        self.layernorm_bias = (
            handle.get_tensor(f"embeddings.LayerNorm.bias").to(dtype).to(device)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )
        self.config = config

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        inputs_embeds = F.embedding(input_ids, self.word_embeddings_weight)
        token_type_embeddings = F.embedding(
            token_type_ids, self.token_type_embeddings_weight
        )

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = F.layer_norm(
            embeddings,
            self.layernorm_weight.shape,
            self.layernorm_weight,
            self.layernorm_bias,
            eps=self.config.layer_norm_eps,
        )
        embeddings = self.dropout(embeddings)
        return embeddings


class JinaBertSelfAttention:
    def __init__(self, prefix, handle, device, dtype, config: JinaBertConfig):
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
            config, "embedding_size"
        ):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_weight = (
            handle.get_tensor(f"{prefix}.query.weight").to(dtype).to(device)
        )
        self.query_bias = handle.get_tensor(f"{prefix}.query.bias").to(dtype).to(device)
        self.key_weight = handle.get_tensor(f"{prefix}.key.weight").to(dtype).to(device)
        self.key_bias = handle.get_tensor(f"{prefix}.key.bias").to(dtype).to(device)
        self.value_weight = (
            handle.get_tensor(f"{prefix}.value.weight").to(dtype).to(device)
        )
        self.value_bias = handle.get_tensor(f"{prefix}.value.bias").to(dtype).to(device)
        self.layer_norm_q_weight = (
            handle.get_tensor(f"{prefix}.layer_norm_q.weight").to(dtype).to(device)
        )
        self.layer_norm_q_bias = (
            handle.get_tensor(f"{prefix}.layer_norm_q.bias").to(dtype).to(device)
        )
        self.layer_norm_k_weight = (
            handle.get_tensor(f"{prefix}.layer_norm_k.weight").to(dtype).to(device)
        )
        self.layer_norm_k_bias = (
            handle.get_tensor(f"{prefix}.layer_norm_k.bias").to(dtype).to(device)
        )

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        q_hidden_states = F.linear(hidden_states, self.query_weight, self.query_bias)
        mixed_query_layer = F.layer_norm(
            q_hidden_states,
            self.layer_norm_q_weight.shape,
            self.layer_norm_q_weight,
            self.layer_norm_q_bias,
            eps=self.config.layer_norm_eps,
        )

        k_hidden_states = F.linear(hidden_states, self.key_weight, self.key_bias)
        key_layer = self.transpose_for_scores(
            F.layer_norm(
                k_hidden_states,
                self.layer_norm_k_weight.shape,
                self.layer_norm_k_weight,
                self.layer_norm_k_bias,
                eps=self.config.layer_norm_eps,
            )
        )

        v_hidden_states = F.linear(hidden_states, self.value_weight, self.value_bias)
        value_layer = self.transpose_for_scores(v_hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores + bias, dim=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer,)

        return outputs


class JinaBertSelfOutput:
    def __init__(self, prefix, handle, device, dtype, config):
        self.config = config
        self.dense_weight = (
            handle.get_tensor(f"{prefix}.dense.weight").to(dtype).to(device)
        )
        self.dense_bias = handle.get_tensor(f"{prefix}.dense.bias").to(dtype).to(device)
        self.layerNorm_weight = (
            handle.get_tensor(f"{prefix}.LayerNorm.weight").to(dtype).to(device)
        )
        self.layerNorm_bias = (
            handle.get_tensor(f"{prefix}.LayerNorm.bias").to(dtype).to(device)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = F.linear(hidden_states, self.dense_weight, self.dense_bias)
        hidden_states = self.dropout(hidden_states)
        hidden_states = F.layer_norm(
            hidden_states + input_tensor,
            self.layerNorm_weight.shape,
            self.layerNorm_weight,
            self.layerNorm_bias,
            eps=self.config.layer_norm_eps,
        )
        return hidden_states


class JinaBertAttention:
    def __init__(self, prefix, handle, device, dtype, config):
        self.self = JinaBertSelfAttention(
            f"{prefix}.self", handle, device, dtype, config
        )
        self.output = JinaBertSelfOutput(
            f"{prefix}.output", handle, device, dtype, config
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self.forward(
            hidden_states,
            attention_mask,
            bias,
        )
        attention_output = self.output.forward(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[
            1:
        ]  # add attentions if we output them
        return outputs


class JinaBertGLUMLP:
    def __init__(self, prefix, handle, device, dtype, config: JinaBertConfig):
        self.config = config
        if config.feed_forward_type == "reglu":
            self.act = nn.ReLU()
        elif config.feed_forward_type == "geglu":
            self.act = nn.GELU()
        else:
            raise ValueError(
                f"feed_forward_type {config.feed_forward_type} not supported"
            )
        self.up_gated_layer_weight = (
            handle.get_tensor(f"{prefix}.up_gated_layer.weight").to(dtype).to(device)
        )
        self.down_layer_weight = (
            handle.get_tensor(f"{prefix}.down_layer.weight").to(dtype).to(device)
        )
        self.down_layer_bias = (
            handle.get_tensor(f"{prefix}.down_layer.bias").to(dtype).to(device)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Up with gate
        hidden_mlp_states = F.linear(hidden_states, self.up_gated_layer_weight, None)
        up = hidden_mlp_states[:, :, : self.config.intermediate_size]
        gated = hidden_mlp_states[:, :, self.config.intermediate_size :]
        hidden_mlp_states = up * self.act(gated)
        hidden_mlp_states = self.dropout(hidden_mlp_states)
        # Down
        return F.linear(hidden_mlp_states, self.down_layer_weight, self.down_layer_bias)


class JinaBertLayer:
    def __init__(self, prefix, handle, device, dtype, config: JinaBertConfig):
        self.attention = JinaBertAttention(
            f"{prefix}.attention", handle, device, dtype, config
        )
        self.config = config
        self.feed_forward_type = config.feed_forward_type
        self.layer_norm_1_weight = (
            handle.get_tensor(f"{prefix}.layer_norm_1.weight").to(dtype).to(device)
        )
        self.layer_norm_1_bias = (
            handle.get_tensor(f"{prefix}.layer_norm_1.bias").to(dtype).to(device)
        )
        self.layer_norm_2_weight = (
            handle.get_tensor(f"{prefix}.layer_norm_2.weight").to(dtype).to(device)
        )
        self.layer_norm_2_bias = (
            handle.get_tensor(f"{prefix}.layer_norm_2.bias").to(dtype).to(device)
        )
        if self.feed_forward_type.endswith("glu"):
            self.mlp = JinaBertGLUMLP(f"{prefix}.mlp", handle, device, dtype, config)
        else:
            raise ValueError(
                f"feed_forward_type {self.feed_forward_type} not supported"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor]:
        # Pre-Norm
        residual = hidden_states

        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attention_outputs = self.attention.forward(
            hidden_states,
            attention_mask,
            bias=bias,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[
            1:
        ]  # add self attentions if we output attention weights

        residual = F.layer_norm(
            residual + attention_output,
            self.layer_norm_1_weight.shape,
            self.layer_norm_1_weight,
            self.layer_norm_1_bias,
            eps=self.config.layer_norm_eps,
        )
        mlp_output = self.mlp.forward(residual)
        layer_output = F.layer_norm(
            residual + mlp_output,
            self.layer_norm_2_weight.shape,
            self.layer_norm_2_weight,
            self.layer_norm_2_bias,
            eps=self.config.layer_norm_eps,
        )
        outputs = (layer_output,) + outputs

        return outputs


class JinaBertEncoder:
    def __init__(self, handle, device, dtype, config: JinaBertConfig):
        self.config = config
        self.layers = [
            JinaBertLayer(f"encoder.layer.{i}", handle, device, dtype, config)
            for i in range(config.num_hidden_layers)
        ]
        self.num_attention_heads = config.num_attention_heads

    def rebuild_alibi_tensor(
        self, size: int, device: Optional[Union[torch.device, str]] = None
    ):
        # Alibi
        # Following https://github.com/ofirpress/attention_with_linear_biases/issues/5 (Implementation 1)
        # In the causal case, you can exploit the fact that softmax is invariant to a uniform translation
        # of the logits, which makes the math work out *after* applying causal masking. If no causal masking
        # will be applied, it is necessary to construct the diagonal mask.
        n_heads = self.num_attention_heads

        def _get_alibi_head_slopes(n_heads: int) -> List[float]:
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n_heads).is_integer():
                return get_slopes_power_of_2(
                    n_heads
                )  # In the paper, we only train models that have 2^a heads for some a. This function has
            else:  # some good properties that only occur when the input is a power of 2. To maintain that even
                closest_power_of_2 = (
                    2 ** math.floor(math.log2(n_heads))
                )  # when the number of heads is not a power of 2, we use this workaround.
                return (
                    get_slopes_power_of_2(closest_power_of_2)
                    + _get_alibi_head_slopes(2 * closest_power_of_2)[0::2][
                        : n_heads - closest_power_of_2
                    ]
                )

        context_position = torch.arange(size, device=device)[:, None]
        memory_position = torch.arange(size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(_get_alibi_head_slopes(n_heads)).to(device) * -1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        # [1, n_heads, max_token_length, max_token_length]
        alibi = alibi.unsqueeze(0)
        assert alibi.shape == torch.Size([1, n_heads, size, size])

        self._current_alibi_size = size
        return alibi

    def forward(
        self,
        hidden_states: torch.Tensor,
        max_len: int,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # Add alibi matrix to extended_attention_mask
        bs, seqlen, _ = hidden_states.size()
        alibi_bias = self.rebuild_alibi_tensor(
            size=max_len, device=hidden_states.device
        ).to(hidden_states.dtype)
        full_alibi_bias = torch.full(
            (bs, self.num_attention_heads, seqlen, seqlen),
            fill_value=torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        full_alibi_bias[:, :, :max_len, :max_len] = alibi_bias

        for i, layer_module in enumerate(self.layers):
            layer_outputs = layer_module.forward(
                hidden_states,
                attention_mask,
                full_alibi_bias,
            )

            hidden_states = layer_outputs[0]

        return hidden_states


class FlashJinaBertModel:
    def __init__(self, handle, device, dtype, config: AutoConfig):
        self.embeddings = JinaBertEmbeddings(handle, device, dtype, config)
        self.encoder = JinaBertEncoder(handle, device, dtype, config)

    def forward(
        self,
        input_ids,
        token_type_ids,
        position_ids,
        max_len,
        attn_mask=None,
    ):
        embeddings = self.embeddings.forward(input_ids, token_type_ids, position_ids)
        encoder_outputs = self.encoder.forward(embeddings, max_len, attn_mask)
        return encoder_outputs


class FlashJinaBert(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "mean",
        trust_remote: bool = True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote)
        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        else:
            self.max_input_length = config.max_position_embeddings

        with safe_open(model_path / "model.safetensors", framework="pt") as f:
            model = FlashJinaBertModel(f, device, dtype, config)
        self.hidden_size = config.hidden_size
        self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)
        self.device = device
        self.dtype = dtype
        self.hidden_size = config.hidden_size

        super(FlashJinaBert, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def mean_pooling(
        self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor
    ):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids}
        kwargs["token_type_ids"] = batch.token_type_ids
        kwargs["position_ids"] = batch.position_ids
        input_lens = batch.attention_mask.cumsum(-1)[:, -1].to(torch.int32)
        max_input_lens = input_lens.max().item()
        kwargs["max_len"] = max_input_lens
        outputs = self.model.forward(**kwargs)

        embedding = self.mean_pooling(outputs, batch.attention_mask)
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        pass
