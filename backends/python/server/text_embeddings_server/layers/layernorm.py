import torch
from text_embeddings_server.utils.import_utils import SYSTEM

from transformers.models.bert import BertConfig

if SYSTEM == "cuda":
    import dropout_layer_norm

    class FastLayerNorm:
        def __init__(self, prefix, handle, device, dtype, config: BertConfig):
            self.weight = handle.get_tensor(f"{prefix}.weight").to(dtype).to(device)
            self.bias = handle.get_tensor(f"{prefix}.bias").to(dtype).to(device)
            self.variance_epsilon = config.layer_norm_eps

        def forward(self, hidden_states, residual=None):
            normed_hidden_states, residual, *rest = dropout_layer_norm.dropout_add_ln_fwd(
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
            if residual is None:
                residual = hidden_states

            return normed_hidden_states, residual

elif SYSTEM == "rocm":
    class FastLayerNorm:
        def __init__(self, prefix, handle, device, dtype, config: BertConfig):
            self.weight = handle.get_tensor(f"{prefix}.weight").to(dtype).to(device)
            self.bias = handle.get_tensor(f"{prefix}.bias").to(dtype).to(device)
            self.variance_epsilon = config.layer_norm_eps

        def forward(self, hidden_states, residual=None):
            if residual is not None:
                hidden_states += residual
            residual = hidden_states

            hidden_states = torch.nn.functional.layer_norm(hidden_states, self.weight.shape, self.weight, self.bias, eps=self.variance_epsilon)

            return hidden_states, residual
else:
    raise ValueError("System not recognized")
