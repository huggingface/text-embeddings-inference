from abc import ABC, abstractmethod

import torch
from opentelemetry import trace
from sentence_transformers.models import Pooling
from torch import Tensor

tracer = trace.get_tracer(__name__)


class _Pooling(ABC):
    @abstractmethod
    def forward(self, model_output, attention_mask) -> Tensor:
        pass


class DefaultPooling(_Pooling):
    def __init__(self, hidden_size, pooling_mode) -> None:
        assert (
            pooling_mode != "splade"
        ), "Splade pooling is not supported for DefaultPooling"
        self.pooling = Pooling(hidden_size, pooling_mode=pooling_mode)

    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        pooling_features = {
            "token_embeddings": model_output[0],
            "attention_mask": attention_mask,
        }
        return self.pooling.forward(pooling_features)["sentence_embedding"]


class SpladePooling(_Pooling):
    @tracer.start_as_current_span("pooling")
    def forward(self, model_output, attention_mask) -> Tensor:
        # Implement Splade pooling
        hidden_states = torch.relu(model_output[0])
        hidden_states = (1 + hidden_states).log()
        hidden_states = torch.mul(hidden_states, attention_mask.unsqueeze(-1))
        return hidden_states.max(dim=1).values
