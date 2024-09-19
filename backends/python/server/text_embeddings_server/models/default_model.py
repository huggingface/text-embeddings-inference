import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModel
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding

tracer = trace.get_tracer(__name__)


class DefaultModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        model = AutoModel.from_pretrained(model_path).to(dtype).to(device)
        self.hidden_size = model.config.hidden_size

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(DefaultModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids
        output = self.model(**kwargs)
        embedding = output[0][:, 0]
        cpu_results = embedding.reshape(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
