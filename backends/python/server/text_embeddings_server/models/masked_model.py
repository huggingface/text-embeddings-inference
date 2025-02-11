import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModelForMaskedLM
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score
from text_embeddings_server.models.pooling import DefaultPooling, SpladePooling

tracer = trace.get_tracer(__name__)


class MaskedLanguageModel(Model):
    def __init__(
        self, model_path: Path, device: torch.device, dtype: torch.dtype, pool: str
    ):
        model = AutoModelForMaskedLM.from_pretrained(model_path).to(dtype).to(device)
        self.hidden_size = model.config.hidden_size
        self.vocab_size = model.config.vocab_size
        self.pooling_mode = pool
        if pool == "splade":
            self.pooling = SpladePooling()
        else:
            self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)

        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(MaskedLanguageModel, self).__init__(model=model, dtype=dtype, device=device)

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
        embedding = self.pooling.forward(output, batch.attention_mask)
        cpu_results = embedding.view(-1).tolist()

        step_size = embedding.shape[-1]
        return [
            Embedding(values=cpu_results[i * step_size : (i + 1) * step_size])
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        pass
