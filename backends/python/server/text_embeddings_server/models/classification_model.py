import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModelForSequenceClassification
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)


class ClassificationModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model = model.to(dtype).to(device)

        self.hidden_size = model.config.hidden_size
        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(ClassificationModel, self).__init__(
            model=model, dtype=dtype, device=device
        )

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        pass

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids

        output = self.model(**kwargs, return_dict=True)
        all_scores = output.logits.tolist()
        return [Score(values=scores) for scores in all_scores]
