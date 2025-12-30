import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModelForMaskedLM
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score
from text_embeddings_server.models.pooling import SpladePooling

tracer = trace.get_tracer(__name__)


class MaskedLanguageModel(Model):
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        model = (
            AutoModelForMaskedLM.from_pretrained(
                model_path, trust_remote_code=trust_remote
            )
            .to(dtype)
            .to(device)
        )
        self.pooling = SpladePooling()
        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = model.config.pad_token_id + 1
        if hasattr(model.config, "max_seq_length"):
            self.max_input_length = model.config.max_seq_length
        else:
            self.max_input_length = (
                model.config.max_position_embeddings - position_offset
            )
        self.has_position_ids = (
            inspect.signature(model.forward).parameters.get("position_ids", None)
            is not None
        )
        self.has_token_type_ids = (
            inspect.signature(model.forward).parameters.get("token_type_ids", None)
            is not None
        )

        super(MaskedLanguageModel, self).__init__(
            model=model, dtype=dtype, device=device
        )

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
