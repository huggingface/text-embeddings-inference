import inspect
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModel
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding

tracer = trace.get_tracer(__name__)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class JinaModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype).to(device)
        self.hidden_size = model.config.hidden_size

        self.has_token_type_ids = (
                inspect.signature(model.forward).parameters.get("token_type_ids", None)
                is not None
        )

        super(JinaModel, self).__init__(model=model, dtype=dtype, device=device)

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

        batch_model_output = self.model(**kwargs)
        embeddings = mean_pooling(batch_model_output, batch.attention_mask)
        results = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        cpu_results = results.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size: (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]
