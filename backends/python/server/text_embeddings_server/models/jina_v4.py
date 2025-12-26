import os
from typing import List

import torch
from opentelemetry import trace
from transformers import AutoModel

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import Embedding, PaddedBatch

tracer = trace.get_tracer(__name__)


class JinaV4Model(Model):
    def __init__(
        self,
        model_path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        _ = pool  # Unused but kept for interface parity
        _ = trust_remote
        model = (
            AutoModel.from_pretrained(
                model_path, trust_remote_code=True, torch_dtype=dtype
            )
            .to(dtype)
            .to(device)
        )
        config = model.config

        task_names = getattr(config, "task_names", []) or []
        env_task = os.getenv("JINA_V4_TASK")
        if env_task:
            self.task_label = env_task
        elif task_names:
            self.task_label = task_names[0]
        else:
            self.task_label = "retrieval"

        max_lengths = [
            getattr(config, "sliding_window", None),
            getattr(getattr(config, "text_config", None), "sliding_window", None),
            getattr(config, "max_position_embeddings", None),
            getattr(getattr(config, "text_config", None), "max_position_embeddings", None),
        ]
        self.max_input_length = next((v for v in max_lengths if v), 8192)

        self.hidden_size = getattr(
            getattr(config, "text_config", None), "hidden_size", config.hidden_size
        )

        model.eval()
        super().__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self):
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        attention_mask = batch.attention_mask.to(self.device, dtype=torch.long)
        kwargs = {
            "input_ids": batch.input_ids.to(self.device, dtype=torch.long),
            "attention_mask": attention_mask,
            "task_label": self.task_label,
        }
        outputs = self.model(**kwargs)
        embeddings = outputs.single_vec_emb.to(torch.float32)

        active_rows = attention_mask.sum(dim=1) > 0
        embeddings = embeddings[active_rows]

        cpu_results = embeddings.view(-1, self.hidden_size).tolist()

        return [Embedding(values=row) for row in cpu_results]

    def predict(self, batch):
        raise NotImplementedError("Prediction is not supported for Jina Embeddings V4.")
