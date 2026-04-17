import inspect
import os
import torch
import torch.nn.functional as F

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Type, List, Tuple
from opentelemetry import trace
from loguru import logger

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.pooling import DefaultPooling
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)

NEURON_MODE = os.getenv("NEURON_MODE", "eager")  # "eager" | "compile"
NEURON_BATCH_SIZE = int(os.getenv("NEURON_BATCH_SIZE", "1"))
NEURON_SEQUENCE_LENGTH = int(os.getenv("NEURON_SEQUENCE_LENGTH", "512"))


def _get_orig_module(model) -> torch.nn.Module:
    """Return the unwrapped module whether or not it has been torch.compiled."""
    return getattr(model, "_orig_mod", model)


def _check_param(model, param_name: str) -> bool:
    try:
        fn = model.forward if hasattr(model, "forward") else model.__call__
        return inspect.signature(fn).parameters.get(param_name) is not None
    except (ValueError, TypeError):
        return False


class NeuronBaseModel(Model, ABC):
    """Base class for Neuron models using torch-native eager or torch.compile mode."""

    def __init__(self, model, device: torch.device, dtype: torch.dtype):
        orig = _get_orig_module(model)
        config = orig.config

        self.hidden_size = config.hidden_size

        position_offset = 0
        if config.model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = getattr(config, "pad_token_id", 1) + 1

        if hasattr(config, "max_seq_length"):
            self.max_input_length = config.max_seq_length
        elif hasattr(config, "n_positions"):
            self.max_input_length = config.n_positions
        else:
            self.max_input_length = config.max_position_embeddings - position_offset

        self.has_position_ids = _check_param(orig, "position_ids")
        self.has_token_type_ids = _check_param(orig, "token_type_ids")

        super().__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def _pad_to_static_shape(self, batch: PaddedBatch) -> Tuple[dict, int]:
        """Pad all inputs to (NEURON_BATCH_SIZE, NEURON_SEQUENCE_LENGTH).

        Neuron requires static shapes; padding to fixed dims avoids recompilation
        on every distinct (batch, seq) pair seen in production.
        Returns (padded_kwargs_on_cpu, actual_batch_size).
        """
        actual_bs = batch.input_ids.shape[0]
        actual_seq = batch.input_ids.shape[1]

        if actual_bs > NEURON_BATCH_SIZE:
            raise ValueError(
                f"Batch size {actual_bs} exceeds NEURON_BATCH_SIZE={NEURON_BATCH_SIZE}. "
                f"Set NEURON_BATCH_SIZE>={actual_bs} to serve this batch."
            )

        seq_pad = max(0, NEURON_SEQUENCE_LENGTH - actual_seq)
        batch_pad = max(0, NEURON_BATCH_SIZE - actual_bs)

        def _pad(t: torch.Tensor) -> torch.Tensor:
            if seq_pad > 0:
                t = F.pad(t, (0, seq_pad), value=0)
            if batch_pad > 0:
                t = F.pad(t, (0, 0, 0, batch_pad), value=0)
            return t

        input_ids = _pad(batch.input_ids.to(torch.long))
        attention_mask = _pad(batch.attention_mask.to(torch.long))
        kwargs: dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        if self.has_token_type_ids:
            kwargs["token_type_ids"] = _pad(batch.token_type_ids.to(torch.long))
        if self.has_position_ids:
            kwargs["position_ids"] = _pad(batch.position_ids.to(torch.long))

        return kwargs, actual_bs


class NeuronDefaultModel(NeuronBaseModel):
    """Neuron model for dense sentence embeddings."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=trust_remote
        ).to(dtype).to(device)

        # Extract before optional compile so DefaultPooling gets the hidden size
        self.pooling = DefaultPooling(model.config.hidden_size, pooling_mode=pool)

        if NEURON_MODE == "compile":
            logger.info("Wrapping NeuronDefaultModel with torch.compile(backend='neuron')")
            model = torch.compile(model, backend="neuron", fullgraph=False)

        super().__init__(model, device, dtype)
        logger.info(f"NeuronDefaultModel ready (mode={NEURON_MODE}, pool={pool})")

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs, actual_bs = self._pad_to_static_shape(batch)

        output = self.model(**{k: v.to(self.device) for k, v in kwargs.items()})

        # Move token embeddings back to CPU; pooling runs on CPU
        token_embeddings = output[0][:actual_bs].to("cpu")
        pool_mask = kwargs["attention_mask"][:actual_bs]  # already on CPU

        # DefaultPooling.forward accepts list[tensor] so it can index [0]
        embedding = self.pooling.forward([token_embeddings], pool_mask)
        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size])
            for i in range(actual_bs)
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        raise NotImplementedError("predict not supported for embedding models")


class NeuronClassificationModel(NeuronBaseModel):
    """Neuron model for sequence classification."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, trust_remote_code=trust_remote
        ).to(dtype).to(device)

        if NEURON_MODE == "compile":
            logger.info("Wrapping NeuronClassificationModel with torch.compile(backend='neuron')")
            model = torch.compile(model, backend="neuron", fullgraph=False)

        super().__init__(model, device, dtype)
        logger.info(f"NeuronClassificationModel ready (mode={NEURON_MODE})")

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        raise NotImplementedError("embed not supported for classification models")

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        kwargs, actual_bs = self._pad_to_static_shape(batch)

        output = self.model(**{k: v.to(self.device) for k, v in kwargs.items()})

        logits = output.logits if hasattr(output, "logits") else output[0]
        logits_cpu = logits[:actual_bs].to("cpu").tolist()

        return [Score(values=scores) for scores in logits_cpu]


class NeuronMaskedLMModel(NeuronBaseModel):
    """Neuron model for masked language modeling (SPLADE sparse embeddings)."""

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "splade",
        trust_remote: bool = False,
    ):
        from transformers import AutoModelForMaskedLM

        model = AutoModelForMaskedLM.from_pretrained(
            model_path, trust_remote_code=trust_remote
        ).to(dtype).to(device)

        # Extract before optional compile
        self.vocab_size = model.config.vocab_size

        if NEURON_MODE == "compile":
            logger.info("Wrapping NeuronMaskedLMModel with torch.compile(backend='neuron')")
            model = torch.compile(model, backend="neuron", fullgraph=False)

        super().__init__(model, device, dtype)
        logger.info(f"NeuronMaskedLMModel ready (mode={NEURON_MODE}, vocab_size={self.vocab_size})")

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs, actual_bs = self._pad_to_static_shape(batch)

        output = self.model(**{k: v.to(self.device) for k, v in kwargs.items()})

        hidden_states = output.logits if hasattr(output, "logits") else output[0]
        hidden_states = hidden_states[:actual_bs].to("cpu")
        mask = kwargs["attention_mask"][:actual_bs].unsqueeze(-1).float()

        # SPLADE pooling: ReLU → log(1+x) → mask → max over sequence
        hidden_states = torch.relu(hidden_states)
        hidden_states = (1 + hidden_states).log()
        hidden_states = hidden_states * mask
        sparse_embedding = hidden_states.max(dim=1).values

        cpu_results = sparse_embedding.view(-1).tolist()
        return [
            Embedding(values=cpu_results[i * self.vocab_size : (i + 1) * self.vocab_size])
            for i in range(actual_bs)
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        raise NotImplementedError("predict not supported for masked LM models")


def create_neuron_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    pool: str = "cls",
    trust_remote: bool = False,
    config=None,
) -> Model:
    """Factory: pick the right Neuron model class from the model architecture."""
    from transformers import AutoConfig

    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote)

    architectures = getattr(config, "architectures", []) or []
    architecture = architectures[0] if architectures else ""

    logger.info(
        f"Creating Neuron model: architecture={architecture}, pool={pool}, mode={NEURON_MODE}"
    )

    if architecture.endswith("ForSequenceClassification") or architecture.endswith("Classification"):
        return NeuronClassificationModel(model_path, device, dtype, pool, trust_remote)

    if pool == "splade" or architecture.endswith("ForMaskedLM"):
        return NeuronMaskedLMModel(model_path, device, dtype, pool, trust_remote)

    return NeuronDefaultModel(model_path, device, dtype, pool, trust_remote)
