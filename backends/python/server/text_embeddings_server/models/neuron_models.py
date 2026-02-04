import inspect
import os
import torch

from abc import ABC
from pathlib import Path
from typing import Type, List
from opentelemetry import trace
from loguru import logger

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)

# Neuron compilation parameters from environment variables
NEURON_BATCH_SIZE = int(os.getenv("NEURON_BATCH_SIZE", "1"))
NEURON_SEQUENCE_LENGTH = int(os.getenv("NEURON_SEQUENCE_LENGTH", "512"))


class NeuronBaseModel(Model, ABC):
    """Base class for all Neuron models with common functionality."""

    def __init__(
        self,
        model,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.hidden_size = model.config.hidden_size

        # Calculate max input length based on model type
        position_offset = 0
        model_type = model.config.model_type
        if model_type in ["xlm-roberta", "camembert", "roberta"]:
            position_offset = getattr(model.config, "pad_token_id", 1) + 1

        if hasattr(model.config, "max_seq_length"):
            self.max_input_length = model.config.max_seq_length
        elif hasattr(model.config, "n_positions"):
            self.max_input_length = model.config.n_positions
        else:
            self.max_input_length = (
                model.config.max_position_embeddings - position_offset
            )

        # Check which inputs the model supports
        self.has_position_ids = self._check_param_exists(model, "position_ids")
        self.has_token_type_ids = self._check_param_exists(model, "token_type_ids")

        super().__init__(model=model, dtype=dtype, device=device)

    @staticmethod
    def _check_param_exists(model, param_name: str) -> bool:
        """Check if a parameter exists in the model's forward signature."""
        try:
            forward_fn = model.forward if hasattr(model, 'forward') else model.__call__
            return (
                inspect.signature(forward_fn).parameters.get(param_name, None)
                is not None
            )
        except (ValueError, TypeError):
            return False

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    def _prepare_inputs(self, batch: PaddedBatch) -> dict:
        """Prepare input kwargs for model forward pass.

        Note: Neuron models require int64 (long) tensors for inputs.
        """
        kwargs = {
            "input_ids": batch.input_ids.to(torch.long),
            "attention_mask": batch.attention_mask.to(torch.long),
        }
        if self.has_token_type_ids:
            kwargs["token_type_ids"] = batch.token_type_ids.to(torch.long)
        if self.has_position_ids:
            kwargs["position_ids"] = batch.position_ids.to(torch.long)
        return kwargs


class NeuronSentenceTransformersModel(NeuronBaseModel):
    """
    Neuron-optimized model for sentence-transformers.

    Uses optimum.neuron.NeuronModelForSentenceTransformers which is designed
    for sentence embedding models that output sentence_embedding directly.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        try:
            from optimum.neuron import NeuronModelForSentenceTransformers
            is_compiled = self._is_neuron_compiled(model_path)
            export_kwargs = {}
            if not is_compiled:
                export_kwargs = {
                    "export": True,
                    "batch_size": NEURON_BATCH_SIZE,
                    "sequence_length": NEURON_SEQUENCE_LENGTH,
                }
                logger.info(f"Compiling model for Neuron with batch_size={NEURON_BATCH_SIZE}, sequence_length={NEURON_SEQUENCE_LENGTH}")
            model = NeuronModelForSentenceTransformers.from_pretrained(
                model_path,
                **export_kwargs,
            )
        except ImportError:
            # Fallback to legacy import
            from optimum.neuron import NeuronSentenceTransformers
            model = NeuronSentenceTransformers.from_pretrained(model_path)

        super().__init__(model, model_path, device, dtype)
        self.pool = pool
        logger.info(f"Loaded NeuronSentenceTransformersModel with pool={pool}")

    @staticmethod
    def _is_neuron_compiled(model_path: Path) -> bool:
        """Check if the model is already compiled for Neuron."""
        neuron_files = list(model_path.glob("*.neuron")) if model_path.is_dir() else []
        return len(neuron_files) > 0

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = self._prepare_inputs(batch)
        output = self.model(**kwargs)

        sentence_embedding = None
        # NeuronModelForSentenceTransformers returns sentence_embedding directly
        if hasattr(output, "sentence_embedding") and output.sentence_embedding is not None:
            candidate = output.sentence_embedding
            if candidate.abs().sum() > 0:
                sentence_embedding = candidate
        
        # If sentence_embedding is invalid, fall back to manual pooling of token_embeddings
        if sentence_embedding is None:
            # Get token embeddings
            if hasattr(output, "token_embeddings") and output.token_embeddings is not None:
                token_embeddings = output.token_embeddings
            else:
                raise ValueError(f"Cannot extract embeddings from model output: {type(output)}")

            # Apply pooling based on self.pool setting
            if self.pool == "cls":
                sentence_embedding = token_embeddings[:, 0, :]
            elif self.pool == "mean":
                attention_mask = kwargs["attention_mask"].unsqueeze(-1).float()
                sentence_embedding = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            elif self.pool == "last_token":
                seq_lengths = kwargs["attention_mask"].sum(dim=1) - 1
                sentence_embedding = token_embeddings[torch.arange(token_embeddings.size(0)), seq_lengths]
            else:
                raise ValueError(f"Invalid pooling mode: {self.pool}")

        # Convert to list format expected by the gRPC interface
        cpu_results = sentence_embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        raise NotImplementedError("Prediction not supported for sentence transformer models")


class NeuronEmbeddingModel(NeuronBaseModel):
    """
    Neuron-optimized model for feature extraction / embeddings.

    Uses optimum.neuron.NeuronModelForFeatureExtraction for models that
    output hidden states which need to be pooled.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        from optimum.neuron import NeuronModelForFeatureExtraction

        is_compiled = self._is_neuron_compiled(model_path)
        export_kwargs = {}
        if not is_compiled:
            export_kwargs = {
                "export": True,
                "batch_size": NEURON_BATCH_SIZE,
                "sequence_length": NEURON_SEQUENCE_LENGTH,
            }
            logger.info(f"Compiling model for Neuron with batch_size={NEURON_BATCH_SIZE}, sequence_length={NEURON_SEQUENCE_LENGTH}")
        model = NeuronModelForFeatureExtraction.from_pretrained(
            model_path,
            **export_kwargs,
        )

        logger.info(f"DEBUG: model type = {type(model)}")

        super().__init__(model, model_path, device, dtype)
        self.pool = pool

        # Initialize pooling layer
        from text_embeddings_server.models.pooling import DefaultPooling
        self.pooling = DefaultPooling(self.hidden_size, pooling_mode=pool)

        logger.info(f"Loaded NeuronEmbeddingModel with pool={pool}")

    @staticmethod
    def _is_neuron_compiled(model_path: Path) -> bool:
        """Check if the model is already compiled for Neuron."""
        neuron_files = list(model_path.glob("*.neuron")) if model_path.is_dir() else []
        return len(neuron_files) > 0

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = self._prepare_inputs(batch)
        output = self.model(**kwargs)

        # Apply pooling to get sentence embeddings
        embedding = self.pooling.forward(output, batch.attention_mask)

        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        raise NotImplementedError("Prediction not supported for embedding models")


class NeuronClassificationModel(NeuronBaseModel):
    """
    Neuron-optimized model for sequence classification.

    Uses optimum.neuron.NeuronModelForSequenceClassification for classification tasks.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
    ):
        from optimum.neuron import NeuronModelForSequenceClassification

        is_compiled = self._is_neuron_compiled(model_path)
        export_kwargs = {}
        if not is_compiled:
            export_kwargs = {
                "export": True,
                "batch_size": NEURON_BATCH_SIZE,
                "sequence_length": NEURON_SEQUENCE_LENGTH,
            }
            logger.info(f"Compiling model for Neuron with batch_size={NEURON_BATCH_SIZE}, sequence_length={NEURON_SEQUENCE_LENGTH}")
        model = NeuronModelForSequenceClassification.from_pretrained(
            model_path,
            **export_kwargs,
        )

        super().__init__(model, model_path, device, dtype)
        logger.info("Loaded NeuronClassificationModel")

    @staticmethod
    def _is_neuron_compiled(model_path: Path) -> bool:
        """Check if the model is already compiled for Neuron."""
        neuron_files = list(model_path.glob("*.neuron")) if model_path.is_dir() else []
        return len(neuron_files) > 0

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        raise NotImplementedError("Embedding not supported for classification models")

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        kwargs = self._prepare_inputs(batch)
        output = self.model(**kwargs)

        # Get logits from output
        if hasattr(output, "logits"):
            logits = output.logits
        else:
            logits = output[0]

        all_scores = logits.tolist()
        return [Score(values=scores) for scores in all_scores]


class NeuronMaskedLMModel(NeuronBaseModel):
    """
    Neuron-optimized model for Masked Language Modeling (SPLADE).

    Uses optimum.neuron.NeuronModelForMaskedLM for SPLADE-style sparse embeddings.
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "splade",
        trust_remote: bool = False,
    ):
        from optimum.neuron import NeuronModelForMaskedLM

        is_compiled = self._is_neuron_compiled(model_path)
        export_kwargs = {}
        if not is_compiled:
            export_kwargs = {
                "export": True,
                "batch_size": NEURON_BATCH_SIZE,
                "sequence_length": NEURON_SEQUENCE_LENGTH,
            }
            logger.info(f"Compiling model for Neuron with batch_size={NEURON_BATCH_SIZE}, sequence_length={NEURON_SEQUENCE_LENGTH}")
        model = NeuronModelForMaskedLM.from_pretrained(
            model_path,
            **export_kwargs,
        )

        super().__init__(model, model_path, device, dtype)

        # Get vocab size for SPLADE output
        self.vocab_size = model.config.vocab_size
        logger.info(f"Loaded NeuronMaskedLMModel with vocab_size={self.vocab_size}")

    @staticmethod
    def _is_neuron_compiled(model_path: Path) -> bool:
        """Check if the model is already compiled for Neuron."""
        neuron_files = list(model_path.glob("*.neuron")) if model_path.is_dir() else []
        return len(neuron_files) > 0

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = self._prepare_inputs(batch)
        output = self.model(**kwargs)

        # Get logits for SPLADE pooling
        if hasattr(output, "logits"):
            hidden_states = output.logits
        else:
            hidden_states = output[0]

        # SPLADE pooling: ReLU -> log(1+x) -> max pooling
        hidden_states = torch.relu(hidden_states)
        hidden_states = (1 + hidden_states).log()
        hidden_states = torch.mul(hidden_states, batch.attention_mask.unsqueeze(-1))
        sparse_embedding = hidden_states.max(dim=1).values

        cpu_results = sparse_embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.vocab_size : (i + 1) * self.vocab_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        raise NotImplementedError("Prediction not supported for masked LM models")


def create_neuron_model(
    model_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    pool: str = "cls",
    trust_remote: bool = False,
    config=None,
) -> Model:
    """
    Factory function to create the appropriate Neuron model based on the model config.

    Args:
        model_path: Path to the model
        device: Target device (should be xla for Neuron)
        dtype: Data type for the model
        pool: Pooling strategy (cls, mean, lasttoken, splade)
        trust_remote: Whether to trust remote code
        config: Pre-loaded model config (optional)

    Returns:
        Appropriate Neuron model instance
    """
    from transformers import AutoConfig

    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote)

    architectures = getattr(config, "architectures", []) or []
    architecture = architectures[0] if architectures else ""

    logger.info(f"Creating Neuron model for architecture: {architecture}, pool: {pool}")

    # Check for classification models
    if architecture.endswith("ForSequenceClassification") or architecture.endswith("Classification"):
        return NeuronClassificationModel(model_path, device, dtype, pool, trust_remote)

    # Check for SPLADE (masked LM) models
    if pool == "splade" or architecture.endswith("ForMaskedLM"):
        return NeuronMaskedLMModel(model_path, device, dtype, pool, trust_remote)

    # Check for sentence-transformers models
    # These typically have specific config attributes or are in specific repositories
    is_sentence_transformer = (
        hasattr(config, "sentence_transformers_config") or
        hasattr(config, "_name_or_path") and "sentence-transformers" in str(config._name_or_path).lower() or
        hasattr(config, "pooling_mode") or
        (model_path / "sentence_bert_config.json").exists() if model_path.is_dir() else False
    )

    if is_sentence_transformer:
        try:
            return NeuronSentenceTransformersModel(model_path, device, dtype, pool, trust_remote)
        except Exception as e:
            logger.warning(f"Failed to load as SentenceTransformer, falling back to FeatureExtraction: {e}")

    # Default to feature extraction model
    try:
        return NeuronEmbeddingModel(model_path, device, dtype, pool, trust_remote)
    except Exception as e:
        logger.warning(f"Failed to load NeuronEmbeddingModel, trying NeuronSentenceTransformersModel: {e}")
        return NeuronSentenceTransformersModel(model_path, device, dtype, pool, trust_remote)
