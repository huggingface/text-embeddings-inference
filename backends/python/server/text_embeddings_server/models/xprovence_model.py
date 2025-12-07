import os
import torch

from pathlib import Path
from typing import Type, List
from transformers import AutoModel
from huggingface_hub import snapshot_download
from opentelemetry import trace
from loguru import logger

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Score

tracer = trace.get_tracer(__name__)


def _parse_bool(value: str) -> bool:
    """Parse boolean from string with common conventions."""
    return str(value).lower() in ("true", "1", "t", "yes", "on")


class XProvenceModel(Model):
    """
    XProvence: Zero-cost context pruning model for RAG.

    XProvence removes irrelevant sentences from passages based on relevance
    to the query, returning both a reranking score and pruned context.

    Based on bge-reranker-v2-m3 (XLM-RoBERTa), supports 16+ languages.

    Environment Variables:
        XPROVENCE_THRESHOLD (float): Pruning threshold between 0.0-1.0.
            - 0.3 (default): Conservative pruning, minimal performance drop
            - 0.7: Aggressive pruning, higher compression
        XPROVENCE_ALWAYS_SELECT_TITLE (bool): Keep first sentence as title.
            - true (default): Always include first sentence (useful for Wikipedia)
            - false: Only include sentences above threshold
    """

    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = True,
    ):
        # Download all model files including custom Python files for trust_remote_code
        # The Rust router only downloads config/tokenizer/weights, but not custom modeling files
        model_path_str = str(model_path)
        if model_path_str.startswith("/data/models--"):
            # Extract model_id from HF cache path format: /data/models--org--name/...
            # Convert "models--naver--xprovence-reranker-bgem3-v1" to "naver/xprovence-reranker-bgem3-v1"
            parts = model_path_str.split("/")
            for part in parts:
                if part.startswith("models--"):
                    model_id = part.replace("models--", "").replace("--", "/", 1)
                    logger.info(f"XProvence: Downloading custom files for {model_id}")
                    cache_dir = os.getenv("HUGGINGFACE_HUB_CACHE", "/data")
                    snapshot_download(
                        repo_id=model_id,
                        cache_dir=cache_dir,
                        local_files_only=False,
                    )
                    break

        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

        if dtype == torch.bfloat16:
            logger.info("XProvence: using float32 instead of bfloat16 for process() compatibility")
            dtype = torch.float32

        model = model.to(dtype).to(device)

        self.hidden_size = model.config.hidden_size

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

        try:
            threshold_env = os.getenv("XPROVENCE_THRESHOLD", "0.3")
            self.threshold = float(threshold_env)
            if not (0.0 <= self.threshold <= 1.0):
                logger.warning(
                    f"XPROVENCE_THRESHOLD={self.threshold} out of bounds [0.0, 1.0], "
                    f"defaulting to 0.3"
                )
                self.threshold = 0.3
        except ValueError:
            logger.error(
                f"Invalid XPROVENCE_THRESHOLD='{threshold_env}', defaulting to 0.3"
            )
            self.threshold = 0.3

        self.always_select_title = _parse_bool(
            os.getenv("XPROVENCE_ALWAYS_SELECT_TITLE", "true")
        )

        logger.info(
            f"XProvence model loaded: threshold={self.threshold}, "
            f"always_select_title={self.always_select_title} "
            f"(Configure via XPROVENCE_THRESHOLD, XPROVENCE_ALWAYS_SELECT_TITLE env vars)"
        )

        super(XProvenceModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        pass

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Score]:
        """
        XProvence prediction with context pruning support.

        For single-item batches with raw_query/raw_text available,
        uses XProvence's process() method for sentence-level pruning.
        Otherwise falls back to standard forward pass.
        """
        batch_size = len(batch)

        if batch_size == 1 and batch.raw_query and batch.raw_text:
            return self._predict_with_pruning(batch.raw_query, batch.raw_text)

        return self._predict_standard(batch)

    def _predict_with_pruning(self, raw_query: str, raw_text: str) -> List[Score]:
        """
        Use XProvence's process() method for context pruning.

        Returns score with pruned_text containing only relevant sentences.
        """
        try:
            os.environ["TQDM_DISABLE"] = "1"

            original_dtype = torch.get_default_dtype()
            torch.set_default_dtype(torch.float32)

            try:
                output = self.model.process(
                    raw_query,
                    raw_text,
                    threshold=self.threshold,
                    always_select_title=self.always_select_title,
                )
            finally:
                torch.set_default_dtype(original_dtype)

            reranking_score = float(output["reranking_score"])
            pruned_context = output["pruned_context"]

            logger.debug(
                f"XProvence pruning: score={reranking_score:.4f}, "
                f"original_len={len(raw_text)}, pruned_len={len(pruned_context)}"
            )

            return [Score(values=[reranking_score], pruned_text=pruned_context)]

        except Exception as e:
            logger.error(f"XProvence process() failed: {e}, falling back to standard")
            return [Score(values=[0.0], pruned_text=None)]

    def _predict_standard(self, batch: PaddedBatch) -> List[Score]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}

        output = self.model(**kwargs, return_dict=True)

        if hasattr(output, "ranking_scores"):
            scores_tensor = output.ranking_scores
        elif hasattr(output, "logits"):
            scores_tensor = output.logits[:, 0] if output.logits.dim() == 2 else output.logits
        else:
            scores_tensor = output[0]

        if scores_tensor.dim() == 0:
            scores = [float(scores_tensor.item())]
        else:
            scores = scores_tensor.view(-1).tolist()

        if isinstance(scores, float):
            scores = [scores]

        return [Score(values=[float(s)], pruned_text=None) for s in scores]
