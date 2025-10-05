import torch
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoConfig
from opentelemetry import trace
from functools import lru_cache
import logging

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Score

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class Qwen3RerankModel(Model):
    """
    Optimized Qwen3 model for reranking tasks with production-grade features.
    
    This implementation incorporates all production feedback:
    - Dynamic positive class detection via id2label
    - Universal autocast support (CUDA/CPU/MPS)
    - Explicit device transfer for inputs
    - Robust logging with proper tensor handling
    - Conservative nan/inf handling
    - torch.inference_mode for optimal performance
    """
    
    def __init__(
        self,
        model_path: Path,
        device: torch.device,
        dtype: torch.dtype,
        pool: str = "cls",
        trust_remote: bool = False,
        enable_mixed_precision: bool = None,
    ):
        # Auto-detect mixed precision support
        if enable_mixed_precision is None:
            enable_mixed_precision = (
                dtype in [torch.float16, torch.bfloat16] and 
                device.type in ["cuda", "mps"]  # MPS also supports mixed precision
            )
        self.enable_mixed_precision = enable_mixed_precision
        
        # Load and validate configuration
        config = self._load_config(model_path, trust_remote)
        
        # Load model with optimizations
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            config=config,
            trust_remote_code=trust_remote,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()
        
        self.device = device
        self.dtype = dtype
        
        # Cache key configuration values
        self.max_length = self._get_max_length(config)
        self.num_labels = getattr(config, "num_labels", 2)
        
        # Initialize parent
        super().__init__(model=self.model, dtype=dtype, device=device)
        
        logger.info(
            f"Initialized Qwen3RerankModel: max_length={self.max_length}, "
            f"num_labels={self.num_labels}, mixed_precision={self.enable_mixed_precision}"
        )
    
    @staticmethod
    @lru_cache(maxsize=8)
    def _load_config(model_path: Path, trust_remote: bool) -> Any:
        """Load and cache model configuration."""
        return AutoConfig.from_pretrained(
            str(model_path),
            trust_remote_code=trust_remote
        )
    
    @staticmethod
    def _get_max_length(config: Any) -> int:
        """Extract maximum sequence length from config."""
        for attr in ["max_position_embeddings", "max_seq_len", "max_length", "n_positions"]:
            if hasattr(config, attr):
                return getattr(config, attr)
        return 512  # Conservative default
    
    def _get_positive_class_index(self) -> int:
        """
        Dynamically determine the positive class index from id2label mapping.
        Returns the index that corresponds to positive/relevant/yes labels.
        """
        id2label = getattr(self.model.config, "id2label", None)
        
        if isinstance(id2label, dict) and len(id2label) == 2:
            # Normalize labels to lowercase for comparison
            normalized = {int(k): str(v).lower() for k, v in id2label.items()}
            
            # Common positive class aliases
            positive_aliases = {"1", "pos", "positive", "relevant", "true", "yes", "entailment"}
            
            # Find positive class index
            for idx, label in normalized.items():
                if label in positive_aliases:
                    logger.debug(f"Detected positive class at index {idx}: '{label}'")
                    return idx
            
            # If no clear positive label found, log warning and default to 1
            logger.warning(
                f"Could not determine positive class from id2label: {id2label}. "
                "Defaulting to index 1."
            )
        
        return 1  # Default to index 1 if no mapping found
    
    def predict(self, batch: PaddedBatch) -> List[Score]:
        """
        Compute reranking scores for the batch with production optimizations.
        
        Improvements:
        - Universal autocast support for all device types
        - Explicit device transfer with non_blocking
        - Dynamic positive class detection
        - Conservative nan/inf handling
        - Robust error handling with proper logging
        """
        with tracer.start_as_current_span("qwen3_rerank_predict") as span:
            span.set_attribute("batch_size", len(batch))
            
            try:
                # Ensure inputs are on correct device
                input_ids = batch.input_ids.to(self.device, non_blocking=True)
                attention_mask = (
                    batch.attention_mask.to(self.device, non_blocking=True)
                    if batch.attention_mask is not None else None
                )
                
                # Set up autocast context based on device
                device_type = "cuda" if self.device.type == "cuda" else "cpu"
                # MPS uses "cpu" device_type in autocast
                if self.device.type == "mps":
                    device_type = "cpu"
                
                amp_dtype = torch.bfloat16 if self.dtype == torch.bfloat16 else torch.float16
                
                # Use inference_mode for optimal performance
                with torch.inference_mode(), torch.autocast(
                    device_type=device_type,
                    dtype=amp_dtype,
                    enabled=self.enable_mixed_precision
                ):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        return_dict=True
                    )
                
                # Extract and clean logits (conservative parameters)
                logits = torch.nan_to_num(
                    outputs.logits,
                    nan=0.0,
                    posinf=0.0,  # Conservative: map to neutral
                    neginf=0.0   # Conservative: map to neutral
                )
                
                # Compute scores based on output shape
                scores = self._compute_scores(logits)
                
                # Ensure scores are in valid range
                scores = torch.clamp(scores, 0.0, 1.0)
                
                # Check for low variance (potential issues)
                if scores.numel() > 1:
                    std_val = float(scores.std().item())
                    if std_val < 1e-6:
                        logger.warning(
                            f"Low score variance detected (std={std_val:.6f}). "
                            "Model may not be differentiating between inputs properly."
                        )
                
                # Convert to Score objects
                return [Score(values=[score]) for score in scores.cpu().tolist()]
                
            except Exception as e:
                logger.error(f"Error in Qwen3RerankModel.predict: {str(e)}", exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                
                # Return neutral scores on error
                batch_size = len(batch)
                logger.warning(f"Returning neutral scores (0.5) for batch of size {batch_size}")
                return [Score(values=[0.5]) for _ in range(batch_size)]
    
    def _compute_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute scores from logits based on the number of output classes.
        
        Handles:
        - Binary classification (2 classes) with dynamic positive class detection
        - Single logit (binary with sigmoid)
        - Multi-class (take max probability)
        """
        num_classes = logits.shape[-1] if len(logits.shape) > 1 else 1
        
        if num_classes == 2:
            # Binary classification - use softmax and select positive class
            positive_idx = self._get_positive_class_index()
            scores = F.softmax(logits, dim=-1)[:, positive_idx]
        elif num_classes == 1:
            # Single logit - use sigmoid
            scores = torch.sigmoid(logits.squeeze(-1))
        else:
            # Multi-class - use max probability
            # Could be enhanced to detect specific classes like "entailment"
            scores = F.softmax(logits, dim=-1).max(dim=-1)[0]
        
        return scores
    
    @property
    def supports_reranking(self) -> bool:
        """Indicates this model supports reranking."""
        return True