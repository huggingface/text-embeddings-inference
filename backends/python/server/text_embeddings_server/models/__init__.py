import torch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.classification_model import ClassificationModel

__all__ = ["Model"]

# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)


def get_model(model_path: Path, dtype: Optional[str]):
    if dtype == "float32":
        datatype = torch.float32
    elif dtype == "float16":
        datatype = torch.float16
    elif dtype == "bfloat16":
        datatype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if datatype != torch.float32:
            raise ValueError("CPU device only supports float32 dtype")
        device = torch.device("cpu")

    config = AutoConfig.from_pretrained(model_path)

    if config.model_type == "bert":
        config: BertConfig
        if (
            device.type == "cuda"
            and config.position_embedding_type == "absolute"
            and datatype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            return FlashBert(model_path, device, datatype)
        else:
            if config.architectures[0] in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values():
                return ClassificationModel(model_path, device, datatype)
            else:
                return DefaultModel(model_path, device, datatype)
    else:
        try:
            if config.architectures[0] in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values():
                return ClassificationModel(model_path, device, datatype)
            else:
                return DefaultModel(model_path, device, datatype)
        except:
            raise RuntimeError(f"Unsupported model_type {config.model_type}")
