import torch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.classification_model import ClassificationModel
from text_embeddings_server.utils.device import get_device, use_ipex

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


def get_model(model_path: Path, dtype: Optional[str], pool: str):
    if dtype == "float32":
        datatype = torch.float32
    elif dtype == "float16":
        datatype = torch.float16
    elif dtype == "bfloat16":
        datatype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    device = get_device()
    logger.info(f"backend device: {device}")

    config = AutoConfig.from_pretrained(model_path)
    if config.model_type == "bert":
        config: BertConfig
        if (
            use_ipex()
            or device.type in ["cuda", "hpu"]
            and config.position_embedding_type == "absolute"
            and datatype in [torch.float16, torch.bfloat16]
            and FLASH_ATTENTION
        ):
            if pool != "cls":
                return DefaultModel(model_path, device, datatype, pool)
            return FlashBert(model_path, device, datatype)
        if config.architectures[0].endswith("Classification"):
            return ClassificationModel(model_path, device, datatype)
        else:
            return DefaultModel(model_path, device, datatype, pool)
    else:
        if device.type == "hpu":
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph
            from optimum.habana.transformers.modeling_utils import (
                adapt_transformers_to_gaudi,
            )

            adapt_transformers_to_gaudi()
            if config.architectures[0].endswith("Classification"):
                model_handle = ClassificationModel(model_path, device, datatype)
            else:
                model_handle = DefaultModel(model_path, device, datatype, pool)
            model_handle.model = wrap_in_hpu_graph(model_handle.model)
            return model_handle
        elif use_ipex():
            if config.architectures[0].endswith("Classification"):
                return ClassificationModel(model_path, device, datatype)
            else:
                return DefaultModel(model_path, device, datatype, pool)
