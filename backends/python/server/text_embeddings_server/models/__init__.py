import os
import torch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.masked_model import MaskedLanguageModel
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.classification_model import ClassificationModel
from text_embeddings_server.models.flash_mistral import FlashMistral
from text_embeddings_server.utils.device import get_device, use_ipex

__all__ = ["Model"]

TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "false").lower() in ["true", "1"]
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

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=TRUST_REMOTE_CODE)
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
                if config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
                    return MaskedLanguageModel(
                        model_path,
                        device,
                        datatype,
                        trust_remote=TRUST_REMOTE_CODE,
                    )
                return DefaultModel(
                    model_path, device, datatype, pool, trust_remote=TRUST_REMOTE_CODE
                )
            try:
                return FlashBert(model_path, device, datatype)
            except FileNotFoundError as e:
                logger.info(
                    "Do not have safetensors file for this model, use default transformers model path instead"
                )
                return DefaultModel(
                    model_path, device, datatype, pool, trust_remote=TRUST_REMOTE_CODE
                )
        if config.architectures[0].endswith("Classification"):
            return ClassificationModel(
                model_path, device, datatype, trust_remote=TRUST_REMOTE_CODE
            )
        elif config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
            return MaskedLanguageModel(
                model_path, device, datatype, trust_remote=TRUST_REMOTE_CODE
            )
        else:
            return DefaultModel(
                model_path,
                device,
                datatype,
                pool,
                trust_remote=TRUST_REMOTE_CODE,
            )
    elif config.model_type == "mistral" and device.type == "hpu":
        try:
            return FlashMistral(
                model_path,
                device,
                datatype,
                pool,
            )
        except FileNotFoundError as e:
            return DefaultModel(
                model_path,
                device,
                datatype,
                pool,
                trust_remote=TRUST_REMOTE_CODE,
            )
    else:
        if device.type == "hpu":
            from habana_frameworks.torch.hpu import wrap_in_hpu_graph

            if config.architectures[0].endswith("Classification"):
                model_handle = ClassificationModel(
                    model_path,
                    device,
                    datatype,
                    trust_remote=TRUST_REMOTE_CODE,
                )
            elif config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
                model_handle = MaskedLanguageModel(
                    model_path, device, datatype, trust_remote=TRUST_REMOTE_CODE
                )
            else:
                model_handle = DefaultModel(
                    model_path,
                    device,
                    datatype,
                    pool,
                    trust_remote=TRUST_REMOTE_CODE,
                )
            model_handle.model = wrap_in_hpu_graph(model_handle.model)
            return model_handle
        elif use_ipex():
            if config.architectures[0].endswith("Classification"):
                return ClassificationModel(
                    model_path,
                    device,
                    datatype,
                    trust_remote=TRUST_REMOTE_CODE,
                )
            elif config.architectures[0].endswith("ForMaskedLM") and pool == "splade":
                return MaskedLanguageModel(
                    model_path, device, datatype, trust_remote=TRUST_REMOTE_CODE
                )
            else:
                return DefaultModel(
                    model_path,
                    device,
                    datatype,
                    pool,
                    trust_remote=TRUST_REMOTE_CODE,
                )
