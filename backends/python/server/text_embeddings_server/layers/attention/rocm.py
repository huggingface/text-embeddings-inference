import os
import torch
from text_embeddings_server.utils.import_utils import SYSTEM
from loguru import logger

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5

if SYSTEM == "rocm":
    try:
        import flash_attn_2_cuda

        logger.info("ROCm: using Flash Attention 2 Composable Kernel implementation.")
    except ImportError as e:
        if major >= 8 or is_sm75:
            architecture_suffix = f"-{SYSTEM}"
            raise ImportError(f"Flash Attention V2 is not installed. {e}")
        else:
            for idx in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(idx)
                if "MI210" not in name and "MI250" not in name and "MI300" not in name:
                    raise ImportError(
                        f"AMD GPU {torch.cuda.get_device_name(idx)} does not support flash-attention"
                    )
            raise ImportError(
                f"AMD GPU with ROCm capability {major} {minor} is not supported"
            ) from e

def attention(q, k, v, out, cu_seqlens, max_s, softmax_scale, is_causal=False):
    return flash_attn_2_cuda.varlen_fwd(
        q,
        k,
        v,
        out,
        cu_seqlens,
        cu_seqlens,
        max_s,
        max_s,
        0.0,
        softmax_scale,
        False,
        is_causal,
        False,
        None,
    )