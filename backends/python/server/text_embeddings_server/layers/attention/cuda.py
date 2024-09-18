import os
import torch

from loguru import logger

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

if not torch.cuda.is_available():
    raise ImportError("CUDA is not available")

major, minor = torch.cuda.get_device_capability()
is_sm75 = major == 7 and minor == 5
is_sm8x = major == 8 and minor >= 0
is_sm90 = major == 9 and minor == 0

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False
try:
    try:
        import flash_attn_2_cuda
    except ImportError:
        raise ImportError(
            "Flash Attention V2 is not installed.\n"
            "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
            "or install flash attention v2 with `cd server && make install install-flash-attention-v2`"
        )
    if not (is_sm8x or is_sm90):
        raise ImportError(
            f"GPU with CUDA capability {major} {minor} is not supported for "
            "Flash Attention V2"
        )
    HAS_FLASH_ATTN_V2 = True
except ImportError as e:
    try:
        import flash_attn_cuda
    except ImportError:
        raise ImportError(
            "Flash Attention is not installed.\n"
            "Use the official Docker image (ghcr.io/huggingface/text-generation-inference:latest) "
            "or install flash attention with `cd server && make install install-flash-attention`"
        ) from e

    if not (is_sm75 or is_sm8x or is_sm90):
        raise ImportError(
            f"GPU with CUDA capability {major} {minor} is not supported"
        ) from e
    logger.warning(f"Unable to use Flash Attention V2: {e}")
    HAS_FLASH_ATTN = True


def attention(q, k, v, out, cu_seqlens, max_s, softmax_scale, is_causal=False):
    if HAS_FLASH_ATTN_V2:
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
            -1,
            -1,
            False,
            None,
        )

    if HAS_FLASH_ATTN:
        return flash_attn_cuda.fwd(
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
            0,
            None,
        )

    raise NotImplementedError("flash attention is not installed")
