import os
import torch
from text_embeddings_server.utils.device import use_ipex, is_hpu

from loguru import logger

if os.getenv("USE_FLASH_ATTENTION", "").lower() == "false":
    raise ImportError("`USE_FLASH_ATTENTION` is false.")

HAS_FLASH_ATTN = False
HAS_FLASH_ATTN_V2 = False

is_hpu = is_hpu()
use_ipex = use_ipex()

if use_ipex or is_hpu:
    HAS_FLASH_ATTN_V2 = True
else:
    if not torch.cuda.is_available():
        raise ImportError("CUDA is not available")

    major, minor = torch.cuda.get_device_capability()
    is_sm75 = major == 7 and minor == 5
    is_sm8x = major == 8 and minor >= 0
    is_sm90 = major == 9 and minor == 0

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


def hpu_attn(
    q,
    k,
    v,
    out,
    attn_mask,
    seqlen_q,
    seqlen_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal=False,
):
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if is_causal:
        attn_mask = None

    out_ = FusedSDPA.apply(
        q, k, v, attn_mask, 0.0, is_causal, softmax_scale, "fast", False
    )
    out_ = out_.transpose(1, 2)
    out.copy_(out_)
    return out


def attention(
    q, k, v, out, cu_seqlens, max_s, softmax_scale, is_causal=False, attn_mask=None
):
    if HAS_FLASH_ATTN_V2:
        if use_ipex:
            import intel_extension_for_pytorch as ipex

            return ipex.llm.functional.varlen_attention(
                q.contiguous() if q.device.type == "xpu" else q,
                k.contiguous() if k.device.type == "xpu" else k,
                v.contiguous() if v.device.type == "xpu" else v,
                out,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                0,
                softmax_scale,
                zero_tensors=False,
                is_causal=False,
                return_softmax=False,
                gen_=None,
            )
        elif is_hpu:
            return hpu_attn(
                q,
                k,
                v,
                out,
                attn_mask,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                softmax_scale,
                is_causal,
            )

        else:
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
