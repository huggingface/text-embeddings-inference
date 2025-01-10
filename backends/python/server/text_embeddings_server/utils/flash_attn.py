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
    seqlen_q,
    seqlen_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale,
    is_causal=False,
):
    from habana_frameworks.torch.hpex.kernels import FusedSDPA

    total_q, num_head, head_size = q.size()
    total_k, num_head_k, _ = k.size()
    batch_size = seqlen_q.size(0) - 1
    seqlen_q_ = seqlen_q.clone()
    seqlen_q_[:batch_size] = seqlen_q[1:]
    seqlen_q = (seqlen_q_ - seqlen_q)[:batch_size]
    seqlen_k_ = seqlen_k.clone()
    seqlen_k_[:batch_size] = seqlen_k[1:]
    seqlen_k = (seqlen_k_ - seqlen_k)[:batch_size]

    pad_q = torch.zeros(
        [batch_size, max_seqlen_q, num_head, head_size],
        dtype=q.dtype,
        device=q.device,
    )
    pad_k = torch.zeros(
        [batch_size, max_seqlen_k, num_head_k, head_size],
        dtype=k.dtype,
        device=k.device,
    )
    pad_v = torch.zeros(
        [batch_size, max_seqlen_k, num_head_k, head_size],
        dtype=v.dtype,
        device=v.device,
    )
    q_mask = torch.arange(0, max_seqlen_q, device=q.device)[None, :].repeat(
        batch_size, 1
    )
    q_mask = q_mask < seqlen_q[:, None].repeat(1, q_mask.size(-1))
    k_mask = torch.arange(0, max_seqlen_k, device=k.device)[None, :].repeat(
        batch_size, 1
    )
    k_mask = k_mask < seqlen_k[:, None].repeat(1, k_mask.size(-1))
    align_mask_seqlen = max_seqlen_k
    attn_mask = torch.empty(
        [batch_size, 1, 1, align_mask_seqlen],
        dtype=q.dtype,
        device=q.device,
    ).fill_(float("-inf"))
    attn_mask[:, :, :, :max_seqlen_k].masked_fill_(k_mask[:, None, None, :], 0)

    pad_q[q_mask] = q
    pad_k[k_mask] = k
    pad_v[k_mask] = v

    pad_q = pad_q.permute(0, 2, 1, 3)
    pad_k = pad_k.permute(0, 2, 1, 3)
    pad_v = pad_v.permute(0, 2, 1, 3)
    if is_causal:
        attn_mask = None

    out_ = FusedSDPA.apply(
        pad_q, pad_k, pad_v, attn_mask, 0.0, is_causal, softmax_scale
    )
    out_ = out_.permute(0, 2, 1, 3)
    out.copy_(out_[q_mask])
    return out


def attention(q, k, v, out, cu_seqlens, max_s, softmax_scale, is_causal=False):
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
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                softmax_scale,
                is_causal=False,
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
