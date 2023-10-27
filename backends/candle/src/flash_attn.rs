use crate::compute_cap::RUNTIME_COMPUTE_CAP;
use candle::Tensor;

#[allow(clippy::too_many_arguments, unused)]
pub(crate) fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor, candle::Error> {
    if *RUNTIME_COMPUTE_CAP == 75 {
        #[cfg(feature = "flash-attn-v1")]
        {
            use candle_flash_attn_v1::flash_attn_varlen;
            return flash_attn_varlen(
                q,
                k,
                v,
                seqlens_q,
                seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
            );
        }
        #[cfg(not(feature = "flash-attn-v1"))]
        candle::bail!("Flash attention v1 is not installed. Use `flash-attn-v1` feature.")
    } else if (80..90).contains(&*RUNTIME_COMPUTE_CAP) {
        #[cfg(feature = "flash-attn")]
        {
            use candle_flash_attn::flash_attn_varlen;
            return flash_attn_varlen(
                q,
                k,
                v,
                seqlens_q,
                seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
            );
        }
        #[cfg(not(feature = "flash-attn"))]
        candle::bail!("Flash attention is not installed. Use `flash-attn-v1` feature.")
    } else if *RUNTIME_COMPUTE_CAP == 90 {
        #[cfg(feature = "flash-attn")]
        {
            use candle_flash_attn::flash_attn_varlen;
            return flash_attn_varlen(
                q,
                k,
                v,
                seqlens_q,
                seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale,
                causal,
            );
        }
        #[cfg(not(feature = "flash-attn"))]
        candle::bail!("Flash attention is not installed. Use `flash-attn-v1` feature.")
    }
    candle::bail!(
        "GPU with CUDA capability {} is not supported",
        *RUNTIME_COMPUTE_CAP
    );
}
