use crate::compute_cap::get_runtime_compute_cap;
use candle::Tensor;

#[allow(clippy::too_many_arguments, unused)]
pub(crate) fn flash_attn_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    max_seqlen_q: usize,
    max_seqlen_k: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor, candle::Error> {
    let runtime_compute_cap = get_runtime_compute_cap();

    if runtime_compute_cap == 75 {
        if alibi_slopes.is_some() {
            candle::bail!("Flash attention v1 does not support alibi");
        }

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
    } else if (80..90).contains(&runtime_compute_cap) || runtime_compute_cap == 90 {
        #[cfg(feature = "flash-attn")]
        {
            use candle_flash_attn::{flash_attn_varlen, flash_attn_varlen_alibi};

            let attention = if let Some(alibi_slopes) = alibi_slopes {
                flash_attn_varlen_alibi(
                    q,
                    k,
                    v,
                    alibi_slopes,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    causal,
                )
            } else {
                flash_attn_varlen(
                    q,
                    k,
                    v,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    causal,
                )
            };

            return attention;
        }
        #[cfg(not(feature = "flash-attn"))]
        candle::bail!("Flash attention is not installed. Use `flash-attn` feature.")
    }
    candle::bail!(
        "GPU with CUDA capability {} is not supported",
        runtime_compute_cap
    );
}
