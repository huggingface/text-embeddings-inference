use candle::Tensor;
use std::sync::Once;

static INIT: Once = Once::new();
static mut RUNTIME_COMPUTE_CAP: usize = 0;
fn init_runtime_compute_cap() {
    unsafe {
        INIT.call_once(|| {
            use crate::compute_cap::get_runtime_compute_cap;
            RUNTIME_COMPUTE_CAP = get_runtime_compute_cap().unwrap();
        });
    }
}

pub fn get_runtime_compute_cap() -> usize {
    unsafe {
        init_runtime_compute_cap();
        RUNTIME_COMPUTE_CAP
    }
}

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
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    let runtime_compute_cap = get_runtime_compute_cap();

    if runtime_compute_cap == 75 {
        if alibi_slopes.is_some() {
            candle::bail!("Flash attention v1 does not support alibi");
        }
        if window_size_left.is_some() | window_size_right.is_some() {
            candle::bail!("Flash attention v1 does not support attention windowing");
        }

        #[cfg(feature = "flash-attn-v1")]
        {
            use candle_flash_attn_v1::flash_attn_varlen;
            // Flash attention v1 has no notion of grouped-query attention: its kernel
            // indexes K and V with the *query* head index (`params.h`), while the Rust
            // wrapper accepts `num_heads % num_heads_k == 0`. With fewer K/V heads than
            // query heads (Qwen2/Qwen3/Mistral/Llama...), heads >= num_heads_k read past
            // the K/V rows and return garbage or NaN. Expand K/V to the query head count
            // first, the same grouping as `repeat_kv` (query head i uses K/V head
            // i / n_rep). This costs O(tokens * heads * head_dim), not O(L^2).
            let (k, v) = repeat_kv_for_v1(q, k, v)?;
            return flash_attn_varlen(
                q,
                &k,
                &v,
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
    } else if (80..90).contains(&runtime_compute_cap)
        || runtime_compute_cap == 90
        || runtime_compute_cap == 100
        || runtime_compute_cap == 120
        || runtime_compute_cap == 121
    {
        #[cfg(feature = "flash-attn")]
        {
            use candle_flash_attn::{flash_attn_varlen_alibi_windowed, flash_attn_varlen_windowed};

            let window_size_right = if causal {
                Some(0)
            } else if window_size_right.is_some() {
                window_size_right
            } else {
                None
            };

            let attention = if let Some(alibi_slopes) = alibi_slopes {
                flash_attn_varlen_alibi_windowed(
                    q,
                    k,
                    v,
                    alibi_slopes,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
                )
            } else {
                flash_attn_varlen_windowed(
                    q,
                    k,
                    v,
                    seqlens_q,
                    seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale,
                    window_size_left,
                    window_size_right,
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

/// Expand grouped K/V heads (`[total, num_heads_k, head_dim]`) to the query's head
/// count for flash attention v1, which does not support GQA/MQA. A no-op when the
/// head counts already match.
#[allow(unused)]
fn repeat_kv_for_v1(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<(Tensor, Tensor), candle::Error> {
    let (_, num_heads, _) = q.dims3()?;
    let (total_k, num_heads_k, head_size) = k.dims3()?;
    if num_heads == num_heads_k {
        return Ok((k.clone(), v.clone()));
    }
    if num_heads_k == 0 || num_heads % num_heads_k != 0 {
        candle::bail!(
            "number of k/v heads {num_heads_k} must divide number of heads in query {num_heads}"
        );
    }
    let n_rep = num_heads / num_heads_k;
    let expand = |t: &Tensor| -> Result<Tensor, candle::Error> {
        t.unsqueeze(2)?
            .broadcast_as((total_k, num_heads_k, n_rep, head_size))?
            .contiguous()?
            .reshape((total_k, num_heads, head_size))
    };
    Ok((expand(k)?, expand(v)?))
}
