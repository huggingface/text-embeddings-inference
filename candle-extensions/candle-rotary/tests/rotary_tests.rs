use anyhow::Result;
use candle::{DType, Device, Tensor, D};

fn to_vec3_round(t: Tensor, digits: i32) -> Result<Vec<Vec<Vec<f32>>>> {
    let b = 10f32.powi(digits);
    let t = t.to_vec3::<f32>()?;
    let t = t
        .iter()
        .map(|t| {
            t.iter()
                .map(|t| t.iter().map(|t| f32::round(t * b) / b).collect())
                .collect()
        })
        .collect();
    Ok(t)
}

fn rotate_half(xs: &Tensor) -> candle::error::Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

fn freqs(rot_dim: usize, seqlen: usize, dev: &Device) -> candle::error::Result<Tensor> {
    let inv_freq: Vec<_> = (0..rot_dim)
        .step_by(2)
        .map(|i| 1f32 / 10000f32.powf(i as f32 / rot_dim as f32))
        .collect();
    let inv_freq_len = inv_freq.len();
    let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
    let t = Tensor::arange(0u32, seqlen as u32, dev)?
        .to_dtype(DType::F32)?
        .reshape((seqlen, 1))?;
    t.matmul(&inv_freq)
}

fn apply_rotary_emb_qkv(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let cos = cos.unsqueeze(1)?; // (seq_len, 1, dim)
    let sin = sin.unsqueeze(1)?; // (seq_len, 1, dim)
    let q_embed = (q.broadcast_mul(&cos)? + rotate_half(q)?.broadcast_mul(&sin))?;
    let k_embed = (k.broadcast_mul(&cos)? + rotate_half(k)?.broadcast_mul(&sin))?;
    Ok((q_embed, k_embed))
}

#[test]
fn rotary() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let seqlen = 12;
    let num_heads = 8;
    let rot_dim = 64;

    let q = Tensor::randn(0.0, 1.0, (seqlen, num_heads, rot_dim), &device)?.to_dtype(DType::F32)?;
    let k = Tensor::randn(0.0, 1.0, (seqlen, num_heads, rot_dim), &device)?.to_dtype(DType::F32)?;

    let (expected_q, expected_k) = {
        let freqs = freqs(rot_dim, seqlen, &device)?;
        let freqs = Tensor::cat(&[&freqs, &freqs], D::Minus1)?;
        apply_rotary_emb_qkv(&q, &k, &freqs.cos()?, &freqs.sin()?)
    }?;

    // Create inv freqs
    let inv_freqs = candle_rotary::inv_freqs(rot_dim, 10000f32, &device)?;
    // Create an over-sized cos sin cache like you would usually do
    let (cos, sin) = candle_rotary::cos_sin(32, &inv_freqs, DType::F32)?;
    // Positions for seqlen
    let position_ids = Tensor::arange(0, seqlen as u32, &device)?;
    // Filter cos and sin
    let cos = cos.index_select(&position_ids, 0)?;
    let sin = sin.index_select(&position_ids, 0)?;

    // Inplace
    candle_rotary::apply_rotary_inplace(&q, &k, &cos, &sin, true)?;

    assert_eq!(to_vec3_round(expected_q, 3)?, to_vec3_round(q, 3)?);
    assert_eq!(to_vec3_round(expected_k, 3)?, to_vec3_round(k, 3)?);

    Ok(())
}
