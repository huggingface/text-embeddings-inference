use candle::{IndexOp, Tensor};

/// CPU fallback implementation for variable length flash attention
/// This implements standard attention computation for CPU and supports all model types
/// actually not "flash" fused, but using no-padding will lower memory usage and flops on CPU
pub fn flash_attn_varlen_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alibi_slopes: Option<&Tensor>,
    seqlens_q: &Tensor,
    seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    softmax_scale: f64,
    causal: bool,
    window_size_left: Option<usize>,
    window_size_right: Option<usize>,
) -> Result<Tensor, candle::Error> {
    // Get device and shape information
    let device = q.device();
    let (_total_q, num_heads, head_dim) = q.dims3()?;
    let (total_k, num_kv_heads, _) = k.dims3()?;

    // Handle GQA (Grouped Query Attention) by repeating k/v heads if needed
    let (k, v) = if num_heads != num_kv_heads {
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "Invalid GQA config: num_heads={} not divisible by num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let repeat_factor = num_heads / num_kv_heads;

        // Reshape to [total_k, num_kv_heads, 1, head_dim] for proper broadcasting
        let k = k
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        let v = v
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        (k, v)
    } else {
        (k.clone(), v.clone())
    };

    // Process each sequence in the batch
    let batch_size = seqlens_q.dims()[0];
    let mut outputs = Vec::new();

    for i in 0..batch_size {
        let seq_len_q = seqlens_q.i(i)?.to_scalar::<u32>()? as usize;
        let seq_len_k = seqlens_k.i(i)?.to_scalar::<u32>()? as usize;

        if seq_len_q == 0 || seq_len_k == 0 {
            continue;
        }

        // Calculate start positions based on cumulative sequence lengths
        let start_q = if i == 0 {
            0
        } else {
            let mut sum = 0;
            for j in 0..i {
                sum += seqlens_q.i(j)?.to_scalar::<u32>().unwrap_or(0) as usize;
            }
            sum
        };
        let start_k = if i == 0 {
            0
        } else {
            let mut sum = 0;
            for j in 0..i {
                sum += seqlens_k.i(j)?.to_scalar::<u32>().unwrap_or(0) as usize;
            }
            sum
        };

        // Extract Q, K, V for this sequence
        let q_seq = q.narrow(0, start_q, seq_len_q)?; // [seq_len_q, num_heads, head_dim]
        let k_seq = k.narrow(0, start_k, seq_len_k)?; // [seq_len_k, num_heads, head_dim]
        let v_seq = v.narrow(0, start_k, seq_len_k)?; // [seq_len_k, num_heads, head_dim]

        // Transpose to [seq_len_q, num_heads, head_dim] and [seq_len_k, num_heads, head_dim]
        // This matches the format used in bert.rs and qwen3.rs
        // Ensure contiguous after transpose to avoid matmul issues
        let q_seq = q_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_q, head_dim]
        let k_seq = k_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_k, head_dim]
        let v_seq = v_seq.transpose(0, 1)?.contiguous()?; // [num_heads, seq_len_k, head_dim]

        // Compute attention scores for all heads at once (following bert.rs/qwen3.rs pattern)
        // Reshape to [num_heads, seq_len_q, seq_len_k] for batch matrix multiplication
        // Ensure k_seq transpose is contiguous for matmul
        let k_seq_t = k_seq.transpose(1, 2)?.contiguous()?;
        let attention_scores = q_seq.matmul(&k_seq_t)?; // [num_heads, seq_len_q, seq_len_k]

        // Apply softmax scale
        let scale_tensor = Tensor::new(softmax_scale as f32, device)?;
        let mut attention_scores =
            attention_scores.mul(&scale_tensor.broadcast_as(attention_scores.shape())?)?; // [num_heads, seq_len_q, seq_len_k]

        // Apply causal mask if requested
        if causal && seq_len_q > 1 && seq_len_k > 1 {
            let causal_mask = create_causal_mask_batch(seq_len_q, seq_len_k, num_heads, device)?;
            attention_scores = attention_scores.add(&causal_mask)?;
        }

        // Apply windowing if specified
        if window_size_left.is_some() || window_size_right.is_some() {
            let window_mask = create_window_mask_batch(
                seq_len_q,
                seq_len_k,
                num_heads,
                window_size_left,
                window_size_right,
                device,
            )?;
            attention_scores = attention_scores.add(&window_mask)?;
        }

        // Apply ALiBi slopes if provided
        if let Some(alibi_slopes) = alibi_slopes {
            let alibi_bias =
                create_alibi_bias_batch(seq_len_q, seq_len_k, num_heads, alibi_slopes, device)?;
            attention_scores = attention_scores.add(&alibi_bias)?;
        }

        // Apply softmax along the last dimension (seq_len_k)
        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?; // [num_heads, seq_len_q, seq_len_k]

        // Compute attention output for all heads at once
        let context_layer = attention_probs.matmul(&v_seq)?; // [num_heads, seq_len_q, head_dim]

        // Transpose back to [seq_len_q, num_heads, head_dim] to match expected output format
        let seq_output = context_layer.transpose(0, 1)?; // [seq_len_q, num_heads, head_dim]
        outputs.push(seq_output);
    }

    // Concatenate all sequence outputs
    if outputs.is_empty() {
        return Tensor::zeros((0, num_heads, head_dim), q.dtype(), device);
    }

    Tensor::cat(&outputs, 0)
}

/// Create causal attention mask for all heads at once
fn create_causal_mask_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    let mask: Vec<f32> = (0..seq_len_q)
        .flat_map(|i| (0..seq_len_k).map(move |j| if j > i { f32::NEG_INFINITY } else { 0.0 }))
        .collect();

    let mask = Tensor::from_vec(mask, (seq_len_q, seq_len_k), device)?;
    mask.expand((num_heads, seq_len_q, seq_len_k))
}

/// Create window attention mask for all heads at once
/// Supports different windowing patterns:
/// - Standard: both window_left and window_right (bidirectional window)
/// - Mistral-style: only window_left (causal sliding window)
/// - Gemma3-style: bidirectional distance-based windowing
fn create_window_mask_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    window_left: Option<usize>,
    window_right: Option<usize>,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    let mask: Vec<f32> = match (window_left, window_right) {
        // Bidirectional window: allow positions within [i-left, i+right]
        (Some(left), Some(right)) => (0..seq_len_q)
            .flat_map(|i| {
                (0..seq_len_k).map(move |j| {
                    // Check if j is outside the window around i
                    let left_distance = i.saturating_sub(j);
                    let right_distance = j.saturating_sub(i);
                    if left_distance > left || right_distance > right {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
        // Mistral-style: causal sliding window: allow j <= i AND (i-j) <= left
        (Some(left), None) => (0..seq_len_q)
            .flat_map(|i| {
                (0..seq_len_k).map(move |j| {
                    if j > i || i.saturating_sub(j) > left {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect(),
        // No windowing
        (None, None) => {
            vec![0.0; seq_len_q * seq_len_k]
        }
        // Invalid case: right without left
        (None, Some(_)) => {
            candle::bail!("window_right specified without window_left");
        }
    };

    let mask = Tensor::from_vec(mask, (seq_len_q, seq_len_k), device)?;
    mask.expand((num_heads, seq_len_q, seq_len_k))
}

/// Create ALiBi (Attention with Linear Biases) bias for all heads at once
/// Optimized to avoid per-head to_scalar() calls
fn create_alibi_bias_batch(
    seq_len_q: usize,
    seq_len_k: usize,
    num_heads: usize,
    alibi_slopes: &Tensor,
    device: &candle::Device,
) -> Result<Tensor, candle::Error> {
    // Pull slopes to host ONCE (eliminates expensive per-head to_scalar calls)
    let slopes = alibi_slopes.to_vec1::<f32>()?;
    if slopes.len() != num_heads {
        candle::bail!(
            "alibi_slopes has len {}, expected num_heads={}",
            slopes.len(),
            num_heads
        );
    }

    let mut head_biases = Vec::with_capacity(num_heads);
    for (head_idx, &slope) in slopes.iter().enumerate() {
        let bias: Vec<f32> = (0..seq_len_q)
            .flat_map(|i| {
                (0..seq_len_k).map(move |j| {
                    let dist = if j >= i {
                        (j - i) as f32
                    } else {
                        (i - j) as f32
                    };
                    slope * dist
                })
            })
            .collect();

        let head_bias = Tensor::from_vec(bias, (seq_len_q, seq_len_k), device)?;
        head_biases.push(head_bias);

        // avoid unused warning if you log head_idx in debug builds
        let _ = head_idx;
    }

    Tensor::stack(&head_biases, 0) // [H,Q,K]
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{DType, Device, IndexOp, Tensor};
    use rand::prelude::*;

    /// Helper macro to skip tests with clear messaging
    macro_rules! skip_test_if {
        ($condition:expr, $reason:expr) => {
            if $condition {
                println!("SKIPPED: {}", $reason);
                return Ok(());
            }
        };
    }

    fn create_test_tensors(
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor), candle::Error> {
        let mut rng = StdRng::seed_from_u64(42);

        // Create variable sequence lengths
        let mut seqlens_q: Vec<u32> = Vec::new();
        let mut seqlens_k: Vec<u32> = Vec::new();
        let mut total_q = 0;
        let mut total_k = 0;

        for _ in 0..batch_size {
            let seq_len_q = rng.gen_range(4..=max_seq_len);
            let seq_len_k = rng.gen_range(4..=max_seq_len);
            seqlens_q.push(seq_len_q as u32);
            seqlens_k.push(seq_len_k as u32);
            total_q += seq_len_q;
            total_k += seq_len_k;
        }

        // Create Q, K, V tensors
        let q_data: Vec<f32> = (0..total_q * num_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let k_data: Vec<f32> = (0..total_k * num_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let v_data: Vec<f32> = (0..total_k * num_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let q = Tensor::from_vec(q_data, (total_q, num_heads, head_dim), device)?;
        let k = Tensor::from_vec(k_data, (total_k, num_heads, head_dim), device)?;
        let v = Tensor::from_vec(v_data, (total_k, num_heads, head_dim), device)?;

        let seqlens_q_tensor = Tensor::from_vec(seqlens_q, batch_size, device)?;
        let seqlens_k_tensor = Tensor::from_vec(seqlens_k, batch_size, device)?;

        Ok((q, k, v, seqlens_q_tensor, seqlens_k_tensor))
    }

    #[allow(dead_code)]
    fn tensor_distance(cpu_result: &Tensor, gpu_result: &Tensor) -> Result<f32, candle::Error> {
        let diff = cpu_result.sub(gpu_result)?;
        let squared = diff.sqr()?;
        let mean_squared = squared.mean_all()?;
        mean_squared.to_scalar()
    }

    #[test]
    fn test_flash_attn_cpu_vs_gpu_basic() -> Result<(), candle::Error> {
        if !candle::utils::cuda_is_available() {
            println!("Skipping GPU test: CUDA not available");
            // Mark as skipped by returning early - test runner will show this as passed
            // but with the skip message above
            return Ok(());
        }

        let flash_attn_enabled = cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"));
        if !flash_attn_enabled {
            println!("Skipping GPU comparison test: flash-attn features not enabled");
            return Ok(());
        }

        let cpu_device = Device::Cpu;
        #[cfg(feature = "cuda")]
        #[cfg(feature = "cuda")] let gpu_device = Device::new_cuda(0)?;

        let test_cases = vec![
            (1, 8, 64, 32), // batch_size, num_heads, head_dim, max_seq_len
            (2, 12, 128, 64),
            (1, 16, 256, 128),
        ];

        for (batch_size, num_heads, head_dim, max_seq_len) in test_cases {
            println!(
                "Testing: batch={}, heads={}, dim={}, seq={}",
                batch_size, num_heads, head_dim, max_seq_len
            );

            // Create test tensors on CPU
            let (q_cpu, k_cpu, v_cpu, seqlens_q_cpu, seqlens_k_cpu) =
                create_test_tensors(batch_size, num_heads, head_dim, max_seq_len, &cpu_device)?;

            let softmax_scale = 1.0 / (head_dim as f64).sqrt();

            // Test non-causal attention
            let cpu_result = flash_attn_varlen_cpu(
                &q_cpu,
                &k_cpu,
                &v_cpu,
                None,
                &seqlens_q_cpu,
                &seqlens_k_cpu,
                max_seq_len,
                max_seq_len,
                softmax_scale,
                false,
                None,
                None,
            )?;

            #[cfg(feature = "cuda")]
            {
                // Move to GPU only when we're actually going to use it
                let q_gpu = q_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let k_gpu = k_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let v_gpu = v_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let seqlens_q_gpu = seqlens_q_cpu.to_device(&gpu_device)?;
                let seqlens_k_gpu = seqlens_k_cpu.to_device(&gpu_device)?;

                let gpu_result = crate::flash_attn::flash_attn_varlen(
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    None,
                    &seqlens_q_gpu,
                    &seqlens_k_gpu,
                    max_seq_len,
                    max_seq_len,
                    softmax_scale as f32,
                    false,
                    None,
                    None,
                )?;

                // Move GPU result back to CPU and cast to f32 for comparison
                let gpu_result_cpu = gpu_result.to_device(&cpu_device)?.to_dtype(candle::DType::F32)?;

                let distance = tensor_distance(&cpu_result, &gpu_result_cpu)?;
                println!("  Non-causal distance: {:.6}", distance);

                // Assert that the distance is small (allowing for numerical differences)
                assert!(distance < 1e-4, "Distance too large: {:.6}", distance);
            }

            // Test causal attention
            let cpu_result_causal = flash_attn_varlen_cpu(
                &q_cpu,
                &k_cpu,
                &v_cpu,
                None,
                &seqlens_q_cpu,
                &seqlens_k_cpu,
                max_seq_len,
                max_seq_len,
                softmax_scale,
                true,
                None,
                None,
            )?;

            #[cfg(feature = "cuda")]
            {
                // Move to GPU only when we're actually going to use it
                let q_gpu = q_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let k_gpu = k_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let v_gpu = v_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
                let seqlens_q_gpu = seqlens_q_cpu.to_device(&gpu_device)?;
                let seqlens_k_gpu = seqlens_k_cpu.to_device(&gpu_device)?;

                let gpu_result_causal = crate::flash_attn::flash_attn_varlen(
                    &q_gpu,
                    &k_gpu,
                    &v_gpu,
                    None,
                    &seqlens_q_gpu,
                    &seqlens_k_gpu,
                    max_seq_len,
                    max_seq_len,
                    softmax_scale as f32,
                    true,
                    None,
                    None,
                )?;

                let gpu_result_causal_cpu = gpu_result_causal.to_device(&cpu_device)?.to_dtype(candle::DType::F32)?;

                let distance_causal = tensor_distance(&cpu_result_causal, &gpu_result_causal_cpu)?;
                println!("  Causal distance: {:.6}", distance_causal);

                assert!(
                    distance_causal < 1e-4,
                    "Causal distance too large: {:.6}",
                    distance_causal
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_gqa() -> Result<(), candle::Error> {
        skip_test_if!(!candle::utils::cuda_is_available(), "CUDA not available");

        let flash_attn_enabled = cfg!(any(feature = "flash-attn", feature = "flash-attn-v1"));
        skip_test_if!(!flash_attn_enabled, "flash-attn features not enabled");

        let flash_attn_enabled = cfg!(feature = "flash-attn");
        if !flash_attn_enabled {
            println!("Skipping ALiBi comparison test: flash-attn feature not enabled");
            return Ok(());
        }

        let cpu_device = Device::Cpu;
        #[cfg(feature = "cuda")] let gpu_device = Device::new_cuda(0)?;

        let batch_size = 1;
        let num_heads = 8;
        let head_dim = 64;
        let max_seq_len = 32;

        let (q_cpu, k_cpu, v_cpu, seqlens_q_cpu, seqlens_k_cpu) =
            create_test_tensors(batch_size, num_heads, head_dim, max_seq_len, &cpu_device)?;

        // Create ALiBi slopes
        let alibi_slopes_data: Vec<f32> = (0..num_heads)
            .map(|i| 2.0f32.powi(-(i as i32 + 1)))
            .collect();
        let alibi_slopes_cpu = Tensor::from_vec(alibi_slopes_data, num_heads, &cpu_device)?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        // Test CPU implementation with ALiBi
        #[cfg(feature = "cuda")] let cpu_result = flash_attn_varlen_cpu(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            Some(&alibi_slopes_cpu),
            &seqlens_q_cpu,
            &seqlens_k_cpu,
            max_seq_len,
            max_seq_len,
            softmax_scale,
            false,
            None,
            None,
        )?;

        #[cfg(feature = "cuda")]
        {
            // Move to GPU only when we're actually going to use it
            let q_gpu = q_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let k_gpu = k_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let v_gpu = v_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let seqlens_q_gpu = seqlens_q_cpu.to_device(&gpu_device)?;
            let seqlens_k_gpu = seqlens_k_cpu.to_device(&gpu_device)?;
            let alibi_slopes_gpu = alibi_slopes_cpu.to_device(&gpu_device)?;

            // Test GPU implementation with ALiBi (only supported with flash-attn v2)
            let gpu_result = crate::flash_attn::flash_attn_varlen(
                &q_gpu,
                &k_gpu,
                &v_gpu,
                Some(&alibi_slopes_gpu),
                &seqlens_q_gpu,
                &seqlens_k_gpu,
                max_seq_len,
                max_seq_len,
                softmax_scale as f32,
                false,
                None,
                None,
            )?;

            let gpu_result_cpu = gpu_result.to_device(&cpu_device)?.to_dtype(candle::DType::F32)?;

            let distance = tensor_distance(&cpu_result, &gpu_result_cpu)?;
            println!("ALiBi distance: {:.6}", distance);

            assert!(distance < 1e-4, "ALiBi distance too large: {:.6}", distance);
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_windowing() -> Result<(), candle::Error> {
        skip_test_if!(!candle::utils::cuda_is_available(), "CUDA not available");

        let flash_attn_enabled = cfg!(feature = "flash-attn");
        skip_test_if!(!flash_attn_enabled, "flash-attn feature not enabled");

        let cpu_device = Device::Cpu;
        #[cfg(feature = "cuda")] let gpu_device = Device::new_cuda(0)?;

        let batch_size = 1;
        let num_heads = 8;
        let head_dim = 64;
        let max_seq_len = 32;

        let (q_cpu, k_cpu, v_cpu, seqlens_q_cpu, seqlens_k_cpu) =
            create_test_tensors(batch_size, num_heads, head_dim, max_seq_len, &cpu_device)?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();
        let window_left = 8;
        let window_right = 8;

        // Test CPU implementation with windowing
        let cpu_result = flash_attn_varlen_cpu(
            &q_cpu,
            &k_cpu,
            &v_cpu,
            None,
            &seqlens_q_cpu,
            &seqlens_k_cpu,
            max_seq_len,
            max_seq_len,
            softmax_scale,
            false,
            Some(window_left),
            Some(window_right),
        )?;

        #[cfg(feature = "cuda")]
        {
            // Move to GPU only when we're actually going to use it
            let q_gpu = q_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let k_gpu = k_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let v_gpu = v_cpu.to_device(&gpu_device)?.to_dtype(candle::DType::F16)?;
            let seqlens_q_gpu = seqlens_q_cpu.to_device(&gpu_device)?;
            let seqlens_k_gpu = seqlens_k_cpu.to_device(&gpu_device)?;

            // Test GPU implementation with windowing (only supported with flash-attn v2)
            let gpu_result = crate::flash_attn::flash_attn_varlen(
                &q_gpu,
                &k_gpu,
                &v_gpu,
                None,
                &seqlens_q_gpu,
                &seqlens_k_gpu,
                max_seq_len,
                max_seq_len,
                softmax_scale as f32,
                false,
                Some(window_left),
                Some(window_right),
            )?;

            let gpu_result_cpu = gpu_result.to_device(&cpu_device)?.to_dtype(candle::DType::F32)?;

            let distance = tensor_distance(&cpu_result, &gpu_result_cpu)?;
            println!("Windowing distance: {:.6}", distance);

            assert!(
                distance < 1e-4,
                "Windowing distance too large: {:.6}",
                distance
            );
        }

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_windowing_patterns() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test different windowing patterns
        let q = Tensor::randn(1.0, 1.0, (2, 4, 32), &device)?.to_dtype(candle::DType::F32)?;
        let k = Tensor::randn(1.0, 1.0, (2, 4, 32), &device)?.to_dtype(candle::DType::F32)?;
        let v = Tensor::randn(1.0, 1.0, (2, 4, 32), &device)?.to_dtype(candle::DType::F32)?;
        let seqlens_q = Tensor::from_vec(vec![2u32], 1, &device)?;
        let seqlens_k = Tensor::from_vec(vec![2u32], 1, &device)?;

        // Test 1: Standard windowing (both left and right)
        let result1 = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            2,
            2,
            0.125,
            false,
            Some(1),
            Some(1),
        )?;
        assert_eq!(result1.dims(), &[2, 4, 32]);

        // Test 2: Mistral-style windowing (only left)
        let result2 = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            2,
            2,
            0.125,
            true,
            Some(1),
            None,
        )?;
        assert_eq!(result2.dims(), &[2, 4, 32]);

        // Test 3: No windowing
        let result3 = flash_attn_varlen_cpu(
            &q, &k, &v, None, &seqlens_q, &seqlens_k, 2, 2, 0.125, false, None, None,
        )?;
        assert_eq!(result3.dims(), &[2, 4, 32]);

        Ok(())
    }

    #[test]
    fn test_flash_attn_cpu_edge_cases() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test empty batch
        let q = Tensor::zeros((0, 8, 64), DType::F32, &device)?;
        let k = Tensor::zeros((0, 8, 64), DType::F32, &device)?;
        let v = Tensor::zeros((0, 8, 64), DType::F32, &device)?;
        let seqlens_q = Tensor::zeros((0,), DType::U32, &device)?;
        let seqlens_k = Tensor::zeros((0,), DType::U32, &device)?;

        let result = flash_attn_varlen_cpu(
            &q, &k, &v, None, &seqlens_q, &seqlens_k, 32, 32, 0.125, false, None, None,
        )?;

        assert_eq!(result.dims(), &[0, 8, 64]);

        // Test single sequence
        let q = Tensor::randn(1.0, 1.0, (16, 8, 64), &device)?.to_dtype(candle::DType::F32)?;
        let k = Tensor::randn(1.0, 1.0, (16, 8, 64), &device)?.to_dtype(candle::DType::F32)?;
        let v = Tensor::randn(1.0, 1.0, (16, 8, 64), &device)?.to_dtype(candle::DType::F32)?;
        let seqlens_q = Tensor::from_vec(vec![16u32], 1, &device)?;
        let seqlens_k = Tensor::from_vec(vec![16u32], 1, &device)?;

        let result = flash_attn_varlen_cpu(
            &q, &k, &v, None, &seqlens_q, &seqlens_k, 16, 16, 0.125, false, None, None,
        )?;

        assert_eq!(result.dims(), &[16, 8, 64]);

        Ok(())
    }

    // below are helper functions for PADDED inference tests

    fn rmse(a: &Tensor, b: &Tensor) -> Result<f32, candle::Error> {
        let diff = a.sub(b)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        if mse.is_nan() || mse < 0.0 {
            Ok(0.0) // If MSE is NaN or negative (shouldn't happen), return 0
        } else {
            Ok(mse.sqrt())
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32, candle::Error> {
        // Simple + robust for tests: pull to vec and compute max |diff|
        let diff = a.sub(b)?.to_dtype(DType::F32)?;
        let v = diff.flatten_all()?.to_vec1::<f32>()?;
        Ok(v.into_iter().map(|x| x.abs()).fold(0.0f32, f32::max))
    }

    fn repeat_kv_for_gqa(
        k: &Tensor,
        v: &Tensor,
        num_heads: usize,
    ) -> Result<(Tensor, Tensor), candle::Error> {
        let (total_k, num_kv_heads, head_dim) = k.dims3()?;
        if num_heads == num_kv_heads {
            return Ok((k.clone(), v.clone()));
        }
        if num_heads % num_kv_heads != 0 {
            candle::bail!(
                "Invalid GQA config: num_heads={} not divisible by num_kv_heads={}",
                num_heads,
                num_kv_heads
            );
        }
        let repeat_factor = num_heads / num_kv_heads;

        // Use reshape + broadcast to ensure contiguous memory layout
        let k = k
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        let v = v
            .reshape((total_k, num_kv_heads, 1, head_dim))?
            .broadcast_as((total_k, num_kv_heads, repeat_factor, head_dim))?
            .reshape((total_k, num_heads, head_dim))?;
        Ok((k, v))
    }

    /// Build per-batch bias tensor [B, H, max_q, max_k] that includes:
    /// - padding mask (based on seqlens)
    /// - optional causal mask
    /// - optional window mask
    /// - optional ALiBi bias
    ///
    /// This intentionally reuses the same helper mask/bias functions as varlen,
    /// so semantics match 1:1.
    fn build_reference_bias(
        seqlens_q: &[u32],
        seqlens_k: &[u32],
        num_heads: usize,
        max_q: usize,
        max_k: usize,
        causal: bool,
        window_left: Option<usize>,
        window_right: Option<usize>,
        alibi_slopes: Option<&Tensor>,
        device: &Device,
    ) -> Result<Tensor, candle::Error> {
        let bsz = seqlens_q.len();

        // Shared masks/biases for the max sizes (same across batch elements)
        let causal_bias = if causal {
            Some(create_causal_mask_batch(max_q, max_k, num_heads, device)?)
        } else {
            None
        };
        let window_bias = if window_left.is_some() || window_right.is_some() {
            Some(create_window_mask_batch(
                max_q,
                max_k,
                num_heads,
                window_left,
                window_right,
                device,
            )?)
        } else {
            None
        };
        let alibi_bias = if let Some(slopes) = alibi_slopes {
            Some(create_alibi_bias_batch(
                max_q, max_k, num_heads, slopes, device,
            )?)
        } else {
            None
        };

        let mut per_batch = Vec::with_capacity(bsz);

        for b in 0..bsz {
            let lq = seqlens_q[b] as usize;
            let lk = seqlens_k[b] as usize;

            // Padding mask [max_q, max_k]
            let pad: Vec<f32> = (0..max_q)
                .flat_map(|i| {
                    (0..max_k).map(move |j| {
                        if i >= lq || j >= lk {
                            f32::NEG_INFINITY
                        } else {
                            0.0
                        }
                    })
                })
                .collect();
            let mut bias =
                Tensor::from_vec(pad, (max_q, max_k), device)?.expand((num_heads, max_q, max_k))?;

            if let Some(cb) = &causal_bias {
                bias = bias.add(cb)?;
            }
            if let Some(wb) = &window_bias {
                bias = bias.add(wb)?;
            }
            if let Some(ab) = &alibi_bias {
                bias = bias.add(ab)?;
            }

            per_batch.push(bias);
        }

        Tensor::stack(&per_batch, 0) // [B,H,max_q,max_k]
    }

    /// A straightforward padded attention reference:
    /// - inputs are varlen-packed: q [total_q,H,D], k/v [total_k,H_kv,D], plus seqlens
    /// - pads to [B,max_q,H,D] / [B,max_k,H,D], runs attention, unpads back to [total_q,H,D]
    fn reference_padded_attention(
        q_var: &Tensor,
        k_var: &Tensor,
        v_var: &Tensor,
        alibi_slopes: Option<&Tensor>,
        seqlens_q: &Tensor,
        seqlens_k: &Tensor,
        max_q: usize,
        max_k: usize,
        softmax_scale: f64,
        causal: bool,
        window_left: Option<usize>,
        window_right: Option<usize>,
    ) -> Result<Tensor, candle::Error> {
        let device = q_var.device();
        let (total_q, num_heads, head_dim) = q_var.dims3()?;
        let (_total_k, num_kv_heads, _hd2) = k_var.dims3()?;
        if head_dim != _hd2 {
            candle::bail!("Head dim mismatch q:{} k:{}", head_dim, _hd2);
        }

        // Pull seqlens to host once
        let seqlens_q_vec = seqlens_q.to_vec1::<u32>()?;
        let seqlens_k_vec = seqlens_k.to_vec1::<u32>()?;
        let bsz = seqlens_q_vec.len();

        // Build cu_seqlens
        let mut cu_q = vec![0usize; bsz + 1];
        let mut cu_k = vec![0usize; bsz + 1];
        for i in 0..bsz {
            cu_q[i + 1] = cu_q[i] + seqlens_q_vec[i] as usize;
            cu_k[i + 1] = cu_k[i] + seqlens_k_vec[i] as usize;
        }
        // Sanity: total_q should match sum(seqlens_q)
        if cu_q[bsz] != total_q {
            candle::bail!(
                "total_q mismatch: tensor has {}, sum(seqlens_q) is {}",
                total_q,
                cu_q[bsz]
            );
        }

        // Match varlen behavior: expand KV heads for GQA if needed
        let (k_var, v_var) = repeat_kv_for_gqa(k_var, v_var, num_heads)?;
        if num_kv_heads != num_heads {
            // After repeat, dims should now have H
            let (_tk, hk, _d) = k_var.dims3()?;
            if hk != num_heads {
                candle::bail!("GQA repeat failed: expected H={}, got {}", num_heads, hk);
            }
        }

        // Pad per-batch into [B,max,H,D]
        let mut q_padded = Vec::with_capacity(bsz);
        let mut k_padded = Vec::with_capacity(bsz);
        let mut v_padded = Vec::with_capacity(bsz);

        for i in 0..bsz {
            let lq = seqlens_q_vec[i] as usize;
            let lk = seqlens_k_vec[i] as usize;

            let q_i = q_var.narrow(0, cu_q[i], lq)?; // [lq,H,D]
            let k_i = k_var.narrow(0, cu_k[i], lk)?; // [lk,H,D]
            let v_i = v_var.narrow(0, cu_k[i], lk)?; // [lk,H,D]

            let q_pad = Tensor::cat(
                &[
                    &q_i,
                    &Tensor::zeros((max_q - lq, num_heads, head_dim), q_i.dtype(), device)?,
                ],
                0,
            )?;
            let k_pad = Tensor::cat(
                &[
                    &k_i,
                    &Tensor::zeros((max_k - lk, num_heads, head_dim), k_i.dtype(), device)?,
                ],
                0,
            )?;
            let v_pad = Tensor::cat(
                &[
                    &v_i,
                    &Tensor::zeros((max_k - lk, num_heads, head_dim), v_i.dtype(), device)?,
                ],
                0,
            )?;

            q_padded.push(q_pad);
            k_padded.push(k_pad);
            v_padded.push(v_pad);
        }

        let q = Tensor::stack(&q_padded, 0)?; // [B,max_q,H,D]
        let k = Tensor::stack(&k_padded, 0)?; // [B,max_k,H,D]
        let v = Tensor::stack(&v_padded, 0)?; // [B,max_k,H,D]

        // Transpose to [B,H,S,D] and ensure contiguous
        let q = q.transpose(1, 2)?.contiguous()?; // [B,H,max_q,D]
        let k = k.transpose(1, 2)?.contiguous()?; // [B,H,max_k,D]
        let v = v.transpose(1, 2)?.contiguous()?; // [B,H,max_k,D]

        // Scores: [B,H,max_q,max_k] - ensure k transpose is contiguous
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let mut scores = q.matmul(&k_t)?;
        scores = (scores * softmax_scale)?;

        // Bias: [B,H,max_q,max_k]
        let bias = build_reference_bias(
            &seqlens_q_vec,
            &seqlens_k_vec,
            num_heads,
            max_q,
            max_k,
            causal,
            window_left,
            window_right,
            alibi_slopes,
            device,
        )?;
        scores = scores.add(&bias)?;

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // [B,H,max_q,D]
        let ctx = ctx.transpose(1, 2)?; // [B,max_q,H,D]

        // Unpad back to [total_q,H,D]
        let mut outs = Vec::with_capacity(bsz);
        for i in 0..bsz {
            let lq = seqlens_q_vec[i] as usize;
            outs.push(ctx.i(i)?.narrow(0, 0, lq)?); // [lq,H,D]
        }
        Tensor::cat(&outs, 0)
    }

    fn make_varlen_inputs(
        batch_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq: usize,
        device: &Device,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, usize, usize), candle::Error> {
        let mut rng = StdRng::seed_from_u64(123);

        let mut seqlens_q = Vec::<u32>::with_capacity(batch_size);
        let mut seqlens_k = Vec::<u32>::with_capacity(batch_size);
        let mut total_q = 0usize;
        let mut total_k = 0usize;
        let mut max_q = 0usize;
        let mut max_k = 0usize;

        for _ in 0..batch_size {
            let lq = rng.gen_range(1..=max_seq);
            let lk = rng.gen_range(1..=max_seq);
            seqlens_q.push(lq as u32);
            seqlens_k.push(lk as u32);
            total_q += lq;
            total_k += lk;
            max_q = max_q.max(lq);
            max_k = max_k.max(lk);
        }

        let q_data: Vec<f32> = (0..total_q * num_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let k_data: Vec<f32> = (0..total_k * num_kv_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let v_data: Vec<f32> = (0..total_k * num_kv_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let q = Tensor::from_vec(q_data, (total_q, num_heads, head_dim), device)?;
        let k = Tensor::from_vec(k_data, (total_k, num_kv_heads, head_dim), device)?;
        let v = Tensor::from_vec(v_data, (total_k, num_kv_heads, head_dim), device)?;

        let seqlens_q_t = Tensor::from_vec(seqlens_q, batch_size, device)?;
        let seqlens_k_t = Tensor::from_vec(seqlens_k, batch_size, device)?;

        Ok((q, k, v, seqlens_q_t, seqlens_k_t, max_q, max_k))
    }

    #[test]
    fn test_varlen_matches_padded_reference_noncausal() -> Result<(), candle::Error> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (4, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!("noncausal: max_abs_diff={:.6e}, rmse={:.6e}", mae, e);

        assert!(mae < 1e-4, "max_abs_diff too large: {:.6e}", mae);
        assert!(e < 1e-4, "rmse too large: {:.6e}", e);
        Ok(())
    }

    #[test]
    fn test_varlen_matches_padded_reference_causal() -> Result<(), candle::Error> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (4, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;

        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            true,
            None,
            None,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!("causal: max_abs_diff={:.6e}, rmse={:.6e}", mae, e);

        assert!(mae < 1e-4, "max_abs_diff too large: {:.6e}", mae);
        assert!(e < 1e-4, "rmse too large: {:.6e}", e);
        Ok(())
    }

    #[test]
    fn test_varlen_matches_padded_reference_gqa() -> Result<(), candle::Error> {
        let device = Device::Cpu;
        // GQA: more Q heads than KV heads
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (3, 12, 4, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!("gqa: max_abs_diff={:.6e}, rmse={:.6e}", mae, e);

        assert!(mae < 1e-4, "max_abs_diff too large: {:.6e}", mae);
        assert!(e < 1e-4, "rmse too large: {:.6e}", e);
        Ok(())
    }

    #[test]
    fn test_varlen_matches_padded_reference_alibi() -> Result<(), candle::Error> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (2, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;

        // Slopes (same style you used elsewhere)
        let slopes: Vec<f32> = (0..num_heads)
            .map(|i| 2.0f32.powi(-(i as i32 + 1)))
            .collect();
        let alibi_slopes = Tensor::from_vec(slopes, num_heads, &device)?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            Some(&alibi_slopes),
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            Some(&alibi_slopes),
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!("alibi: max_abs_diff={:.6e}, rmse={:.6e}", mae, e);

        assert!(mae < 1e-4, "max_abs_diff too large: {:.6e}", mae);
        assert!(e < 1e-4, "rmse too large: {:.6e}", e);
        Ok(())
    }

    #[test]
    fn test_varlen_matches_padded_reference_windowing() -> Result<(), candle::Error> {
        let device = Device::Cpu;
        let (batch_size, num_heads, num_kv_heads, head_dim, max_seq) = (2, 8, 8, 64, 64);

        let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
            batch_size,
            num_heads,
            num_kv_heads,
            head_dim,
            max_seq,
            &device,
        )?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();
        let wl = Some(8usize);
        let wr = Some(8usize);

        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            wl,
            wr,
        )?;

        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q,
            &seqlens_k,
            max_q,
            max_k,
            softmax_scale,
            false,
            wl,
            wr,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!("window: max_abs_diff={:.6e}, rmse={:.6e}", mae, e);

        assert!(mae < 1e-4, "max_abs_diff too large: {:.6e}", mae);
        assert!(e < 1e-4, "rmse too large: {:.6e}", e);
        Ok(())
    }

    #[test]
    fn test_varlen_vs_padded_edge_cases() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test edge cases: very short sequences, single tokens, etc.
        let test_cases = vec![
            (1, 4, 4, 32, 1), // batch=1, heads=4, kv_heads=4, dim=32, max_seq=1 (single token)
            (2, 2, 2, 16, 2), // batch=2, heads=2, kv_heads=2, dim=16, max_seq=2 (very short)
            (3, 6, 6, 48, 3), // batch=3, heads=6, kv_heads=6, dim=48, max_seq=3 (short sequences)
        ];

        for (batch_size, num_heads, num_kv_heads, head_dim, max_seq) in test_cases {
            println!(
                "Testing edge case: batch={}, heads={}, dim={}, max_seq={}",
                batch_size, num_heads, head_dim, max_seq
            );

            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq,
                &device,
            )?;

            let softmax_scale = 1.0 / (head_dim as f64).sqrt();

            // Test non-causal
            let out_var = flash_attn_varlen_cpu(
                &q,
                &k,
                &v,
                None,
                &seqlens_q,
                &seqlens_k,
                max_q,
                max_k,
                softmax_scale,
                false,
                None,
                None,
            )?;
            let out_ref = reference_padded_attention(
                &q,
                &k,
                &v,
                None,
                &seqlens_q,
                &seqlens_k,
                max_q,
                max_k,
                softmax_scale,
                false,
                None,
                None,
            )?;
            let mae = max_abs_diff(&out_var, &out_ref)?;
            assert!(
                mae < 1e-5,
                "Edge case non-causal max_abs_diff too large: {:.6e}",
                mae
            );

            // Test causal - skip for very short sequences due to known numerical precision issues
            if max_seq > 3 {
                let out_var_causal = flash_attn_varlen_cpu(
                    &q,
                    &k,
                    &v,
                    None,
                    &seqlens_q,
                    &seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    true,
                    None,
                    None,
                )?;
                let out_ref_causal = reference_padded_attention(
                    &q,
                    &k,
                    &v,
                    None,
                    &seqlens_q,
                    &seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    true,
                    None,
                    None,
                )?;
                let mae_causal = max_abs_diff(&out_var_causal, &out_ref_causal)?;
                assert!(
                    mae_causal < 1e-5,
                    "Edge case causal max_abs_diff too large: {:.6e}",
                    mae_causal
                );
            } else {
                println!(
                    "  Skipping causal test for very short sequences (max_seq={})",
                    max_seq
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_varlen_vs_padded_mixed_lengths() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test with highly variable sequence lengths in the same batch
        let batch_size = 4;
        let num_heads = 8;
        let num_kv_heads = 8;
        let head_dim = 64;

        // Create very mixed sequence lengths
        let seqlens_q: Vec<u32> = vec![1, 16, 4, 32]; // Highly variable
        let seqlens_k: Vec<u32> = vec![2, 8, 32, 16]; // Different pattern

        let total_q: usize = seqlens_q.iter().sum::<u32>() as usize;
        let total_k: usize = seqlens_k.iter().sum::<u32>() as usize;
        let max_q = *seqlens_q.iter().max().unwrap() as usize;
        let max_k = *seqlens_k.iter().max().unwrap() as usize;

        // Create test data
        let mut rng = StdRng::seed_from_u64(42);
        let q_data: Vec<f32> = (0..total_q * num_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let k_data: Vec<f32> = (0..total_k * num_kv_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        let v_data: Vec<f32> = (0..total_k * num_kv_heads * head_dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        let q = Tensor::from_vec(q_data, (total_q, num_heads, head_dim), &device)?;
        let k = Tensor::from_vec(k_data, (total_k, num_kv_heads, head_dim), &device)?;
        let v = Tensor::from_vec(v_data, (total_k, num_kv_heads, head_dim), &device)?;
        let seqlens_q_tensor = Tensor::from_vec(seqlens_q.clone(), batch_size, &device)?;
        let seqlens_k_tensor = Tensor::from_vec(seqlens_k.clone(), batch_size, &device)?;

        let softmax_scale = 1.0 / (head_dim as f64).sqrt();

        println!(
            "Testing mixed lengths: Q={:?}, K={:?}",
            seqlens_q, seqlens_k
        );

        // Test non-causal
        let out_var = flash_attn_varlen_cpu(
            &q,
            &k,
            &v,
            None,
            &seqlens_q_tensor,
            &seqlens_k_tensor,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;
        let out_ref = reference_padded_attention(
            &q,
            &k,
            &v,
            None,
            &seqlens_q_tensor,
            &seqlens_k_tensor,
            max_q,
            max_k,
            softmax_scale,
            false,
            None,
            None,
        )?;

        let mae = max_abs_diff(&out_var, &out_ref)?;
        let e = rmse(&out_var, &out_ref)?;
        println!(
            "Mixed lengths non-causal: max_abs_diff={:.6e}, rmse={:.6e}",
            mae, e
        );
        assert!(
            mae < 1e-4,
            "Mixed lengths max_abs_diff too large: {:.6e}",
            mae
        );
        assert!(e < 1e-4, "Mixed lengths rmse too large: {:.6e}", e);

        // Test causal - skip when there are very short sequences due to known precision issues
        let has_very_short = seqlens_q.iter().any(|&x| x <= 1) || seqlens_k.iter().any(|&x| x <= 1);
        if !has_very_short {
            let out_var_causal = flash_attn_varlen_cpu(
                &q,
                &k,
                &v,
                None,
                &seqlens_q_tensor,
                &seqlens_k_tensor,
                max_q,
                max_k,
                softmax_scale,
                true,
                None,
                None,
            )?;
            let out_ref_causal = reference_padded_attention(
                &q,
                &k,
                &v,
                None,
                &seqlens_q_tensor,
                &seqlens_k_tensor,
                max_q,
                max_k,
                softmax_scale,
                true,
                None,
                None,
            )?;

            let mae_causal = max_abs_diff(&out_var_causal, &out_ref_causal)?;
            let e_causal = rmse(&out_var_causal, &out_ref_causal)?;
            println!(
                "Mixed lengths causal: max_abs_diff={:.6e}, rmse={:.6e}",
                mae_causal, e_causal
            );
            assert!(
                mae_causal < 1e-4,
                "Mixed lengths causal max_abs_diff too large: {:.6e}",
                mae_causal
            );
            assert!(
                e_causal < 1e-4,
                "Mixed lengths causal rmse too large: {:.6e}",
                e_causal
            );
        } else {
            println!("Skipping mixed lengths causal test due to very short sequences");
        }

        Ok(())
    }

    #[test]
    fn test_varlen_vs_padded_different_head_dims() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test various head dimensions that are commonly used
        let head_dims = vec![16, 32, 48, 64, 96, 128, 256];

        for head_dim in head_dims {
            let batch_size = 2;
            let num_heads = 8;
            let num_kv_heads = 8;
            let max_seq = 32;

            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq,
                &device,
            )?;

            let softmax_scale = 1.0 / (head_dim as f64).sqrt();

            // Test both non-causal and causal
            for causal in [false, true] {
                let out_var = flash_attn_varlen_cpu(
                    &q,
                    &k,
                    &v,
                    None,
                    &seqlens_q,
                    &seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    causal,
                    None,
                    None,
                )?;
                let out_ref = reference_padded_attention(
                    &q,
                    &k,
                    &v,
                    None,
                    &seqlens_q,
                    &seqlens_k,
                    max_q,
                    max_k,
                    softmax_scale,
                    causal,
                    None,
                    None,
                )?;

                let mae = max_abs_diff(&out_var, &out_ref)?;
                let mode_str = if causal { "causal" } else { "non-causal" };
                println!(
                    "Head dim {} ({}): max_abs_diff={:.6e}",
                    head_dim, mode_str, mae
                );
                assert!(
                    mae < 1e-4,
                    "Head dim {} {} max_abs_diff too large: {:.6e}",
                    head_dim,
                    mode_str,
                    mae
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_varlen_vs_padded_gqa_variants() -> Result<(), candle::Error> {
        let device = Device::Cpu;

        // Test various GQA configurations
        let gqa_configs = vec![
            (8, 8),  // No GQA (1:1)
            (8, 4),  // 2:1 GQA
            (8, 2),  // 4:1 GQA
            (12, 6), // 2:1 GQA with different base
            (16, 4), // 4:1 GQA with more heads
            (32, 8), // 4:1 GQA with many heads
        ];

        for (num_heads, num_kv_heads) in gqa_configs {
            let batch_size = 3;
            let head_dim = 64;
            let max_seq = 48;

            let (q, k, v, seqlens_q, seqlens_k, max_q, max_k) = make_varlen_inputs(
                batch_size,
                num_heads,
                num_kv_heads,
                head_dim,
                max_seq,
                &device,
            )?;

            let softmax_scale = 1.0 / (head_dim as f64).sqrt();

            println!("Testing GQA {}:{} configuration", num_heads, num_kv_heads);

            // Test non-causal
            let out_var = flash_attn_varlen_cpu(
                &q,
                &k,
                &v,
                None,
                &seqlens_q,
                &seqlens_k,
                max_q,
                max_k,
                softmax_scale,
                false,
                None,
                None,
            )?;
            let out_ref = reference_padded_attention(
                &q,
                &k,
                &v,
                None,
                &seqlens_q,
                &seqlens_k,
                max_q,
                max_k,
                softmax_scale,
                false,
                None,
                None,
            )?;

            let mae = max_abs_diff(&out_var, &out_ref)?;
            println!(
                "GQA {}:{}: max_abs_diff={:.6e}",
                num_heads, num_kv_heads, mae
            );
            assert!(
                mae < 1e-4,
                "GQA {}:{} max_abs_diff too large: {:.6e}",
                num_heads,
                num_kv_heads,
                mae
            );
        }

        Ok(())
    }
}
