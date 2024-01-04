// coding=utf-8
// Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
// Copyright (c) 2023 Jina AI GmbH. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
use candle::{DType, Device, Result, Tensor};

fn get_slopes_power_of_2(n: usize) -> Vec<f64> {
    let start: f64 = 2_f64.powf(-(2_f64.powf(-((n as f64).log2() - 3_f64))));

    (0..n).map(|i| start * start.powi(i as i32)).collect()
}

pub fn alibi_head_slopes(num_attention_heads: usize) -> Vec<f64> {
    if (num_attention_heads as f64).log2().fract() == 0.0 {
        // `num_attention_heads` is a power of 2
        get_slopes_power_of_2(num_attention_heads)
    } else {
        let closest_power_of_2 =
            2_f64.powi((num_attention_heads as f64).log2().floor() as i32) as usize;

        let mut slopes = get_slopes_power_of_2(closest_power_of_2);
        let additional_slopes: Vec<f64> = get_slopes_power_of_2(2 * closest_power_of_2)
            .into_iter()
            .enumerate()
            // Filter odd indices
            .filter(|(i, _)| i % 2 == 0)
            // Remove i
            .map(|(_, v)| v)
            .collect();

        // Extend slopes
        slopes.extend_from_slice(&additional_slopes[0..(num_attention_heads - closest_power_of_2)]);

        slopes
    }
}

pub fn build_alibi_tensor(
    num_positions: usize,
    num_heads: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let context_positions =
        Tensor::arange(0.0, num_positions as f64, &Device::Cpu)?.unsqueeze(1)?;
    let memory_positions = Tensor::arange(0.0, num_positions as f64, &Device::Cpu)?.unsqueeze(0)?;

    let relative_positions = memory_positions.broadcast_sub(&context_positions)?.abs()?;
    // [num_heads, num_positions, num_positions]
    let relative_positions =
        relative_positions
            .unsqueeze(0)?
            .expand((num_heads, num_positions, num_positions))?;

    // [num_heads, 1, 1]
    let slopes = (Tensor::from_vec(
        alibi_head_slopes(num_heads),
        (num_heads, 1, 1),
        &Device::Cpu,
    )? * -1_f64)?;

    // [num_heads, num_positions, num_positions]
    let alibi = relative_positions.broadcast_mul(&slopes)?;

    alibi
        .reshape((1, num_heads, num_positions, num_positions))?
        .to_dtype(dtype)?
        .to_device(device)
}
