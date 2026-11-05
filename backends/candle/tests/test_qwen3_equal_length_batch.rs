//! Regression test for the equal-length-batch causal-mask bug.
//!
//! A causal model must apply the causal attention mask for EVERY multi-sequence batch,
//! even when all sequences are the same length (so the batch needs no padding). Before
//! the fix, `Qwen3Model::forward` only built the causal mask when padding was present;
//! an equal-length batch left `attention_bias = None` and silently ran *bidirectional*
//! attention, so the same text embedded differently in a batch than on its own.
//!
//! This test uses tiny RANDOM weights (no model download). The batch-vs-single
//! divergence is weight-independent and compounds with depth, so a handful of layers
//! makes it unmistakable: on the buggy code the equal-length batch diverges from single
//! inference (cos well below 1.0); with the fix it matches (cos ~1.0).

use std::collections::HashMap;

use anyhow::Result;
use candle::{Device, Tensor};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, Embeddings, ModelType, Pool};

const H: usize = 64;
const HEADS: usize = 4;
const HEAD_DIM: usize = 16;
const KV: usize = 2;
const INTER: usize = 128;
const LAYERS: usize = 24; // deep enough that the bidirectional leak is unmistakable
const VOCAB: usize = 256;

fn w(rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 0.1f32, (rows, cols), dev)?)
}
fn norm(n: usize, dev: &Device) -> Result<Tensor> {
    Ok((Tensor::randn(0f32, 0.05f32, (n,), dev)? + 1.0)?)
}

/// Write a small, architecture-faithful, random-weight Qwen3 model to `dir`.
fn write_random_qwen3(dir: &std::path::Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "model.embed_tokens.weight".to_string(),
        Tensor::randn(0f32, 1f32, (VOCAB, H), &dev)?,
    );
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}.");
        t.insert(
            format!("{p}self_attn.q_proj.weight"),
            w(HEADS * HEAD_DIM, H, &dev)?,
        );
        t.insert(
            format!("{p}self_attn.k_proj.weight"),
            w(KV * HEAD_DIM, H, &dev)?,
        );
        t.insert(
            format!("{p}self_attn.v_proj.weight"),
            w(KV * HEAD_DIM, H, &dev)?,
        );
        t.insert(
            format!("{p}self_attn.o_proj.weight"),
            w(H, HEADS * HEAD_DIM, &dev)?,
        );
        t.insert(format!("{p}self_attn.q_norm.weight"), norm(HEAD_DIM, &dev)?);
        t.insert(format!("{p}self_attn.k_norm.weight"), norm(HEAD_DIM, &dev)?);
        t.insert(format!("{p}mlp.gate_proj.weight"), w(INTER, H, &dev)?);
        t.insert(format!("{p}mlp.up_proj.weight"), w(INTER, H, &dev)?);
        t.insert(format!("{p}mlp.down_proj.weight"), w(H, INTER, &dev)?);
        t.insert(format!("{p}input_layernorm.weight"), norm(H, &dev)?);
        t.insert(
            format!("{p}post_attention_layernorm.weight"),
            norm(H, &dev)?,
        );
    }
    t.insert("model.norm.weight".to_string(), norm(H, &dev)?);
    candle::safetensors::save(&t, dir.join("model.safetensors"))?;

    let config = format!(
        r#"{{
  "model_type": "qwen3",
  "attention_bias": false,
  "vocab_size": {VOCAB},
  "head_dim": {HEAD_DIM},
  "hidden_size": {H},
  "intermediate_size": {INTER},
  "num_hidden_layers": {LAYERS},
  "num_attention_heads": {HEADS},
  "num_key_value_heads": {KV},
  "hidden_act": "silu",
  "max_position_embeddings": 512,
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0,
  "use_sliding_window": false,
  "eos_token_id": 0
}}"#
    );
    std::fs::write(dir.join("config.json"), config)?;
    Ok(())
}

fn pooled(embeddings: &Embeddings, i: usize) -> Vec<f32> {
    match embeddings.get(&i).expect("missing embedding") {
        Embedding::Pooled(e) => e.clone(),
        Embedding::All(_) => panic!("expected pooled embedding"),
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn single(ids: &[u32]) -> Batch {
    Batch {
        input_ids: ids.to_vec(),
        token_type_ids: vec![0; ids.len()],
        position_ids: (0..ids.len() as u32).collect(),
        cumulative_seq_lengths: vec![0, ids.len() as u32],
        max_length: ids.len() as u32,
        pooled_indices: vec![0],
        raw_indices: vec![],
    }
}

#[test]
#[serial_test::serial]
fn test_qwen3_batch_matches_single() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("tei_qwen3_rand_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let res = run(&dir);
    std::fs::remove_dir_all(&dir).ok();
    res
}

fn run(dir: &std::path::Path) -> Result<()> {
    write_random_qwen3(dir)?;

    let backend = CandleBackend::new(
        dir,
        "float32".to_string(),
        ModelType::Embedding(Pool::LastToken),
        None,
    )?;

    // Two DIFFERENT sequences of the SAME length. Batched together they need no
    // padding, which is exactly the case that used to skip the causal mask.
    let a: Vec<u32> = vec![5, 9, 13, 17, 21, 25, 29, 33];
    let b: Vec<u32> = vec![6, 10, 14, 18, 22, 26, 30, 34];
    assert_eq!(a.len(), b.len(), "sequences must be equal length");

    let single_a = pooled(&backend.embed(single(&a))?, 0);
    for b in [&b[..], &b[..5]] {
        let single_b = pooled(&backend.embed(single(b))?, 0);
        let pair = Batch {
            input_ids: [a.as_slice(), b].concat(),
            token_type_ids: vec![0; a.len() + b.len()],
            position_ids: [
                (0..a.len() as u32).collect::<Vec<_>>(),
                (0..b.len() as u32).collect::<Vec<_>>(),
            ]
            .concat(),
            cumulative_seq_lengths: vec![0, a.len() as u32, (a.len() + b.len()) as u32],
            max_length: a.len() as u32,
            pooled_indices: vec![0, 1],
            raw_indices: vec![],
        };
        let pair_emb = backend.embed(pair)?;
        for (index, expected) in [(0, &single_a), (1, &single_b)] {
            let actual = pooled(&pair_emb, index);
            assert_eq!(actual.len(), expected.len());
            assert!(actual.iter().chain(expected).all(|value| value.is_finite()));
            let similarity = cosine(&actual, expected);
            assert!(
                similarity > 0.9999,
                "batch[{index}], length={}: cosine={similarity}",
                b.len()
            );
            for (actual, expected) in actual.iter().zip(expected) {
                assert!(
                    (actual - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                    "batch[{index}], length={}: actual={actual}, expected={expected}",
                    b.len()
                );
            }
        }
    }
    Ok(())
}
