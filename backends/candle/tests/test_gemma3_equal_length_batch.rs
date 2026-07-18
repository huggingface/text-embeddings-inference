//! Regression test for the Gemma3 equal-length-batch attention-mask bug.
//!
//! Gemma3 attention layers add their causal or sliding-window mask only when the model
//! passes an attention bias tensor. Before the fix, an equal-length multi-sequence
//! batch needed no padding, so `Gemma3Model::forward` passed `None` and every layer
//! skipped its attention mask. This test uses tiny random weights and verifies that
//! equal-length batched inference matches single-sequence inference.

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
const LAYERS: usize = 16;
const VOCAB: usize = 256;

fn w(rows: usize, cols: usize, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 0.1f32, (rows, cols), dev)?)
}

fn norm(n: usize, dev: &Device) -> Result<Tensor> {
    Ok(Tensor::randn(0f32, 0.05f32, (n,), dev)?)
}

fn write_random_gemma3(dir: &std::path::Path) -> Result<()> {
    let dev = Device::Cpu;
    let mut t: HashMap<String, Tensor> = HashMap::new();
    t.insert(
        "embed_tokens.weight".to_string(),
        Tensor::randn(0f32, 1f32, (VOCAB, H), &dev)?,
    );
    for l in 0..LAYERS {
        let p = format!("layers.{l}.");
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
        t.insert(
            format!("{p}pre_feedforward_layernorm.weight"),
            norm(H, &dev)?,
        );
        t.insert(
            format!("{p}post_feedforward_layernorm.weight"),
            norm(H, &dev)?,
        );
    }
    t.insert("norm.weight".to_string(), norm(H, &dev)?);
    candle::safetensors::save(&t, dir.join("model.safetensors"))?;

    let config = format!(
        r#"{{
  "model_type": "gemma3_text",
  "attention_bias": false,
  "pad_token_id": 0,
  "vocab_size": {VOCAB},
  "head_dim": {HEAD_DIM},
  "hidden_activation": "silu",
  "hidden_size": {H},
  "intermediate_size": {INTER},
  "max_position_embeddings": 512,
  "num_hidden_layers": {LAYERS},
  "num_attention_heads": {HEADS},
  "num_key_value_heads": {KV},
  "query_pre_attn_scalar": {HEAD_DIM},
  "rms_norm_eps": 1e-6,
  "rope_local_base_freq": 10000.0,
  "rope_theta": 1000000.0,
  "sliding_window": 4,
  "_sliding_window_pattern": 2,
  "use_bidirectional_attention": false
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
fn test_gemma3_equal_length_batch_matches_single() -> Result<()> {
    let dir = std::env::temp_dir().join(format!("tei_gemma3_rand_{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let res = run(&dir);
    std::fs::remove_dir_all(&dir).ok();
    res
}

fn run(dir: &std::path::Path) -> Result<()> {
    write_random_gemma3(dir)?;

    let backend = CandleBackend::new(
        dir,
        "float32".to_string(),
        ModelType::Embedding(Pool::Mean),
        None,
    )?;

    let a: Vec<u32> = vec![5, 9, 13, 17, 21, 25, 29, 33];
    let b: Vec<u32> = vec![6, 10, 14, 18, 22, 26, 30, 34];
    assert_eq!(a.len(), b.len(), "sequences must be equal length");

    let single_a = pooled(&backend.embed(single(&a))?, 0);
    let single_b = pooled(&backend.embed(single(&b))?, 0);

    let pair = Batch {
        input_ids: [a.clone(), b.clone()].concat(),
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
    let pair_a = pooled(&pair_emb, 0);
    let pair_b = pooled(&pair_emb, 1);

    let cos_a = cosine(&pair_a, &single_a);
    let cos_b = cosine(&pair_b, &single_b);
    eprintln!("Gemma3 equal-length batch vs single: cos_a={cos_a:.6}, cos_b={cos_b:.6}");

    assert!(
        cos_a > 0.99,
        "equal-length Gemma3 batch[0] diverged from single inference (cos={cos_a}); attention mask skipped for no-padding batch?"
    );
    assert!(
        cos_b > 0.99,
        "equal-length Gemma3 batch[1] diverged from single inference (cos={cos_b}); attention mask skipped for no-padding batch?"
    );
    Ok(())
}
