use super::*;

// The synthetic graphs cast input_ids and attention_mask to f32 and concatenate them
// along the last axis. The token outputs contain [id, mask] per token. The sentence
// output sums both channels across the sequence, so it differs from mean pooling.
const DUAL_OUTPUTS: &[u8] = include_bytes!("../tests/fixtures/dual.onnx");
const POOLED_OUTPUT: &[u8] = include_bytes!("../tests/fixtures/pooled.onnx");
const TOKEN_OUTPUT: &[u8] = include_bytes!("../tests/fixtures/token-only.onnx");

fn backend(graph: &[u8], padding_side: PaddingSide, pool: Pool) -> OrtBackend {
    let session = Session::builder()
        .unwrap()
        .with_intra_threads(1)
        .unwrap()
        .commit_from_memory(graph)
        .unwrap();
    OrtBackend {
        session: Mutex::new(session),
        config: Config {
            pad_token_id: Some(0),
            eos_token_id: None,
            model_type: None,
            hidden_size: 2,
            num_hidden_layers: 1,
            num_key_value_heads: None,
        },
        tokenizer_config: TokenizerConfig { padding_side },
        token_type_ids: false,
        token_type_ids_key: String::from("token_type_ids"),
        position_ids: false,
        past_key_values: false,
        pool,
    }
}

fn batch(pooled_indices: Vec<u32>, raw_indices: Vec<u32>) -> Batch {
    Batch {
        input_ids: vec![2, 3, 4, 4, 5, 2],
        token_type_ids: vec![0; 6],
        position_ids: vec![0, 1, 2, 0, 1, 0],
        cumulative_seq_lengths: vec![0, 3, 5, 6],
        max_length: 3,
        pooled_indices,
        raw_indices,
    }
}

fn assert_raw(embeddings: &Embeddings, row: usize, ids: &[f32]) {
    let expected: Vec<_> = ids.iter().map(|&id| vec![id, 1.0]).collect();
    let Some(Embedding::All(actual)) = embeddings.get(&row) else {
        panic!("Expected raw token embeddings for row {row}");
    };
    assert_eq!(actual, &expected);
}

fn assert_pooled(embeddings: &Embeddings, row: usize, expected: &[f32]) {
    let Some(Embedding::Pooled(actual)) = embeddings.get(&row) else {
        panic!("Expected pooled embedding for row {row}");
    };
    assert_eq!(actual, expected);
}

#[test]
fn sentence_and_token_outputs_serve_the_requested_rows() {
    for padding_side in [PaddingSide::Left, PaddingSide::Right] {
        let backend = backend(DUAL_OUTPUTS, padding_side, Pool::Mean);
        assert!(backend.supports_token_embeddings());

        let mixed = backend.embed(batch(vec![1], vec![0, 2])).unwrap();
        assert_eq!(mixed.len(), 3);
        assert_pooled(&mixed, 1, &[9.0, 2.0]);
        assert_raw(&mixed, 0, &[2.0, 3.0, 4.0]);
        assert_raw(&mixed, 2, &[2.0]);

        let raw = backend.embed(batch(vec![], vec![0, 1, 2])).unwrap();
        assert_raw(&raw, 1, &[4.0, 5.0]);
        assert_raw(&raw, 2, &[2.0]);
    }
}

#[test]
fn sentence_only_output_cannot_serve_raw_embeddings() {
    let backend = backend(POOLED_OUTPUT, PaddingSide::Right, Pool::Mean);
    assert!(!backend.supports_token_embeddings());
    let pooled = backend.embed(batch(vec![0, 1, 2], vec![])).unwrap();
    assert_pooled(&pooled, 0, &[9.0, 3.0]);
    assert_pooled(&pooled, 1, &[9.0, 2.0]);
    assert_pooled(&pooled, 2, &[2.0, 1.0]);
    assert!(matches!(
        backend.embed(batch(vec![], vec![0, 1, 2])),
        Err(BackendError::Inference(message)) if message.contains("does not provide token embeddings")
    ));
}

#[test]
fn pooled_output_can_have_singleton_dimensions() {
    let values = ndarray::array![[[9.0, 3.0]], [[9.0, 2.0]], [[2.0, 1.0]]];
    let mut embeddings = Embeddings::default();
    insert_pooled_embeddings(
        values.view().into_dyn(),
        &batch(vec![1], vec![0, 2]),
        &mut embeddings,
    )
    .unwrap();
    assert_eq!(embeddings.len(), 1);
    assert_pooled(&embeddings, 1, &[9.0, 2.0]);
}

#[test]
fn token_only_output_can_pool_mixed_batches() {
    for (padding_side, pool, expected) in [
        (PaddingSide::Left, Pool::Cls, [4.0, 1.0]),
        (PaddingSide::Right, Pool::LastToken, [5.0, 1.0]),
        (PaddingSide::Left, Pool::Mean, [4.5, 1.0]),
        (PaddingSide::Right, Pool::Mean, [4.5, 1.0]),
    ] {
        let backend = backend(TOKEN_OUTPUT, padding_side, pool);
        let mixed = backend.embed(batch(vec![1], vec![0, 2])).unwrap();
        assert_pooled(&mixed, 1, &expected);
        assert_raw(&mixed, 0, &[2.0, 3.0, 4.0]);
        assert_raw(&mixed, 2, &[2.0]);
    }
}
