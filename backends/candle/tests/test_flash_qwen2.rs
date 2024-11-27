#![allow(dead_code, unused_imports)]

mod common;

use crate::common::{sort_embeddings, SnapshotEmbeddings};
use anyhow::Result;
use common::{batch, cosine_matcher, download_artifacts, load_tokenizer};
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, ModelType, Pool};
use tokenizers::processors::sequence::Sequence;
use tokenizers::processors::template::TemplateProcessing;
use tokenizers::{PostProcessorWrapper, Tokenizer};

#[test]
#[serial_test::serial]
#[cfg(all(feature = "cuda", feature = "flash-attn"))]
fn test_flash_qwen2() -> Result<()> {
    let model_root = download_artifacts("Alibaba-NLP/gte-Qwen2-1.5B-instruct", None)?;
    let mut tokenizer = load_tokenizer(&model_root)?;
    // Qwen2 updates the post processor manually instead of into the tokenizer.json...
    // https://huggingface.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct/blob/main/tokenization_qwen.py#L246
    let template = TemplateProcessing::builder()
        .try_single("$A:0 <|endoftext|>:0")
        .unwrap()
        .try_pair("$A:0 <|endoftext|>:0 $B:1 <|endoftext|>:1")
        .unwrap()
        .special_tokens(vec![("<|endoftext|>", 151643)])
        .build()
        .unwrap();
    match tokenizer.get_post_processor() {
        None => tokenizer.with_post_processor(template),
        Some(post_processor) => {
            let post_processor = Sequence::new(vec![
                post_processor.clone(),
                PostProcessorWrapper::Template(template),
            ]);
            tokenizer.with_post_processor(post_processor)
        }
    };

    let backend = CandleBackend::new(
        &model_root,
        "float16".to_string(),
        ModelType::Embedding(Pool::LastToken),
    )?;

    let input_batch = batch(
        vec![
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
            tokenizer.encode("Deep Learning is...", true).unwrap(),
            tokenizer.encode("What is Deep Learning?", true).unwrap(),
        ],
        [0, 1, 2].to_vec(),
        vec![],
    );

    let matcher = cosine_matcher();

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_batch)?);
    let embeddings_batch = SnapshotEmbeddings::from(pooled_embeddings);
    insta::assert_yaml_snapshot!("qwen2_batch", embeddings_batch, &matcher);

    let input_single = batch(
        vec![tokenizer.encode("What is Deep Learning?", true).unwrap()],
        [0].to_vec(),
        vec![],
    );

    let (pooled_embeddings, _) = sort_embeddings(backend.embed(input_single)?);
    let embeddings_single = SnapshotEmbeddings::from(pooled_embeddings);

    insta::assert_yaml_snapshot!("qwen2_single", embeddings_single, &matcher);
    assert_eq!(embeddings_batch[0], embeddings_single[0]);
    assert_eq!(embeddings_batch[2], embeddings_single[0]);

    Ok(())
}
