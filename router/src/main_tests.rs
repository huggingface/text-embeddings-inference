#[cfg(test)]
mod tests {
    use crate::Args;
    use clap::Parser;

    #[test]
    fn test_args_default_values() {
        let args = Args::try_parse_from(&["text-embeddings-router"]).unwrap();

        // Test default values
        assert_eq!(args.reranker_mode, "auto");
        assert_eq!(args.max_listwise_docs_per_pass, 125);
        assert_eq!(args.rerank_ordering, "input");
        assert_eq!(args.rerank_rand_seed, None);
        assert_eq!(args.rerank_instruction, None);
        assert_eq!(args.listwise_payload_limit_bytes, 2000000);
        assert_eq!(args.listwise_block_timeout_ms, 30000);
        assert_eq!(args.max_document_length_bytes, 102400);
        assert_eq!(args.max_documents_per_request, 1000);
    }

    #[test]
    fn test_args_parse_reranker_mode() {
        let args = Args::try_parse_from(&["text-embeddings-router", "--reranker-mode", "listwise"])
            .unwrap();
        assert!(args.parse_reranker_mode().is_ok());
        assert_eq!(
            args.parse_reranker_mode().unwrap(),
            text_embeddings_router::strategy::RerankMode::Listwise
        );

        let args = Args::try_parse_from(&["text-embeddings-router", "--reranker-mode", "pairwise"])
            .unwrap();
        assert!(args.parse_reranker_mode().is_ok());
        assert_eq!(
            args.parse_reranker_mode().unwrap(),
            text_embeddings_router::strategy::RerankMode::Pairwise
        );

        let args = Args::try_parse_from(&["text-embeddings-router", "--reranker-mode", "invalid"])
            .unwrap();
        assert!(args.parse_reranker_mode().is_err());
        assert!(args
            .parse_reranker_mode()
            .unwrap_err()
            .to_string()
            .contains("Invalid reranker mode"));
    }

    #[test]
    fn test_args_parse_rerank_ordering() {
        let args = Args::try_parse_from(&["text-embeddings-router", "--rerank-ordering", "random"])
            .unwrap();
        assert!(args.parse_rerank_ordering().is_ok());
        assert_eq!(
            args.parse_rerank_ordering().unwrap(),
            text_embeddings_router::strategy::RerankOrdering::Random
        );

        let args = Args::try_parse_from(&["text-embeddings-router", "--rerank-ordering", "input"])
            .unwrap();
        assert!(args.parse_rerank_ordering().is_ok());
        assert_eq!(
            args.parse_rerank_ordering().unwrap(),
            text_embeddings_router::strategy::RerankOrdering::Input
        );

        let args =
            Args::try_parse_from(&["text-embeddings-router", "--rerank-ordering", "invalid"])
                .unwrap();
        assert!(args.parse_rerank_ordering().is_err());
        assert!(args
            .parse_rerank_ordering()
            .unwrap_err()
            .to_string()
            .contains("Invalid rerank ordering"));
    }

    #[test]
    fn test_args_custom_values() {
        let args = Args::try_parse_from(&[
            "text-embeddings-router",
            "--reranker-mode",
            "listwise",
            "--max-listwise-docs-per-pass",
            "100",
            "--rerank-ordering",
            "random",
            "--rerank-rand-seed",
            "42",
            "--rerank-instruction",
            "Focus on relevance",
            "--listwise-payload-limit-bytes",
            "5000000",
            "--listwise-block-timeout-ms",
            "60000",
            "--max-document-length-bytes",
            "204800",
            "--max-documents-per-request",
            "2000",
        ])
        .unwrap();

        assert_eq!(args.reranker_mode, "listwise");
        assert_eq!(args.max_listwise_docs_per_pass, 100);
        assert_eq!(args.rerank_ordering, "random");
        assert_eq!(args.rerank_rand_seed, Some(42));
        assert_eq!(
            args.rerank_instruction,
            Some("Focus on relevance".to_string())
        );
        assert_eq!(args.listwise_payload_limit_bytes, 5000000);
        assert_eq!(args.listwise_block_timeout_ms, 60000);
        assert_eq!(args.max_document_length_bytes, 204800);
        assert_eq!(args.max_documents_per_request, 2000);
    }

    #[test]
    fn test_args_case_insensitive_parsing() {
        let args =
            Args::try_parse_from(&["text-embeddings-router", "--reranker-mode", "AUTO"]).unwrap();
        assert!(args.parse_reranker_mode().is_ok());
        assert_eq!(
            args.parse_reranker_mode().unwrap(),
            text_embeddings_router::strategy::RerankMode::Auto
        );

        let args = Args::try_parse_from(&["text-embeddings-router", "--rerank-ordering", "INPUT"])
            .unwrap();
        assert!(args.parse_rerank_ordering().is_ok());
        assert_eq!(
            args.parse_rerank_ordering().unwrap(),
            text_embeddings_router::strategy::RerankOrdering::Input
        );
    }
}
