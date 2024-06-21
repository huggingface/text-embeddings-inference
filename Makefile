integration-tests:
	cargo test --profile release-debug

cuda-integration-tests:
	cargo test -F text-embeddings-backend-candle/cuda -F text-embeddings-backend-candle/flash-attn -F text-embeddings-router/candle-cuda --profile release-debug

integration-tests-review:
	cargo insta test --review --profile release-debug

cuda-integration-tests-review:
	cargo insta test --review --features "text-embeddings-backend-candle/cuda text-embeddings-backend-candle/flash-attn text-embeddings-router/candle-cuda" --profile release-debug
