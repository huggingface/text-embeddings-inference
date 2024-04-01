integration-tests:
	cargo test --release

cuda-integration-tests:
	cargo test -F text-embeddings-backend-candle/cuda -F text-embeddings-backend-candle/flash-attn -F text-embeddings-router/candle-cuda --release

integration-tests-review:
	cargo insta test --review --release

cuda-integration-tests-review:
	cargo insta test --review --features "text-embeddings-backend-candle/cuda text-embeddings-backend-candle/flash-attn text-embeddings-router/candle-cuda" --release
