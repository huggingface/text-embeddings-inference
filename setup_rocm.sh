#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$REPO_ROOT/backends/python/server"

# ── 1. Rust ────────────────────────────────────────────────────────────────
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# ── 2. Python deps (no torch — already in the ROCm image) ──────────────────
pip install --no-deps -r "$SERVER_DIR/requirements-rocm.txt"
pip install safetensors opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc grpcio-reflection \
    grpc-interceptor einops packaging

# ── 3. Protobuf stubs ──────────────────────────────────────────────────────
pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 types-protobuf

mkdir -p "$SERVER_DIR/text_embeddings_server/pb"

python -m grpc_tools.protoc \
    -I"$REPO_ROOT/backends/proto" \
    --python_out="$SERVER_DIR/text_embeddings_server/pb" \
    --grpc_python_out="$SERVER_DIR/text_embeddings_server/pb" \
    --mypy_out="$SERVER_DIR/text_embeddings_server/pb" \
    "$REPO_ROOT/backends/proto/embed.proto"

find "$SERVER_DIR/text_embeddings_server/pb/" -name "*.py" \
    -exec sed -i 's/^\(import.*pb2\)/from . \1/g' {} \;

touch "$SERVER_DIR/text_embeddings_server/pb/__init__.py"

# ── 4. Install Python server package ──────────────────────────────────────
pip install -e "$SERVER_DIR"

# ── 5. Build Rust router (Python backend only, no candle/CUDA deps) ────────
cd "$REPO_ROOT"
cargo build --release \
    --no-default-features \
    --features python,http \
    --bin text-embeddings-router

echo ""
echo "✓ Done. Run with (or change the model ID):"
echo "  ./target/release/text-embeddings-router --model-id BAAI/bge-base-en-v1.5 --dtype bfloat16"
