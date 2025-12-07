# Dockerfile for TEI with Python backend and CUDA support
# Supports: L40s (sm_89), RTX 3090 (sm_86)

# =============================================================================
# Stage 1: Rust Builder
# =============================================================================
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS rust-builder

ENV SCCACHE=0.10.0
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"
ENV CARGO_CHEF=0.1.71

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --version $CARGO_CHEF --locked

# =============================================================================
# Stage 2: Recipe Planner
# =============================================================================
FROM rust-builder AS planner

WORKDIR /usr/src

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare --recipe-path recipe.json

# =============================================================================
# Stage 3: Dependency Builder
# =============================================================================
FROM rust-builder AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

WORKDIR /usr/src

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --features python --features http --recipe-path recipe.json && sccache -s

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo build --release --bin text-embeddings-router -F python -F http --no-default-features && sccache -s

# =============================================================================
# Stage 4: Python Environment
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS python-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /opt/server

COPY backends/proto /opt/proto
COPY backends/python/server /opt/server

RUN pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 'types-protobuf' --no-cache-dir && \
    mkdir -p text_embeddings_server/pb && \
    python -m grpc_tools.protoc -I/opt/proto --python_out=text_embeddings_server/pb \
        --grpc_python_out=text_embeddings_server/pb --mypy_out=text_embeddings_server/pb /opt/proto/embed.proto && \
    find text_embeddings_server/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \; && \
    touch text_embeddings_server/pb/__init__.py

RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir .

# =============================================================================
# Stage 5: Final Image
# =============================================================================
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HUGGINGFACE_HUB_CACHE=/data
ENV PORT=80
ENV TQDM_DISABLE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

COPY --from=python-builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=python-builder /usr/local/bin/python-text-embeddings-server /usr/local/bin/python-text-embeddings-server
COPY --from=python-builder /opt/server /opt/server

COPY --from=builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENV PATH="/usr/local/bin:${PATH}"
ENV PYTHONPATH="/opt/server:${PYTHONPATH}"

# Download spacy model in final image (ensures it's available at runtime)
# This is needed because spacy models may not be fully copied from builder stage
RUN pip install --no-cache-dir spacy>=3.7.0 && \
    python -m spacy download xx_sent_ud_sm && \
    python -c "import spacy; spacy.load('xx_sent_ud_sm')" && \
    echo "Spacy model verified successfully"

WORKDIR /opt/server

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
