FROM lukemathwalker/cargo-chef:latest-rust-1.75-bookworm AS chef

WORKDIR /usr/src

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

FROM chef AS planner

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM chef AS builder

ARG GIT_SHA
ARG DOCKER_LABEL

# sccache specific variables
ARG ACTIONS_CACHE_URL
ARG ACTIONS_RUNTIME_TOKEN
ARG SCCACHE_GHA_ENABLED

RUN echo "int mkl_serv_intel_cpu_true() {return 1;}" > fakeintel.c && \
    gcc -shared -fPIC -o libfakeintel.so fakeintel.c

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --features candle --no-default-features --recipe-path recipe.json && sccache -s

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

FROM builder as http-builder

RUN cargo build --release --bin text-embeddings-router -F candle -F http --no-default-features && sccache -s

FROM builder as grpc-builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY proto proto

RUN cargo build --release --bin text-embeddings-router -F grpc -F candle --no-default-features && sccache -s

FROM debian:bookworm-slim as base

COPY --from=builder /usr/src/libfakeintel.so /usr/local/libfakeintel.so

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    MKL_ENABLE_INSTRUCTIONS=AVX512_E4 \
    RAYON_NUM_THREADS=8 \
    LD_PRELOAD=/usr/local/libfakeintel.so \
    LD_LIBRARY_PATH=/usr/local/lib

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*


FROM base as grpc

COPY --from=grpc-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]

FROM base

COPY --from=http-builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]
