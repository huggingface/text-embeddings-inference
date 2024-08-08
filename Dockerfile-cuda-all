FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS base-builder

ENV SCCACHE=0.5.4
ENV RUSTC_WRAPPER=/usr/local/bin/sccache
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Donwload and configure sccache
RUN curl -fsSL https://github.com/mozilla/sccache/releases/download/v$SCCACHE/sccache-v$SCCACHE-x86_64-unknown-linux-musl.tar.gz | tar -xzv --strip-components=1 -C /usr/local/bin sccache-v$SCCACHE-x86_64-unknown-linux-musl/sccache && \
    chmod +x /usr/local/bin/sccache

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN cargo install cargo-chef --locked

FROM base-builder AS planner

WORKDIR /usr/src

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo chef prepare  --recipe-path recipe.json

FROM base-builder AS builder

ARG GIT_SHA
ARG DOCKER_LABEL
ARG VERTEX="false"

# sccache specific variables
ARG ACTIONS_CACHE_URL
ARG ACTIONS_RUNTIME_TOKEN
ARG SCCACHE_GHA_ENABLED

# Limit parallelism
ARG RAYON_NUM_THREADS=4
ARG CARGO_BUILD_JOBS
ARG CARGO_BUILD_INCREMENTAL

WORKDIR /usr/src

COPY --from=planner /usr/src/recipe.json recipe.json

RUN if [ $VERTEX = "true" ]; \
    then \
      cargo chef cook --release --features google --recipe-path recipe.json && sccache -s; \
    else \
      cargo chef cook --release --recipe-path recipe.json && sccache -s; \
    fi;

RUN if [ $VERTEX = "true" ]; \
    then \
      CUDA_COMPUTE_CAP=75 cargo chef cook --release --features google --features candle-cuda-turing --recipe-path recipe.json && sccache -s; \
    else \
      CUDA_COMPUTE_CAP=75 cargo chef cook --release --features candle-cuda-turing --recipe-path recipe.json && sccache -s; \
    fi;

RUN if [ $VERTEX = "true" ]; \
    then \
      CUDA_COMPUTE_CAP=80 cargo chef cook --release --features google --features candle-cuda --recipe-path recipe.json && sccache -s; \
    else \
      CUDA_COMPUTE_CAP=80 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s; \
    fi;

RUN if [ $VERTEX = "true" ]; \
    then \
      CUDA_COMPUTE_CAP=90 cargo chef cook --release --features google --features candle-cuda --recipe-path recipe.json && sccache -s; \
    else \
      CUDA_COMPUTE_CAP=90 cargo chef cook --release --features candle-cuda --recipe-path recipe.json && sccache -s; \
    fi;

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN if [ $VERTEX = "true" ]; \
    then \
        CUDA_COMPUTE_CAP=75 cargo build --release --bin text-embeddings-router -F candle-cuda-turing -F google  && sccache -s; \
    else \
        CUDA_COMPUTE_CAP=75 cargo build --release --bin text-embeddings-router -F candle-cuda-turing && sccache -s; \
    fi;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-75

RUN if [ $VERTEX = "true" ]; \
    then \
        CUDA_COMPUTE_CAP=80 cargo build --release --bin text-embeddings-router -F candle-cuda -F google  && sccache -s; \
    else \
        CUDA_COMPUTE_CAP=80 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s; \
    fi;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-80

RUN if [ $VERTEX = "true" ]; \
    then \
        CUDA_COMPUTE_CAP=90 cargo build --release --bin text-embeddings-router -F candle-cuda -F google  && sccache -s; \
    else \
        CUDA_COMPUTE_CAP=90 cargo build --release --bin text-embeddings-router -F candle-cuda && sccache -s; \
    fi;

RUN mv /usr/src/target/release/text-embeddings-router /usr/src/target/release/text-embeddings-router-90

FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS base

ARG DEFAULT_USE_FLASH_ATTENTION=True

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80 \
    USE_FLASH_ATTENTION=$DEFAULT_USE_FLASH_ATTENTION

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/target/release/text-embeddings-router-75 /usr/local/bin/text-embeddings-router-75
COPY --from=builder /usr/src/target/release/text-embeddings-router-80 /usr/local/bin/text-embeddings-router-80
COPY --from=builder /usr/src/target/release/text-embeddings-router-90 /usr/local/bin/text-embeddings-router-90

# Amazon SageMaker compatible image
FROM base AS sagemaker

COPY --chmod=775 sagemaker-entrypoint-cuda-all.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Default image
FROM base

COPY --chmod=775 cuda-all-entrypoint.sh entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["--json-output"]
