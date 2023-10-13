FROM lukemathwalker/cargo-chef:latest-rust-1.73-bookworm AS chef
WORKDIR /usr/src

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

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
| gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null && \
  echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | \
  tee /etc/apt/sources.list.d/oneAPI.list

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    intel-oneapi-mkl-devel \
    && rm -rf /var/lib/apt/lists/*

COPY --from=planner /usr/src/recipe.json recipe.json

RUN cargo chef cook --release --features candle --features mkl --no-default-features --recipe-path recipe.json

COPY backends backends
COPY core core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

RUN cargo build --release --bin text-embeddings-router -F candle -F mkl --no-default-features

FROM debian:bookworm-slim

ENV HUGGINGFACE_HUB_CACHE=/data \
    PORT=80

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libomp-dev \
    ca-certificates \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/src/target/release/text-embeddings-router /usr/local/bin/text-embeddings-router

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]