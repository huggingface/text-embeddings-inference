<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models.

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on an Nvidia A10 with a sequence
length of 512 tokens:

<p>
  <img src="assets/bs1-lat.png" width="400" />
  <img src="assets/bs1-tp.png" width="400" />
</p>
<p>
  <img src="assets/bs32-lat.png" width="400" />
  <img src="assets/bs32-tp.png" width="400" />
</p>

</div>

## Table of contents

- [Get Started](#get-started)
    - [Supported Models](#supported-models)
    - [Docker](#docker)
    - [Docker Images](#docker-images)
    - [API Documentation](#api-documentation)
    - [Using a private or gated model](#using-a-private-or-gated-model)
    - [Air gapped deployment](#air-gapped-deployment)
    - [Using Re-rankers models](#using-re-rankers-models)
    - [Using Sequence Classification models](#using-sequence-classification-models)
    - [Using SPLADE pooling](#using-splade-pooling)
    - [Distributed Tracing](#distributed-tracing)
    - [gRPC](#grpc)
- [Local Install](#local-install)
- [Docker Build](#docker-build)
    - [Apple M1/M2 Arm](#apple-m1m2-arm64-architectures)
- [Examples](#examples)

Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings and sequence
classification models. TEI enables high-performance extraction for the most popular models, including FlagEmbedding,
Ember, GTE and E5. TEI implements many features such as:

* No model graph compilation step
* Metal support for local execution on Macs
* Small docker images and fast boot times. Get ready for true serverless!
* Token based dynamic batching
* Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
  [Candle](https://github.com/huggingface/candle)
  and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
* [Safetensors](https://github.com/huggingface/safetensors) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

## Get Started

### Supported Models

#### Text Embeddings

Text Embeddings Inference currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT
model with Alibi positions and Mistral, Alibaba GTE, Qwen2 models with Rope positions, and MPNet.

Below are some examples of the currently supported models:

| MTEB Rank | Model Size          | Model Type  | Model ID                                                                                         |
|-----------|---------------------|-------------|--------------------------------------------------------------------------------------------------|
| 1         | 7B (Very Expensive) | Mistral     | [Salesforce/SFR-Embedding-2_R](https://hf.co/Salesforce/SFR-Embedding-2_R)                       |
| 2         | 7B (Very Expensive) | Qwen2       | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-7B-instruct)             |
| 9         | 1.5B (Expensive)    | Qwen2       | [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)         |
| 15        | 0.4B                | Alibaba GTE | [Alibaba-NLP/gte-large-en-v1.5](https://hf.co/Alibaba-NLP/gte-large-en-v1.5)                     |
| 20        | 0.3B                | Bert        | [WhereIsAI/UAE-Large-V1](https://hf.co/WhereIsAI/UAE-Large-V1)                                   |
| 24        | 0.5B                | XLM-RoBERTa | [intfloat/multilingual-e5-large-instruct](https://hf.co/intfloat/multilingual-e5-large-instruct) |
| N/A       | 0.1B                | NomicBert   | [nomic-ai/nomic-embed-text-v1](https://hf.co/nomic-ai/nomic-embed-text-v1)                       |
| N/A       | 0.1B                | NomicBert   | [nomic-ai/nomic-embed-text-v1.5](https://hf.co/nomic-ai/nomic-embed-text-v1.5)                   |
| N/A       | 0.1B                | JinaBERT    | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)             |
| N/A       | 0.1B                | JinaBERT    | [jinaai/jina-embeddings-v2-base-code](https://hf.co/jinaai/jina-embeddings-v2-base-code)         |
| N/A       | 0.1B                | MPNet       | [sentence-transformers/all-mpnet-base-v2](https://hf.co/sentence-transformers/all-mpnet-base-v2)            |

To explore the list of best performing text embeddings models, visit the
[Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

#### Sequence Classification and Re-Ranking

Text Embeddings Inference currently supports CamemBERT, and XLM-RoBERTa Sequence Classification models with absolute positions.

Below are some examples of the currently supported models:

| Task               | Model Type  | Model ID                                                                                                        |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                       |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                         |
| Re-Ranking         | GTE         | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

### Docker

```shell
model=BAAI/bge-large-en-v1.5
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

And then you can make requests like

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

**Note:** To use GPUs, you need to install
the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
NVIDIA drivers on your machine need to be compatible with CUDA version 12.2 or higher.

To see all options to serve your models:

```shell
text-embeddings-router --help
```

```
Usage: text-embeddings-router [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `thenlper/gte-base`.
          Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of
          transformers

          [env: MODEL_ID=]
          [default: thenlper/gte-base]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id
          or a branch like `refs/pr/2`

          [env: REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation.
          Default to the number of CPU cores on the machine

          [env: TOKENIZATION_WORKERS=]

      --dtype <DTYPE>
          The dtype to be forced upon the model

          [env: DTYPE=]
          [possible values: float16, float32]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: POOLING=]

          Possible values:
          - cls:        Select the CLS token as embedding
          - mean:       Apply Mean pooling to the model embeddings
          - splade:     Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings. This option is only
          available if the loaded model is a `ForMaskedLM` Transformer model
          - last-token: Select the last token as embedding

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment.
          Having a low limit will refuse clients requests instead of having them wait for too long and is usually good
          to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory
          overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.

          [env: MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: MAX_BATCH_REQUESTS=]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --auto-truncate
          Automatically truncate inputs that are longer than the maximum supported size

          Unused for gRPC servers

          [env: AUTO_TRUNCATE=]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the
          sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because
          the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be
          encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text
          to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: DEFAULT_PROMPT=]

      --hf-api-token <HF_API_TOKEN>
          Your HuggingFace hub token

          [env: HF_API_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: PORT=]
          [default: 3000]

      --uds-path <UDS_PATH>
          The name of the unix socket some text-embeddings-inference backends will use as they communicate internally
          with gRPC

          [env: UDS_PATH=]
          [default: /tmp/text-embeddings-inference-server]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk
          for instance

          [env: HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization
          header set with the api key as Bearer token.

          [env: API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `text-embeddings-inference.server`

          [env: OTLP_SERVICE_NAME=]
          [default: text-embeddings-inference.server]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          Unused for gRPC servers

          [env: CORS_ALLOW_ORIGIN=]
```

### Docker Images

Text Embeddings Inference ships with multiple Docker images that you can use to target a specific backend:

| Architecture                        | Image                                                                   |
|-------------------------------------|-------------------------------------------------------------------------|
| CPU                                 | ghcr.io/huggingface/text-embeddings-inference:cpu-1.6                   |
| Volta                               | NOT SUPPORTED                                                           |
| Turing (T4, RTX 2000 series, ...)   | ghcr.io/huggingface/text-embeddings-inference:turing-1.6 (experimental) |
| Ampere 80 (A100, A30)               | ghcr.io/huggingface/text-embeddings-inference:1.6                       |
| Ampere 86 (A10, A40, ...)           | ghcr.io/huggingface/text-embeddings-inference:86-1.6                    |
| Ada Lovelace (RTX 4000 series, ...) | ghcr.io/huggingface/text-embeddings-inference:89-1.6                    |
| Hopper (H100)                       | ghcr.io/huggingface/text-embeddings-inference:hopper-1.6 (experimental) |

**Warning**: Flash Attention is turned off by default for the Turing image as it suffers from precision issues.
You can turn Flash Attention v1 ON by using the `USE_FLASH_ATTENTION=True` environment variable.

### API documentation

You can consult the OpenAPI documentation of the `text-embeddings-inference` REST API using the `/docs` route.
The Swagger UI is also available
at: [https://huggingface.github.io/text-embeddings-inference](https://huggingface.github.io/text-embeddings-inference).

### Using a private or gated model

You have the option to utilize the `HF_API_TOKEN` environment variable for configuring the token employed by
`text-embeddings-inference`. This allows you to gain access to protected resources.

For example:

1. Go to https://huggingface.co/settings/tokens
2. Copy your cli READ token
3. Export `HF_API_TOKEN=<your cli READ token>`

or with Docker:

```shell
model=<your private model>
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all -e HF_API_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

### Air gapped deployment

To deploy Text Embeddings Inference in an air-gapped environment, first download the weights and then mount them inside
the container using a volume.

For example:

```shell
# (Optional) create a `models` directory
mkdir models
cd models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5

# Set the models directory as the volume path
volume=$PWD

# Mount the models directory inside the container with a volume and set the model ID
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id /data/gte-base-en-v1.5
```

### Using Re-rankers models

`text-embeddings-inference` v0.4.0 added support for CamemBERT, RoBERTa, XLM-RoBERTa, and GTE Sequence Classification models.
Re-rankers models are Sequence Classification cross-encoders models with a single class that scores the similarity
between a query and a text.

See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

And then you can rank the similarity between a query and a list of texts with:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query": "What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."]}' \
    -H 'Content-Type: application/json'
```

### Using Sequence Classification models

You can also use classic Sequence Classification models like `SamLowe/roberta-base-go_emotions`:

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Using SPLADE pooling

You can choose to activate SPLADE pooling for Bert and Distilbert MaskedLM architectures:

```shell
model=naver/efficient-splade-VI-BT-large-query
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model --pooling splade
```

Once you have deployed the model you can use the `/embed_sparse` endpoint to get the sparse embedding:

```bash
curl 127.0.0.1:8080/embed_sparse \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

### Distributed Tracing

`text-embeddings-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

### gRPC

`text-embeddings-inference` offers a gRPC API as an alternative to the default HTTP API for high performance
deployments. The API protobuf definition can be
found [here](https://github.com/huggingface/text-embeddings-inference/blob/main/proto/tei.proto).

You can use the gRPC API by adding the `-grpc` tag to any TEI Docker image. For example:

```shell
model=BAAI/bge-large-en-v1.5
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6-grpc --model-id $model
```

```shell
grpcurl -d '{"inputs": "What is Deep Learning"}' -plaintext 0.0.0.0:8080 tei.v1.Embed/Embed
```

## Local install

### CPU

You can also opt to install `text-embeddings-inference` locally.

First [install Rust](https://rustup.rs/):

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then run:

```shell
# On x86
cargo install --path router -F mkl
# On M1 or M2
cargo install --path router -F metal
```

You can now launch Text Embeddings Inference on CPU with:

```shell
model=BAAI/bge-large-en-v1.5

text-embeddings-router --model-id $model --port 8080
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### Cuda

GPUs with Cuda compute capabilities < 7.5 are not supported (V100, Titan V, GTX 1000 series, ...).

Make sure you have Cuda and the nvidia drivers installed. NVIDIA drivers on your device need to be compatible with CUDA
version 12.2 or higher.
You also need to add the nvidia binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

Then run:

```shell
# This can take a while as we need to compile a lot of cuda kernels

# On Turing GPUs (T4, RTX 2000 series ... )
cargo install --path router -F candle-cuda-turing -F http --no-default-features

# On Ampere and Hopper
cargo install --path router -F candle-cuda -F http --no-default-features
```

You can now launch Text Embeddings Inference on GPU with:

```shell
model=BAAI/bge-large-en-v1.5

text-embeddings-router --model-id $model --port 8080
```

## Docker build

You can build the CPU container with:

```shell
docker build .
```

To build the Cuda containers, you need to know the compute cap of the GPU you will be using
at runtime.

Then you can build the container with:

```shell
# Example for Turing (T4, RTX 2000 series, ...)
runtime_compute_cap=75

# Example for A100
runtime_compute_cap=80

# Example for A10
runtime_compute_cap=86

# Example for Ada Lovelace (RTX 4000 series, ...)
runtime_compute_cap=89

# Example for H100
runtime_compute_cap=90

docker build . -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=$runtime_compute_cap
```

### Apple M1/M2 arm64 architectures

#### DISCLAIMER

As explained here [MPS-Ready, ARM64 Docker Image](https://github.com/pytorch/pytorch/issues/81224), Metal / MPS is not
supported via Docker. As such inference will be CPU bound and most likely pretty slow when using this docker image on an
M1/M2 ARM CPU.

```
docker build . -f Dockerfile --platform=linux/arm64
```

## Examples

- [Set up an Inference Endpoint with TEI](https://huggingface.co/learn/cookbook/automatic_embedding_tei_inference_endpoints)
- [RAG containers with TEI](https://github.com/plaggy/rag-containers)
