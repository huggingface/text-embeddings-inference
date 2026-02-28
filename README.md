<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models.

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on an NVIDIA A10 with a sequence
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
    - [Apple Silicon (Homebrew)](#apple-silicon-homebrew)
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
* [ONNX](https://github.com/onnx/onnx) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

## Get Started

### Supported Models

#### Text Embeddings

Text Embeddings Inference currently supports Nomic, BERT, CamemBERT, XLM-RoBERTa models with absolute positions, JinaBERT
model with Alibi positions and Mistral, Alibaba GTE, Qwen2 models with Rope positions, MPNet, ModernBERT, Qwen3, and Gemma3.

Below are some examples of the currently supported models:

| MTEB Rank | Model Size             | Model Type     | Model ID                                                                                         |
|-----------|------------------------|----------------|--------------------------------------------------------------------------------------------------|
| 2         | 7.57B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-8B](https://hf.co/Qwen/Qwen3-Embedding-8B)                                 |
| 3         | 4.02B (Very Expensive) | Qwen3          | [Qwen/Qwen3-Embedding-4B](https://hf.co/Qwen/Qwen3-Embedding-4B)                                 |
| 4         | 509M                   | Qwen3          | [Qwen/Qwen3-Embedding-0.6B](https://hf.co/Qwen/Qwen3-Embedding-0.6B)                             |
| 6         | 7.61B (Very Expensive) | Qwen2          | [Alibaba-NLP/gte-Qwen2-7B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-7B-instruct)             |
| 7         | 560M                   | XLM-RoBERTa    | [intfloat/multilingual-e5-large-instruct](https://hf.co/intfloat/multilingual-e5-large-instruct) |
| 8         | 308M                   | Gemma3         | [google/embeddinggemma-300m](https://hf.co/google/embeddinggemma-300m) (gated)                   |
| 15        | 1.78B (Expensive)      | Qwen2          | [Alibaba-NLP/gte-Qwen2-1.5B-instruct](https://hf.co/Alibaba-NLP/gte-Qwen2-1.5B-instruct)         |
| 18        | 7.11B (Very Expensive) | Mistral        | [Salesforce/SFR-Embedding-2_R](https://hf.co/Salesforce/SFR-Embedding-2_R)                       |
| 35        | 568M                   | XLM-RoBERTa    | [Snowflake/snowflake-arctic-embed-l-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-l-v2.0) |
| 41        | 305M                   | Alibaba GTE    | [Snowflake/snowflake-arctic-embed-m-v2.0](https://hf.co/Snowflake/snowflake-arctic-embed-m-v2.0) |
| 52        | 335M                   | BERT           | [WhereIsAI/UAE-Large-V1](https://hf.co/WhereIsAI/UAE-Large-V1)                                   |
| 58        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1](https://hf.co/nomic-ai/nomic-embed-text-v1)                       |
| 79        | 137M                   | NomicBERT      | [nomic-ai/nomic-embed-text-v1.5](https://hf.co/nomic-ai/nomic-embed-text-v1.5)                   |
| 103       | 109M                   | MPNet          | [sentence-transformers/all-mpnet-base-v2](https://hf.co/sentence-transformers/all-mpnet-base-v2) |
| N/A       | 475M-A305M             | NomicBERT      | [nomic-ai/nomic-embed-text-v2-moe](https://hf.co/nomic-ai/nomic-embed-text-v2-moe)               |
| N/A       | 434M                   | Alibaba GTE    | [Alibaba-NLP/gte-large-en-v1.5](https://hf.co/Alibaba-NLP/gte-large-en-v1.5)                     |
| N/A       | 396M                   | ModernBERT     | [answerdotai/ModernBERT-large](https://hf.co/answerdotai/ModernBERT-large)                       |
| N/A       | 340M                   | Qwen3          | [voyageai/voyage-4-nano](https://hf.co/voyageai/voyage-4-nano)                                   |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)             |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-code](https://hf.co/jinaai/jina-embeddings-v2-base-code)         |

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
| Re-Ranking         | ModernBert  | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

### Docker

```shell
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id $model
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

```console
$ text-embeddings-router --help
Text Embedding Webserver

Usage: text-embeddings-router [OPTIONS] --model-id <MODEL_ID>

Options:
      --model-id <MODEL_ID>
          The Hugging Face model ID, can be any model listed on <https://huggingface.co/models> with the `text-embeddings-inference` tag (meaning it's compatible with Text Embeddings Inference).

          Alternatively, the specified ID can also be a path to a local directory containing the necessary model files saved by the `save_pretrained(...)` methods of either Transformers or Sentence Transformers.

          [env: MODEL_ID=]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id or a branch like `refs/pr/2`

          [env: REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Default to the number of CPU cores on the machine

          [env: TOKENIZATION_WORKERS=]

      --dtype <DTYPE>
          The dtype to be forced upon the model

          [env: DTYPE=]
          [possible values: float16, float32]

      --served-model-name <SERVED_MODEL_NAME>
          The name of the model that is being served. If not specified, defaults to `--model-id`. It is only used for the OpenAI-compatible endpoints via HTTP

          [env: SERVED_MODEL_NAME=]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: POOLING=]

          Possible values:
          - cls:        Select the CLS token as embedding
          - mean:       Apply Mean pooling to the model embeddings
          - splade:     Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings. This option is only available if the loaded model is a `ForMaskedLM` Transformer model
          - last-token: Select the last token as embedding

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment. Having a low limit will refuse clients requests instead of having them wait for too long and is usually good to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.

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
          Control automatic truncation of inputs that exceed the model's maximum supported size. Defaults to `true` (truncation enabled). Set to `false` to disable truncation; when disabled and the model's maximum input length exceeds `--max-batch-tokens`, the server will refuse to start with an error instead of silently truncating sequences.

          Unused for gRPC servers

          [env: AUTO_TRUNCATE=]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: DEFAULT_PROMPT=]

      --dense-path <DENSE_PATH>
          Optionally, define the path to the Dense module required for some embedding models.

          Some embedding models require an extra `Dense` module which contains a single Linear layer and an activation function. By default, those `Dense` modules are stored under the `2_Dense` directory, but there might be cases where different `Dense` modules are provided, to convert the pooled embeddings into different dimensions, available as `2_Dense_<dims>` e.g. https://huggingface.co/NovaSearch/stella_en_400M_v5.

          Note that this argument is optional, only required to be set if there is no `modules.json` file or when you want to override a single Dense module path, only when running with the `candle` backend.

          [env: DENSE_PATH=]

      --hf-token <HF_TOKEN>
          Your Hugging Face Hub token. If neither `--hf-token` nor `HF_TOKEN` are set, the token will be read from the `$HF_HOME/token` path, if it exists. This ensures access to private or gated models, and allows for a more permissive rate limiting

          [env: HF_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=]
          [default: 0.0.0.0]

      -p, --port <PORT>
          The port to listen on

          [env: PORT=]
          [default: 3000]

      --uds-path <UDS_PATH>
          The name of the unix socket some text-embeddings-inference backends will use as they communicate internally with gRPC

          [env: UDS_PATH=]
          [default: /tmp/text-embeddings-inference-server]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance

          [env: HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization header set with the api key as Bearer token.

          [env: API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --disable-spans
          Whether or not to include the log trace through spans

          [env: DISABLE_SPANS=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `text-embeddings-inference.server`

          [env: OTLP_SERVICE_NAME=]
          [default: text-embeddings-inference.server]

      --prometheus-port <PROMETHEUS_PORT>
          The Prometheus port to listen on

          [env: PROMETHEUS_PORT=]
          [default: 9000]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          Unused for gRPC servers

          [env: CORS_ALLOW_ORIGIN=]

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```

### Docker Images

Text Embeddings Inference ships with multiple Docker images that you can use to target a specific backend:

| Architecture                           | Image                                                                   |
|----------------------------------------|-------------------------------------------------------------------------|
| CPU                                    | ghcr.io/huggingface/text-embeddings-inference:cpu-1.9                   |
| Volta                                  | NOT SUPPORTED                                                           |
| Turing (T4, RTX 2000 series, ...)      | ghcr.io/huggingface/text-embeddings-inference:turing-1.9 (experimental) |
| Ampere 8.0 (A100, A30)                 | ghcr.io/huggingface/text-embeddings-inference:1.9                       |
| Ampere 8.6 (A10, A40, ...)             | ghcr.io/huggingface/text-embeddings-inference:86-1.9                    |
| Ada Lovelace (RTX 4000 series, ...)    | ghcr.io/huggingface/text-embeddings-inference:89-1.9                    |
| Hopper (H100)                          | ghcr.io/huggingface/text-embeddings-inference:hopper-1.9                |
| Blackwell 10.0 (B200, GB200, ...)      | ghcr.io/huggingface/text-embeddings-inference:100-1.9 (experimental)    |
| Blackwell 12.0 (GeForce RTX 50X0, ...) | ghcr.io/huggingface/text-embeddings-inference:120-1.9 (experimental)    |

**Warning**: Flash Attention is turned off by default for the Turing image as it suffers from precision issues.
You can turn Flash Attention v1 ON by using the `USE_FLASH_ATTENTION=True` environment variable.

### API documentation

You can consult the OpenAPI documentation of the `text-embeddings-inference` REST API using the `/docs` route.
The Swagger UI is also available
at: [https://huggingface.github.io/text-embeddings-inference](https://huggingface.github.io/text-embeddings-inference).

### Using a private or gated model

You have the option to utilize the `HF_TOKEN` environment variable for configuring the token employed by
`text-embeddings-inference`. This allows you to gain access to protected resources.

For example:

1. Go to https://huggingface.co/settings/tokens
2. Copy your CLI READ token
3. Export `HF_TOKEN=<your CLI READ token>`

or with Docker:

```shell
model=<your private model>
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your CLI READ token>

docker run --gpus all -e HF_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id $model
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
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B

# Set the models directory as the volume path
volume=$PWD

# Mount the models directory inside the container with a volume and set the model ID
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id /data/Qwen3-Embedding-0.6B
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

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id $model
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

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id $model
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

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 --model-id $model --pooling splade
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
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:cuda-1.9-grpc --model-id $model
```

```shell
grpcurl -d '{"inputs": "What is Deep Learning"}' -plaintext 0.0.0.0:8080 tei.v1.Embed/Embed
```

## Local install

### Apple Silicon (Homebrew)

On Apple Silicon (M1/M2/M3/M4), you can install a prebuilt binary via Homebrew:

```shell
brew install text-embeddings-inference
```

Then launch Text Embeddings Inference with Metal acceleration:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

### CPU

You can also opt to install `text-embeddings-inference` locally.

First [install Rust](https://rustup.rs/):

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then run:

```shell
# On x86 with ONNX backend (recommended)
cargo install --path router -F ort
# On x86 with Intel backend
cargo install --path router -F mkl
# On M1 or M2
cargo install --path router -F metal
```

You can now launch Text Embeddings Inference on CPU with:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### CUDA

GPUs with CUDA compute capabilities < 7.5 are not supported (V100, Titan V, GTX 1000 series, ...).

Make sure you have CUDA and the NVIDIA drivers installed. NVIDIA drivers on your device need to be compatible with CUDA
version 12.2 or higher. You also need to add the NVIDIA binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

Then run the following (might take a while as it needs to compile the CUDA kernels):

```shell
# On Turing GPUs (T4, RTX 2000 series ... )
cargo install --path router -F candle-cuda-turing

# On Ampere, Ada Lovelace, Hopper and Blackwell
cargo install --path router -F candle-cuda
```

You can now launch Text Embeddings Inference on GPU as follows:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

## Docker

You can build the CPU container with Docker as:

```shell
docker build -f Dockerfile .
```

To build the CUDA containers, you need to know the compute cap of the GPU you will be using
at runtime, to build the image accordingly:

```shell
# Get submodule dependencies
git submodule update --init

# Example for Turing (T4, RTX 2000 series, ...)
runtime_compute_cap=75

# Example for Ampere (A100, ...)
runtime_compute_cap=80

# Example for Ampere (A10, ...)
runtime_compute_cap=86

# Example for Ada Lovelace (RTX 4000 series, ...)
runtime_compute_cap=89

# Example for Hopper (H100, ...)
runtime_compute_cap=90

# Example for Blackwell (B200, GB200, ...)
runtime_compute_cap=100

# Example for Blackwell (GeForce RTX 50X0, RTX PRO 6000, ...)
runtime_compute_cap=120

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
