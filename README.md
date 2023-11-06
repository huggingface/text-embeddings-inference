<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models. 

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on an Nvidia A10 with a sequence length of 512 tokens:

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
  - [Distributed Tracing](#distributed-tracing)
- [Local Install](#local-install)
- [Docker Build](#docker-build)

Text Embeddings Inference (TEI) is a toolkit for deploying and serving open source text embeddings models. TEI enables
high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE and E5. TEI implements many features
such as:

* No model graph compilation step
* Small docker images and fast boot times. Get ready for true serverless!
* Token based dynamic batching
* Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
[Candle](https://github.com/huggingface/candle) and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
* [Safetensors](https://github.com/huggingface/safetensors) weight loading
* Production ready (distributed tracing with Open Telemetry, Prometheus metrics)


## Get Started

### Supported Models

You can use any JinaBERT model with Alibi or absolute positions or any BERT, CamemBERT, RoBERTa, or XLM-RoBERTa model with absolute positions in `text-embeddings-inference`.

**Support for other model types will be added in the future.**

Examples of supported models:

| MTEB Rank | Model Type  | Model ID                                                                               | 
|-----------|-------------|----------------------------------------------------------------------------------------|
| 1         | Bert        | [BAAI/bge-large-en-v1.5](https://hf.co/BAAI/bge-large-en-v1.5)                         |
| 2         |             | [BAAI/bge-base-en-v1.5](https://hf.co/BAAI/bge-base-en-v1.5)                           |
| 3         |             | [llmrails/ember-v1](https://hf.co/llmrails/ember-v1)                                   |
| 4         |             | [thenlper/gte-large](https://hf.co/thenlper/gte-large)                                 |
| 5         |             | [thenlper/gte-base](https://hf.co/thenlper/gte-base)                                   |
| 6         |             | [intfloat/e5-large-v2](https://hf.co/intfloat/e5-large-v2)                             |
| 7         |             | [BAAI/bge-small-en-v1.5](https://hf.co/BAAI/bge-small-en-v1.5)                         |
| 10        |             | [intfloat/e5-base-v2](https://hf.co/intfloat/e5-base-v2)                               |
| 11        | XLM-RoBERTa | [intfloat/multilingual-e5-large](https://hf.co/intfloat/multilingual-e5-large)         |
| N/A       | JinaBERT    | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)   |
| N/A       | JinaBERT    | [jinaai/jina-embeddings-v2-small-en](https://hf.co/jinaai/jina-embeddings-v2-small-en) |


You can explore the list of best performing text embeddings models [here](https://huggingface.co/spaces/mteb/leaderboard).

### Docker

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.3.0 --model-id $model --revision $revision
```

And then you can make requests like

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
We also recommend using NVIDIA drivers with CUDA version 12.0 or higher.

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
          
          If `dtype` is not set, it defaults to float32 on accelerate, and float16 for all other architectures

          [env: DTYPE=]
          [possible values: float16, float32]

      --pooling <POOLING>
          Optionally control the pooling method. 
          
          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json`
          configuration. 
          
          If `pooling` is set, it will override the model pooling configuration
          
          [env: POOLING=]
          [possible values: cls, mean]

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

          [env: HUGGINGFACE_HUB_CACHE=/data]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          [env: OTLP_ENDPOINT=]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          [env: CORS_ALLOW_ORIGIN=]
```

### Docker Images

Text Embeddings Inference ships with multiple Docker images that you can use to target a specific backend:

| Architecture                        | Image                                                                     |
|-------------------------------------|---------------------------------------------------------------------------|
| CPU                                 | ghcr.io/huggingface/text-embeddings-inference:cpu-0.3.0                   |
| Volta                               | NOT SUPPORTED                                                             |
| Turing (T4, RTX 2000 series, ...)   | ghcr.io/huggingface/text-embeddings-inference:turing-0.3.0 (experimental) |
| Ampere 80 (A100, A30)               | ghcr.io/huggingface/text-embeddings-inference:0.3.0                       |
| Ampere 86 (A10, A40, ...)           | ghcr.io/huggingface/text-embeddings-inference:86-0.3.0                    |
| Ada Lovelace (RTX 4000 series, ...) | ghcr.io/huggingface/text-embeddings-inference:89-0.3.0                    |
| Hopper (H100)                       | ghcr.io/huggingface/text-embeddings-inference:hopper-0.3.0 (experimental) |

**Warning**: Flash Attention is turned off by default for the Turing image as it suffers from precision issues.
You can turn Flash Attention v1 ON by using the `USE_FLASH_ATTENTION=True` environment variable.

### API documentation

You can consult the OpenAPI documentation of the `text-embeddings-inference` REST API using the `/docs` route.
The Swagger UI is also available at: [https://huggingface.github.io/text-embeddings-inference](https://huggingface.github.io/text-embeddings-inference).

### Using a private or gated model

You have the option to utilize the `HUGGING_FACE_HUB_TOKEN` environment variable for configuring the token employed by
`text-embeddings-inference`. This allows you to gain access to protected resources.

For example:

1. Go to https://huggingface.co/settings/tokens
2. Copy your cli READ token
3. Export `HUGGING_FACE_HUB_TOKEN=<your cli READ token>`

or with Docker:

```shell
model=<your private model>
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run
token=<your cli READ token>

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.3.0 --model-id $model
```

### Distributed Tracing

`text-embeddings-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

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
cargo install --path router -F candle -F mkl
# On M1 or M2
cargo install --path router -F candle -F accelerate
```

You can now launch Text Embeddings Inference on CPU with: 

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080
```

**Note:** on some machines, you may also need the OpenSSL libraries and gcc. On Linux machines, run:

```shell
sudo apt-get install libssl-dev gcc -y
```

### Cuda

GPUs with Cuda compute capabilities < 7.5 are not supported (V100, Titan V, GTX 1000 series, ...).

Make sure you have Cuda and the nvidia drivers installed. We recommend using NVIDIA drivers with CUDA version 12.0 or higher. 
You also need to add the nvidia binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

Then run:

```shell
# This can take a while as we need to compile a lot of cuda kernels

# On Turing GPUs (T4, RTX 2000 series ... )
cargo install --path router -F candle-cuda-turing --no-default-features

# On Ampere and Hopper
cargo install --path router -F candle-cuda --no-default-features
```

You can now launch Text Embeddings Inference on GPU with:

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080
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
