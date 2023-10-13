<div align="center">

# Text Embeddings Inference

<a href="https://github.com/huggingface/text-embeddings-inference">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/huggingface/text-embeddings-inference?style=social">
</a>
<a href="https://huggingface.github.io/text-embeddings-inference">
  <img alt="Swagger API documentation" src="https://img.shields.io/badge/API-Swagger-informational">
</a>

A blazing fast inference solution for text embeddings models. 

Benchmark for [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) on a Nvidia A10 with a sequence length of 512 tokens:

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

- No compilation step
- Dynamic shapes
- Small docker images and fast boot times. Get ready for true serverless!
- Token based dynamic batching
- Optimized transformers code for inference using [Flash Attention](https://github.com/HazyResearch/flash-attention),
[Candle](https://github.com/huggingface/candle) and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api)
- [Safetensors](https://github.com/huggingface/safetensors) weight loading
- Production ready (distributed tracing with Open Telemetry, Prometheus metrics)

## Get Started

### Supported Models

You can use any BERT model with absolute positions in `text-embeddings-inference`. If the model does not have `safetensors` weights
you can convert it using [this space](https://huggingface.co/spaces/safetensors/convert).

**Support for other model types will be added in the future.**

| MTEB Rank | Model Type | Model ID               | Specific Revision                                                        |
|-----------|------------|------------------------|--------------------------------------------------------------------------|
| 1         | Bert       | BAAI/bge-large-en-v1.5 | [refs/pr/5](https://huggingface.co/BAAI/bge-large-en-v1.5/discussions/5) |
| 2         |            | BAAI/bge-base-en-v1.5  | [refs/pr/1](https://huggingface.co/BAAI/bge-base-en-v1.5/discussions/1)  |
| 3         |            | llmrails/ember-v1      |                                                                          |
| 4         |            | thenlper/gte-large     |                                                                          |
| 5         |            | thenlper/gte-base      |                                                                          |
| 6         |            | intfloat/e5-large-v2   |                                                                          |
| 7         |            | BAAI/bge-small-en-v1.5 | [refs/pr/3](https://huggingface.co/BAAI/bge-small-en-v1.5/discussions/3) |
| 10        |            | intfloat/e5-base-v2    |                                                                          |

You can explore the list of best performing text embeddings models [here](https://huggingface.co/spaces/mteb/leaderboard).

### Docker

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5
volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:latest --model-id $model --revision $revision
```

And then you can make requests like

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

**Note:** To use GPUs, you need to install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). 
We also recommend using NVIDIA drivers with CUDA version 12 or higher. 

To see all options to serve your models:
```
text-embeddings-router --help
```

### Docker Images

Text Embeddings Inference ships with multiple Docker images that you can use to target a specific backend:

| Architecture | Image                                                       |
|--------------|-------------------------------------------------------------|
| CPU          | ghcr.io/huggingface/text-embeddings-inference:cpu-latest    |
| Turing       | ghcr.io/huggingface/text-embeddings-inference:turing-latest |
| Ampere 80    | ghcr.io/huggingface/text-embeddings-inference:latest        |
| Ampere 86    | ghcr.io/huggingface/text-embeddings-inference:86-latest     |
| Hopper       | ghcr.io/huggingface/text-embeddings-inference:hopper-latest |

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

docker run --gpus all -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:latest --model-id $model
```

### Distributed Tracing

`text-embeddings-inference` is instrumented with distributed tracing using OpenTelemetry. You can use this feature
by setting the address to an OTLP collector with the `--otlp-endpoint` argument.

### Local install

#### CPU

You can also opt to install `text-embeddings-inference` locally.

First [install Rust](https://rustup.rs/):

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then run:

```shell
cargo install --path router -F candle --no-default-features
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

#### Cuda

Make sure you have Cuda and the nvidia drivers installed. We recommend using NVIDIA drivers with CUDA version 12 or higher. 
You also need to add the nvidia binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

Then run:

```shell
# This can take a while as we need to compile a lot of cuda kernels
cargo install --path router -F candle-cuda --no-default-features
```

You can now launch Text Embeddings Inference on GPU with: 

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080
```
