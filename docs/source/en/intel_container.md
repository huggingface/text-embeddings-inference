<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Using TEI Container with Intel® Hardware

This guide explains how to build and deploy `text-embeddings-inference` containers optimized for Intel® hardware, including CPUs, XPUs, and HPUs.

## CPU

### Build Docker Image

To build a container optimized for Intel® CPUs, run the following command:

```shell
platform="cpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_cpu_ipex
```

### Deploy Docker Container

To deploy your model on an Intel® CPU, use the following command:

```shell
model='Qwen/Qwen3-Embedding-0.6B'
volume=$PWD/data

docker run -p 8080:80 -v $volume:/data tei_cpu_ipex --model-id $model
```

## XPU

### Build Docker Image

To build a container optimized for Intel® XPUs, run the following command:

```shell
platform="xpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_xpu_ipex
```

### Deploy Docker Container

To deploy your model on an Intel® XPU, use the following command:

```shell
model='Qwen/Qwen3-Embedding-0.6B'
volume=$PWD/data

docker run -p 8080:80 -v $volume:/data --device=/dev/dri -v /dev/dri/by-path:/dev/dri/by-path tei_xpu_ipex --model-id $model --dtype float16
```

## HPU

> [!WARNING]
> TEI is supported only on Gaudi 2 and Gaudi 3. Gaudi 1 is **not** supported.

### Build Docker Image

To build a container optimized for Intel® HPUs (Gaudi), run the following command:

```shell
platform="hpu"

docker build . -f Dockerfile-intel --build-arg PLATFORM=$platform -t tei_hpu
```

### Deploy Docker Container

To deploy your model on an Intel® HPU (Gaudi), use the following command:

```shell
model='Qwen/Qwen3-Embedding-0.6B'
volume=$PWD/data

docker run -p 8080:80 -v $volume:/data --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e MAX_WARMUP_SEQUENCE_LENGTH=512 tei_hpu --model-id $model --dtype bfloat16
```

## Prebuilt Docker Images

For convenience, prebuilt Docker images are available on GitHub Container Registry (GHCR). You can pull these images directly without the need to build them manually:

### CPU
To use the prebuilt image optimized for Intel® CPUs, run:
```shell
docker pull ghcr.io/huggingface/text-embeddings-inference:cpu-ipex-latest
```

### XPU
To use the prebuilt image optimized for Intel® XPUs, run:
```shell
docker pull ghcr.io/huggingface/text-embeddings-inference:xpu-ipex-latest
```

### HPU

> [!WARNING]
> TEI is supported only on Gaudi 2 and Gaudi 3. Gaudi 1 is **not** supported.

To use the prebuilt image optimized for Intel® HPUs (Gaudi), run:
```shell
docker pull ghcr.io/huggingface/text-embeddings-inference:hpu-latest
```
