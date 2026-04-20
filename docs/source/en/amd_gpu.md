<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Using TEI on AMD Instinct GPUs (ROCm)

> [!WARNING]
> AMD ROCm support is **experimental**. Only AMD Instinct GPUs (MI200, MI300 series) are tested.

Text Embeddings Inference can run on AMD Instinct GPUs using [ROCm](https://rocm.docs.amd.com/). The implementation uses PyTorch's built-in `scaled_dot_product_attention` as the attention backend.

## Prerequisites

- AMD Instinct GPU (MI200, MI300 series) with ROCm 6.x drivers on the host
- Either a working ROCm PyTorch installation, **or** the `rocm/pytorch:latest` Docker image (recommended)

---

The recommended way to get started is to use AMD's official ROCm PyTorch image, which ships with PyTorch and ROCm pre-installed. Alternatively, you can install ROCm PyTorch directly on the host with `pip install torch --index-url https://download.pytorch.org/whl/rocm6.2` and skip Step 1.

## Step 1: Start the container

```shell
docker run -it --device=/dev/kfd --device=/dev/dri \
  --group-add video --shm-size 8g \
  -v $PWD:/workspace \
  rocm/pytorch:latest bash
```

Inside the container, clone the TEI repository (or mount it via `-v`) and run the remaining steps from the repo root.

## Step 2: Install Rust

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## Step 3: Install Python dependencies

PyTorch is already provided by the container image, so install the remaining dependencies without pulling a new torch:

```shell
pip install --no-deps -r backends/python/server/requirements-amd.txt
pip install safetensors opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp-proto-grpc grpcio-reflection \
    grpc-interceptor einops packaging
```

## Step 4: Generate protobuf stubs

```shell
pip install grpcio-tools==1.62.2 mypy-protobuf==3.6.0 types-protobuf

mkdir -p backends/python/server/text_embeddings_server/pb

python -m grpc_tools.protoc \
    -I backends/proto \
    --python_out=backends/python/server/text_embeddings_server/pb \
    --grpc_python_out=backends/python/server/text_embeddings_server/pb \
    --mypy_out=backends/python/server/text_embeddings_server/pb \
    backends/proto/embed.proto

# Fix relative imports in generated files
find backends/python/server/text_embeddings_server/pb/ -name "*.py" \
    -exec sed -i 's/^\(import.*pb2\)/from . \1/g' {} \;

touch backends/python/server/text_embeddings_server/pb/__init__.py
```

## Step 5: Install the Python server package

```shell
pip install -e backends/python/server
```

## Step 6: Build the Rust router

```shell
cargo build --release \
    --no-default-features \
    --features python,http \
    --bin text-embeddings-router
```

## Step 7: Launch TEI

```shell
model=BAAI/bge-base-en-v1.5

./target/release/text-embeddings-router --model-id $model --dtype bfloat16 --port 8080
```

Once the server is ready, you can test it with a simple embed request:

```shell
curl http://localhost:8080/embed \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "What is Deep Learning?"}'
```

## Verifying GPU detection

After launch you should see a log line confirming ROCm was detected:

```
INFO text_embeddings_server::utils::device: ROCm / HIP version: X.Y.Z
```

You can also verify from Python:

```python
import torch
print(torch.cuda.is_available())  # True
print(torch.version.hip)          # e.g. 6.2.12345-...
```

## Notes

This is a work in progress — more model support and optimized operations for AMD GPUs are coming soon.
