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

Text Embeddings Inference supports AMD Instinct GPUs (MI200, MI300 series) using [ROCm](https://rocm.docs.amd.com/).

## Prerequisites

- AMD Instinct GPU (MI200, MI300 series) with ROCm drivers on the host

## Option A: Docker (recommended)

The easiest way to run TEI on AMD GPUs is with the pre-built Docker image:

```shell
model=BAAI/bge-base-en-v1.5
volume=$PWD/data  # share a volume to avoid re-downloading weights

docker run \
  --device /dev/kfd --device /dev/dri/renderD128 \
  --group-add video \
  --ipc=host \
  -p 8080:80 \
  -v $volume:/data \
  --pull always \
  ghcr.io/huggingface/text-embeddings-inference:rocm-latest \
  --model-id $model --dtype bfloat16
```

Then test it:

```shell
curl http://localhost:8080/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{
      "input": "What is Deep Learning?",
      "model": "text-embeddings-inference",
      "encoding_format": "float"
    }'
```

---

## Option B: Build from source inside the TEI ROCm image

If you want to modify TEI and rebuild it, use the official TEI ROCm image
(`ghcr.io/huggingface/text-embeddings-inference:rocm-latest`) as your base environment instead of AMD's
`rocm/pytorch` image. It already ships the exact ROCm build of PyTorch, `flash-attn`, the compute kernels, and
the installed Python backend that TEI expects, so you skip the PyTorch/dependency setup entirely and avoid
version-mismatch issues — you only rebuild the parts you change.

## Step 1: Start the container

Mount your checkout and override the entrypoint to get a shell:

```shell
docker run -it --device=/dev/kfd --device=/dev/dri/renderD128 \
  --group-add video --ipc=host --shm-size 8g \
  -v $PWD:/workspace -w /workspace \
  --entrypoint bash \
  ghcr.io/huggingface/text-embeddings-inference:rocm-latest
```

PyTorch (ROCm), `flash-attn`, the compute kernels, and the `text-embeddings-server` Python backend are already
installed in this image, and a prebuilt `text-embeddings-router` is on the `PATH`. The steps below only cover
rebuilding after you edit the source.

## Step 2: Install Rust (only needed to rebuild the router)

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## Step 3: Rebuild the Rust router from your source

```shell
cargo build --release \
    --no-default-features \
    --features python,http \
    --bin text-embeddings-router
```

> If you also changed the Python backend, reinstall it in editable mode from your mounted checkout:
> ```shell
> pip install --no-deps -e backends/python/server
> ```
> The protobuf stubs are already generated in the image; regenerate them only for a fresh checkout with
> `cd backends/python/server && make gen-server`.

## Step 4: Launch your build

```shell
model=BAAI/bge-base-en-v1.5

./target/release/text-embeddings-router --model-id $model --dtype bfloat16 --port 8080
```

Once the server is ready, you can test it with a simple embed request:

```shell
curl http://localhost:8080/v1/embeddings \
    -H 'Content-Type: application/json' \
    -d '{
      "input": "What is Deep Learning?",
      "model": "text-embeddings-inference",
      "encoding_format": "float"
    }'
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
