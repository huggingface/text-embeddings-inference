<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Using TEI locally with GPU

You can install `text-embeddings-inference` locally to run it on your own machine with a GPU.
To make sure that your hardware is supported, check out the [Supported models and hardware](supported_models) page.

## Step 1: CUDA and NVIDIA drivers

Make sure you have CUDA and the NVIDIA drivers installed - NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher.

Add the NVIDIA binaries to your path:

```shell
export PATH=$PATH:/usr/local/cuda/bin
```

## Step 2: Install Rust

[Install Rust](https://rustup.rs/) on your machine by run the following in your terminal, then following the instructions:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Step 3: Install necessary packages

This step  can take a while as we need to compile a lot of cuda kernels.

### For Turing GPUs (T4, RTX 2000 series ... )

```shell
cargo install --path router -F candle-cuda-turing -F http --no-default-features
```

### For Ampere and Hopper

```shell
cargo install --path router -F candle-cuda -F http --no-default-features
```

## Step 4: Launch Text Embeddings Inference

You can now launch Text Embeddings Inference on GPU with:

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080
```
