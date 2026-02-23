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

# Using TEI locally with Metal

You can install `text-embeddings-inference` locally to run it on your own Mac with Metal support.

## Homebrew (Apple Silicon)

On Apple Silicon (M1/M2/M3/M4), you can install a prebuilt binary via Homebrew:

```shell
brew install text-embeddings-inference
```

Then launch Text Embeddings Inference:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

## Build from source

Alternatively, you can build from source. Here are the step-by-step instructions:

## Step 1: Install Rust

[Install Rust](https://rustup.rs/) on your machine by run the following in your terminal, then following the instructions:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Step 2: Install with Metal support

```shell
cargo install --path router -F metal
```

## Step 3: Launch Text Embeddings Inference

Once the installation is successfully complete, you can launch Text Embeddings Inference with Metal with the following command:

```shell
model=Qwen/Qwen3-Embedding-0.6B

text-embeddings-router --model-id $model --port 8080
```

Now you are ready to use `text-embeddings-inference` locally on your machine.
