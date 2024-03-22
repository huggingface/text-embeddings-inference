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

# Using TEI locally with CPU

You can install `text-embeddings-inference` locally to run it on your own machine. Here are the step-by-step instructions for installation:

## Step 1: Install Rust

[Install Rust](https://rustup.rs/) on your machine by run the following in your terminal, then following the instructions:

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Step 2: Install necessary packages

Depending on your machine's architecture, run one of the following commands:

### For x86 Machines

```shell
cargo install --path router -F mkl
```

### For M1 or M2 Machines

```shell
cargo install --path router -F metal
```

## Step 3: Launch Text Embeddings Inference

Once the installation is successfully complete, you can launch Text Embeddings Inference on CPU with the following command:

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080
```

<Tip>

In some cases, you might also need the OpenSSL libraries and gcc installed. On Linux machines, run the following command:

```shell
sudo apt-get install libssl-dev gcc -y
```

</Tip>

Now you are ready to use `text-embeddings-inference` locally on your machine.
If you want to run TEI locally with a GPU, check out the [Using TEI locally with GPU](local_gpu) page.
