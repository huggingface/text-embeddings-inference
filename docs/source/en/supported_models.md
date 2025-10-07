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

# Supported models and hardware

We are continually expanding our support for other model types and plan to include them in future updates.

## Supported embeddings models

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
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-en](https://hf.co/jinaai/jina-embeddings-v2-base-en)             |
| N/A       | 137M                   | JinaBERT       | [jinaai/jina-embeddings-v2-base-code](https://hf.co/jinaai/jina-embeddings-v2-base-code)         |

To explore the list of best performing text embeddings models, visit the
[Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Supported re-rankers and sequence classification models

Text Embeddings Inference currently supports CamemBERT, and XLM-RoBERTa Sequence Classification models with absolute positions.

Below are some examples of the currently supported models:

| Task               | Model Type  | Model ID                                                                                                        |
|--------------------|-------------|-----------------------------------------------------------------------------------------------------------------|
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-large](https://huggingface.co/BAAI/bge-reranker-large)                                       |
| Re-Ranking         | XLM-RoBERTa | [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base)                                         |
| Re-Ranking         | GTE         | [Alibaba-NLP/gte-multilingual-reranker-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base) |
| Re-Ranking         | ModernBert  | [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) |
| Sentiment Analysis | RoBERTa     | [SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions)                     |

## Supported hardware

Text Embeddings Inference supports can be used on CPU, Turing (T4, RTX 2000 series, ...), Ampere 80 (A100, A30),
Ampere 86 (A10, A40, ...), Ada Lovelace (RTX 4000 series, ...), and Hopper (H100) architectures.

The library does **not** support CUDA compute capabilities < 7.5, which means V100, Titan V, GTX 1000 series, etc. are not supported.

To leverage your GPUs, make sure to install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and use
NVIDIA drivers with CUDA version 12.2 or higher.

Find the appropriate Docker image for your hardware in the following table:

| Architecture                        | Image                                                                    |
|-------------------------------------|--------------------------------------------------------------------------|
| CPU                                 | ghcr.io/huggingface/text-embeddings-inference:cpu-1.8                    |
| Volta                               | NOT SUPPORTED                                                            |
| Turing (T4, RTX 2000 series, ...)   | ghcr.io/huggingface/text-embeddings-inference:turing-1.8 (experimental)  |
| Ampere 80 (A100, A30)               | ghcr.io/huggingface/text-embeddings-inference:1.8                        |
| Ampere 86 (A10, A40, ...)           | ghcr.io/huggingface/text-embeddings-inference:86-1.8                     |
| Ada Lovelace (RTX 4000 series, ...) | ghcr.io/huggingface/text-embeddings-inference:89-1.8                     |
| Hopper (H100)                       | ghcr.io/huggingface/text-embeddings-inference:hopper-1.8 (experimental)  |

**Warning**: Flash Attention is turned off by default for the Turing image as it suffers from precision issues.
You can turn Flash Attention v1 ON by using the `USE_FLASH_ATTENTION=True` environment variable.
