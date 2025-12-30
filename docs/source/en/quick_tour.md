<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Quick Tour

## Set up

The easiest way to get started with TEI is to use one of the official Docker containers
(see [Supported models and hardware](supported_models) to choose the right container).

Hence one needs to install Docker following their [installation instructions](https://docs.docker.com/get-docker/).

TEI supports inference both on GPU and CPU. If you plan on using a GPU, make sure to check that your hardware is supported by checking [this table](https://github.com/huggingface/text-embeddings-inference?tab=readme-ov-file#docker-images).
Next, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher.

## Deploy

Next it's time to deploy your model. Let's say you want to use [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B). Here's how you can do this:

```shell
model=Qwen/Qwen3-Embedding-0.6B
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
```

<Tip>

We also recommend sharing a volume with the Docker container (`volume=$PWD/data`) to avoid downloading weights every run.

</Tip>

## Inference

Inference can be performed in 3 ways: using cURL, or via the `InferenceClient` or `OpenAI` Python SDKs.

#### cURL

To send a POST request to the TEI endpoint using cURL, you can run the following command:

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

#### Python

To run inference using Python, you can either use the [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/en/index) Python SDK (recommended) or the `openai` Python SDK.

##### huggingface_hub

You can install it via pip as `pip install --upgrade --quiet huggingface_hub`, and then run:

```python
from huggingface_hub import InferenceClient

client = InferenceClient()

embedding = client.feature_extraction("What is deep learning?",
                                      model="http://localhost:8080/embed")
print(len(embedding[0]))
```

#### OpenAI
To send requests to the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create) exposed on Text Embeddings Inference (TEI) with the OpenAI Python SDK, you can install it as `pip install --upgrade openai`, and then run the following snippet:

```python
import os
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key= "-")

response = client.embeddings.create(
    model="text-embeddings-inference",
    input="What is Deep Learning?",
)

print(response.data[0].embedding)
```

Alternatively, you can also send the request with cURL as follows:
```bash
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "What is Deep Learning?",
    "model": "text-embeddings-inference",
    "encoding_format": "float"
  }'
```

## Re-rankers and sequence classification

TEI also supports re-ranker and classic sequence classification models.

### Re-rankers

Rerankers, also called cross-encoders, are sequence classification models with a single class that score the similarity between a query and a text. See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

Let's say you want to use [`BAAI/bge-reranker-large`](https://huggingface.co/BAAI/bge-reranker-large). First, you can deploy it like so:

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
```

Once you have deployed a model, you can use the `rerank` endpoint to rank the similarity between a query and a list of texts. With `cURL` this can be done like so:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."], "raw_scores": false}' \
    -H 'Content-Type: application/json'
```

### Sequence classification models

You can also use classic Sequence Classification models like [`SamLowe/roberta-base-go_emotions`](https://huggingface.co/SamLowe/roberta-base-go_emotions):

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

## Batching

You can send multiple inputs in a batch. For example, for embeddings:

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":["Today is a nice day", "I like you"]}' \
    -H 'Content-Type: application/json'
```

And for Sequence Classification:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":[["I like you."], ["I hate pineapples"]]}' \
    -H 'Content-Type: application/json'
```

## Air gapped deployment

To deploy Text Embeddings Inference in an air-gapped environment, first download the weights and then mount them inside
the container using a volume.

For example:

```shell
# (Optional) create a `models` directory
mkdir models
cd models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5

# Set the models directory as the volume path
volume=$PWD

# Mount the models directory inside the container with a volume and set the model ID
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.8 --model-id /data/gte-base-en-v1.5
```
