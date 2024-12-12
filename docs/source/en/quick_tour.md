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

# Quick Tour

## Text Embeddings

The easiest way to get started with TEI is to use one of the official Docker containers
(see [Supported models and hardware](supported_models) to choose the right container).

After making sure that your hardware is supported, install the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) if you
plan on utilizing GPUs. NVIDIA drivers on your device need to be compatible with CUDA version 12.2 or higher.

Next, install Docker following their [installation instructions](https://docs.docker.com/get-docker/).

Finally, deploy your model. Let's say you want to use `BAAI/bge-large-en-v1.5`. Here's how you can do this:

```shell
model=BAAI/bge-large-en-v1.5
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

<Tip>

We also recommend sharing a volume with the Docker container (`volume=$PWD/data`) to avoid downloading weights every run.

</Tip>

Once you have deployed a model, you can use the `embed` endpoint by sending requests:

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

## Re-rankers

Re-rankers models are Sequence Classification cross-encoders models with a single class that scores the similarity
between a query and a text.

See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use re-rankers models in your RAG pipeline to improve
downstream performance.

Let's say you want to use `BAAI/bge-reranker-large`:

```shell
model=BAAI/bge-reranker-large
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

Once you have deployed a model, you can use the `rerank` endpoint to rank the similarity between a query and a list
of texts:

```bash
curl 127.0.0.1:8080/rerank \
    -X POST \
    -d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is not...", "Deep learning is..."], "raw_scores": false}' \
    -H 'Content-Type: application/json'
```

## Sequence Classification

You can also use classic Sequence Classification models like `SamLowe/roberta-base-go_emotions`:

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```

## Batching

You can send multiple inputs in a batch. For example, for embeddings

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
docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id /data/gte-base-en-v1.5
```
