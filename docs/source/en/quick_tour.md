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
plan on utilizing GPUs. We also recommend using NVIDIA drivers with CUDA version 12.2 or higher. 

Next, install Docker following their [installation instructions](https://docs.docker.com/get-docker/).

Finally, deploy your model. Let's say you want to use `BAAI/bge-large-en-v1.5`. Here's how you can do this:

```shell
model=BAAI/bge-large-en-v1.5
revision=refs/pr/5
volume=$PWD/data 

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.2.2 --model-id $model --revision $revision
```

<Tip>

Here we pass a `revision=refs/pr/5`, because the `safetensors` variant of this model is currently in a pull request.
We also recommend sharing a volume with the Docker container (`volume=$PWD/data`) to avoid downloading weights every run.

</Tip>

Once you have deployed a model you can use the `embed` endpoint by sending requests:

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```

## Sequence Classification

TEI can also be used to deploy Sequence Classification models.
See [this blogpost](https://blog.llamaindex.ai/boosting-rag-picking-the-best-embedding-reranker-models-42d079022e83) by
the LlamaIndex team to understand how you can use Sequence Classification models in your RAG pipeline to improve
downstream performance.

Let's say you want to use `BAAI/bge-reranker-large`:

```shell
model=BAAI/bge-reranker-large
revision=refs/pr/4
volume=$PWD/data 

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.3.0 --model-id $model --revision $revision
```

Once you have deployed a model you can use the `predict` endpoint and rank the similarity between a pair of inputs:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":["What is Deep Learning?", "Deep learning is..."], "raw_scores": true}' \
    -H 'Content-Type: application/json'
```

You can also use classic Sequence Classification models like `SamLowe/roberta-base-go_emotions`:

```shell
model=SamLowe/roberta-base-go_emotions
volume=$PWD/data 

docker run --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.3.0 --model-id $model 
```

Once you have deployed the model you can use the `predict` endpoint to get the emotions most associated with an input:

```bash
curl 127.0.0.1:8080/predict \
    -X POST \
    -d '{"inputs":"I like you."}' \
    -H 'Content-Type: application/json'
```
