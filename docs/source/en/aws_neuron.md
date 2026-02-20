<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
# Using TEI with AWS Trainium and Inferentia

Text Embeddings Inference (TEI) supports AWS Trainium and Inferentia accelerators through the [optimum-neuron](https://huggingface.co/docs/optimum-neuron) library.

## Supported Model Types

- **Embedding models**: Uses `NeuronSentenceTransformers` for sentence embeddings (e.g., BGE, sentence-transformers models)
- **Classification models**: Uses `NeuronModelForSequenceClassification` for sequence classification tasks
- **SPLADE models**: Uses `NeuronModelForMaskedLM` for sparse embeddings

## Build Docker Image

To build a container optimized for AWS Neuron devices:

```shell
docker build . -f Dockerfile-neuron -t tei-neuron:main
```

## Deploy with Pre-compiled Models

Pre-compiled models are recommended for production use as they skip the compilation step and start faster.

```shell
model='optimum/bge-base-en-v1.5-neuronx'
volume=$PWD/data

docker run --privileged \
    -p 8080:80 \
    -v $volume:/data \
    tei-neuron:main \
    --model-id $model \
    --dtype float32
```

> **Note**: The `--privileged` flag is required for the Neuron OCI hook to work properly.

## Deploy with On-the-fly Compilation

You can also use non-pre-compiled models. TEI will compile the model for Neuron automatically on first load. This takes additional time but allows you to use any compatible model.

```shell
model='BAAI/bge-base-en-v1.5'
volume=$PWD/data

docker run --privileged \
    -p 8080:80 \
    -v $volume:/data \
    -e NEURON_BATCH_SIZE=1 \
    -e NEURON_SEQUENCE_LENGTH=512 \
    tei-neuron:main \
    --model-id $model \
    --dtype float32
```

### Compilation Environment Variables

When using on-the-fly compilation, you can configure the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURON_BATCH_SIZE` | 1 | Batch size for Neuron compilation (static shape) |
| `NEURON_SEQUENCE_LENGTH` | 512 | Maximum sequence length for Neuron compilation (static shape) |

> **Note**: Neuron requires static shapes for compilation. The batch size and sequence length are fixed at compilation time.

## Runtime Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NEURON_RT_NUM_CORES` | 1 | Number of Neuron cores to use |
| `NEURON_RT_VISIBLE_CORES` | 0 | Which Neuron cores are visible to the runtime |

## Pre-compiled Models

For faster startup, use pre-compiled Neuron models from the Hugging Face Hub like:

- [optimum/bge-base-en-v1.5-neuronx](https://huggingface.co/optimum/bge-base-en-v1.5-neuronx)

You can also compile your own models using the [Optimum Neuron guide](https://huggingface.co/docs/optimum-neuron/en/model_doc/sentence_transformers/overview).

## Testing Your Deployment

Once the container is running, you can test the embedding endpoint:

```shell
curl 127.0.0.1:8080/embed \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "What is Deep Learning?"}'
```
