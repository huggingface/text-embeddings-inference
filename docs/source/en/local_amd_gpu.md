<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Using TEI locally with an AMD GPU

Text-Embeddings-Inference supports the [AMD GPUs officially supporting ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html), including AMD Instinct MI210, MI250, MI300 and some of the AMD Radeon series GPUs.

To leverage AMD GPUs, Text-Embeddings-Inference relies on its Python backend, and not on the [candle](https://github.com/huggingface/candle) backend that is used for CPU, Nvidia GPUs and Metal. The support in the python backend is more limited (Bert embeddings) but easily extendible. We welcome contributions to extend the supported models.

## Usage through docker

Using docker is the recommended approach.

```bash
docker run --rm -it --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --net host \
    --device=/dev/kfd --device=/dev/dri --group-add video --ipc=host --shm-size 32g \
    ghcr.io/huggingface/text-embeddings-inference:rocm-1.2.4 \
    --model-id sentence-transformers/all-MiniLM-L6-v2
```

and

```bash
curl 127.0.0.1:80/embed \
    -X POST -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```