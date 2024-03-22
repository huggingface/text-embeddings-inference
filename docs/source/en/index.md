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

# Text Embeddings Inference

Text Embeddings Inference (TEI) is a comprehensive toolkit designed for efficient deployment and serving of open source
text embeddings models. It enables high-performance extraction for the most popular models, including FlagEmbedding, Ember, GTE, and E5.

TEI offers multiple features tailored to optimize the deployment process and enhance overall performance.

**Key Features:**

* **Streamlined Deployment:** TEI eliminates the need for a model graph compilation step for an easier deployment process.
* **Efficient Resource Utilization:** Benefit from small Docker images and rapid boot times, allowing for true serverless capabilities.
* **Dynamic Batching:** TEI incorporates token-based dynamic batching thus optimizing resource utilization during inference.
* **Optimized Inference:** TEI leverages [Flash Attention](https://github.com/HazyResearch/flash-attention), [Candle](https://github.com/huggingface/candle), and [cuBLASLt](https://docs.nvidia.com/cuda/cublas/#using-the-cublaslt-api) by using optimized transformers code for inference.
* **Safetensors weight loading:** TEI loads [Safetensors](https://github.com/huggingface/safetensors) weights for faster boot times.
* **Production-Ready:** TEI supports distributed tracing through Open Telemetry and exports Prometheus metrics.

**Benchmarks**

Benchmark for [BAAI/bge-base-en-v1.5](https://hf.co/BAAI/bge-large-en-v1.5) on an NVIDIA A10 with a sequence length of 512 tokens:

<p>
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tei/bs1-lat.png" width="400" alt="Latency comparison for batch size of 1" />
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tei/bs1-tp.png" width="400" alt="Throughput comparison for batch size of 1"/>
</p>
<p>
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tei/bs32-lat.png" width="400" alt="Latency comparison for batch size of 32"/>
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/tei/bs32-tp.png" width="400" alt="Throughput comparison for batch size of 32" />
</p>

**Getting Started:**

To start using TEI, check the [Quick Tour](quick_tour) guide.
