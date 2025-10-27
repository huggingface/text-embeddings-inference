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
# Using TEI Container with AWS Trainium and Inferentia Instances

## Build Docker Image

To build a container optimized for AWS Neuron devices, run the following command:

```shell
platform="neuron"

docker build . -f Dockerfile-neuron -t tei_neuron
```

### Deploy Docker Container

To deploy your model on an AWS Trainium or Inferentia instance, use the following command:

```shell
model='Qwen/Qwen3-Embedding-0.6B'
volume=$PWD/data

docker run -p 8080:80 -v $volume:/data tei_neuron --model-id $model
```