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

# Build a custom container for TEI

You can build our own CPU or CUDA TEI container using Docker.  To build a CPU container, run the following command in the
directory containing your custom Dockerfile:

```shell
docker build .
```

To build a CUDA container, it is essential to determine the compute capability (compute cap) of the GPU that will be
used at runtime. This information is crucial for the proper configuration of the CUDA containers. The following are
the examples of runtime compute capabilities for various GPU types:

- Turing (T4, RTX 2000 series, ...) - `runtime_compute_cap=75`
- A100 - `runtime_compute_cap=80`
- A10 - `runtime_compute_cap=86`
- Ada Lovelace (RTX 4000 series, ...) - `runtime_compute_cap=89`
- H100 - `runtime_compute_cap=90`

Once you have determined the compute capability is determined, set it as the `runtime_compute_cap` variable and build
the container as shown in the example below:

```shell
runtime_compute_cap=80

docker build . -f Dockerfile-cuda --build-arg CUDA_COMPUTE_CAP=$runtime_compute_cap
```
