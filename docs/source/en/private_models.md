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

# Serving private and gated models

If the model you wish to serve is behind gated access or resides in a private model repository on Hugging Face Hub,
you will need to have access to the model to serve it.

Once you have confirmed that you have access to the model:

- Navigate to your account's [Profile | Settings | Access Tokens page](https://huggingface.co/settings/tokens).
- Generate and copy a read token.

If you're the CLI, set the `HF_API_TOKEN` environment variable. For example:

```shell
export HF_API_TOKEN=<YOUR READ TOKEN>
```

Alternatively, you can provide the token when deploying the model with Docker:

```shell
model=<your private model>
volume=$PWD/data
token=<your cli Hugging Face Hub token>

docker run --gpus all -e HF_API_TOKEN=$token -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:1.6 --model-id $model
```
