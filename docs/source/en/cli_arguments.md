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

# CLI arguments

To see all options to serve your models, run the following:

```shell
text-embeddings-router --help
```

```
Usage: text-embeddings-router [OPTIONS]

Options:
      --model-id <MODEL_ID>
          The name of the model to load. Can be a MODEL_ID as listed on <https://hf.co/models> like `thenlper/gte-base`.
          Or it can be a local directory containing the necessary files as saved by `save_pretrained(...)` methods of
          transformers

          [env: MODEL_ID=]
          [default: thenlper/gte-base]

      --revision <REVISION>
          The actual revision of the model if you're referring to a model on the hub. You can use a specific commit id
          or a branch like `refs/pr/2`

          [env: REVISION=]

      --tokenization-workers <TOKENIZATION_WORKERS>
          Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation.
          Default to the number of CPU cores on the machine

          [env: TOKENIZATION_WORKERS=]

      --dtype <DTYPE>
          The dtype to be forced upon the model

          [env: DTYPE=]
          [possible values: float16, float32]

      --pooling <POOLING>
          Optionally control the pooling method for embedding models.

          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.

          If `pooling` is set, it will override the model pooling configuration

          [env: POOLING=]

          Possible values:
          - cls:        Select the CLS token as embedding
          - mean:       Apply Mean pooling to the model embeddings
          - splade:     Apply SPLADE (Sparse Lexical and Expansion) to the model embeddings. This option is only
          available if the loaded model is a `ForMaskedLM` Transformer model
          - last-token: Select the last token as embedding

      --max-concurrent-requests <MAX_CONCURRENT_REQUESTS>
          The maximum amount of concurrent requests for this particular deployment.
          Having a low limit will refuse clients requests instead of having them wait for too long and is usually good
          to handle backpressure correctly

          [env: MAX_CONCURRENT_REQUESTS=]
          [default: 512]

      --max-batch-tokens <MAX_BATCH_TOKENS>
          **IMPORTANT** This is one critical control to allow maximum usage of the available hardware.

          This represents the total amount of potential tokens within a batch.

          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.

          Overall this number should be the largest possible until the model is compute bound. Since the actual memory
          overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.

          [env: MAX_BATCH_TOKENS=]
          [default: 16384]

      --max-batch-requests <MAX_BATCH_REQUESTS>
          Optionally control the maximum number of individual requests in a batch

          [env: MAX_BATCH_REQUESTS=]

      --max-client-batch-size <MAX_CLIENT_BATCH_SIZE>
          Control the maximum number of inputs that a client can send in a single request

          [env: MAX_CLIENT_BATCH_SIZE=]
          [default: 32]

      --auto-truncate
          Automatically truncate inputs that are longer than the maximum supported size

          Unused for gRPC servers

          [env: AUTO_TRUNCATE=]

      --default-prompt-name <DEFAULT_PROMPT_NAME>
          The name of the prompt that should be used by default for encoding. If not set, no prompt will be applied.

          Must be a key in the `sentence-transformers` configuration `prompts` dictionary.

          For example if ``default_prompt_name`` is "query" and the ``prompts`` is {"query": "query: ", ...}, then the
          sentence "What is the capital of France?" will be encoded as "query: What is the capital of France?" because
          the prompt text will be prepended before any text to encode.

          The argument '--default-prompt-name <DEFAULT_PROMPT_NAME>' cannot be used with '--default-prompt <DEFAULT_PROMPT>`

          [env: DEFAULT_PROMPT_NAME=]

      --default-prompt <DEFAULT_PROMPT>
          The prompt that should be used by default for encoding. If not set, no prompt will be applied.

          For example if ``default_prompt`` is "query: " then the sentence "What is the capital of France?" will be
          encoded as "query: What is the capital of France?" because the prompt text will be prepended before any text
          to encode.

          The argument '--default-prompt <DEFAULT_PROMPT>' cannot be used with '--default-prompt-name <DEFAULT_PROMPT_NAME>`

          [env: DEFAULT_PROMPT=]

      --hf-api-token <HF_API_TOKEN>
          Your HuggingFace hub token

          [env: HF_API_TOKEN=]

      --hostname <HOSTNAME>
          The IP address to listen on

          [env: HOSTNAME=]
          [default: 0.0.0.0]

  -p, --port <PORT>
          The port to listen on

          [env: PORT=]
          [default: 3000]

      --uds-path <UDS_PATH>
          The name of the unix socket some text-embeddings-inference backends will use as they communicate internally
          with gRPC

          [env: UDS_PATH=]
          [default: /tmp/text-embeddings-inference-server]

      --huggingface-hub-cache <HUGGINGFACE_HUB_CACHE>
          The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk
          for instance

          [env: HUGGINGFACE_HUB_CACHE=]

      --payload-limit <PAYLOAD_LIMIT>
          Payload size limit in bytes

          Default is 2MB

          [env: PAYLOAD_LIMIT=]
          [default: 2000000]

      --api-key <API_KEY>
          Set an api key for request authorization.

          By default the server responds to every request. With an api key set, the requests must have the Authorization
          header set with the api key as Bearer token.

          [env: API_KEY=]

      --json-output
          Outputs the logs in JSON format (useful for telemetry)

          [env: JSON_OUTPUT=]

      --otlp-endpoint <OTLP_ENDPOINT>
          The grpc endpoint for opentelemetry. Telemetry is sent to this endpoint as OTLP over gRPC. e.g. `http://localhost:4317`

          [env: OTLP_ENDPOINT=]

      --otlp-service-name <OTLP_SERVICE_NAME>
          The service name for opentelemetry. e.g. `text-embeddings-inference.server`

          [env: OTLP_SERVICE_NAME=]
          [default: text-embeddings-inference.server]

      --cors-allow-origin <CORS_ALLOW_ORIGIN>
          Unused for gRPC servers

          [env: CORS_ALLOW_ORIGIN=]
```
