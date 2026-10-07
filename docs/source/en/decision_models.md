<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Structured decision models

Text Embeddings Inference (TEI) can serve structured decision models that answer several typed questions about one input state in a single request. The first supported model is [Laya](https://huggingface.co/convaiinnovations/laya).

> [!WARNING]
> Decision models use the OpenAI-compatible `/v1/decisions` endpoint. The HTTP server does not expose embedding, reranking, or prediction routes for them.

## Deploy a decision model

Start TEI with a decision model as you would with an embedding model. For example:

```shell
model=convaiinnovations/laya
volume=$PWD/data

docker run --gpus all -p 8080:80 -v $volume:/data --pull always \
  ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
  --model-id $model
```

Decision models are identified by their `rl_agent_config.json` file. TEI downloads the model's tokenizer, encoder configuration, weights, and decision configuration when it starts.

## Make a decision request

Send a text input and an array of named questions to `/v1/decisions`:

```bash
curl http://localhost:8080/v1/decisions \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "convaiinnovations/laya",
    "input": "We were billed twice for March. Please refund the duplicate today.",
    "questions": [{
      "type": "choice",
      "name": "department",
      "instructions": "Which department should handle this request?",
      "choices": [
        {"value": "billing", "description": "Invoices, payments, and refunds."},
        {"value": "technical", "description": "Bugs, outages, and system errors."},
        {"value": "other", "description": "Everything else."}
      ]
    }, {
      "type": "score",
      "name": "urgency",
      "instructions": "How urgent is this request?",
      "levels": [
        {"label": "not urgent"},
        {"label": "soon"},
        {"label": "critical"}
      ]
    }, {
      "type": "predicate",
      "name": "refund_requested",
      "instructions": "Does the user explicitly request a refund?"
    }]
  }'
```

`input` can be a text string or user messages containing `input_text` parts. Each question has a unique `name` and uses one of the OpenAI `predicate`, `choice`, or `score` types. Choice values must be distinct, and score levels are ordered from lowest to highest.

The response contains typed answers in the same order as the request:

```json
{
  "answers": [
    {
      "type": "choice",
      "name": "department",
      "choice": "billing",
      "probabilities": [
        {"value": "billing", "probability": 0.94},
        {"value": "technical", "probability": 0.02},
        {"value": "other", "probability": 0.04}
      ],
      "confidence": 0.79
    },
    {
      "type": "score",
      "name": "urgency",
      "score": 1.1,
      "probabilities": [
        {"value": 0, "label": "not urgent", "probability": 0.1},
        {"value": 1, "label": "soon", "probability": 0.7},
        {"value": 2, "label": "critical", "probability": 0.2}
      ],
      "confidence": 0.55
    },
    {
      "type": "predicate",
      "name": "refund_requested",
      "probability": 0.84
    }
  ]
}
```

Image parts are rejected because the served decision models accept text input only.

## Model architecture

Laya uses a ModernBERT encoder followed by a decision head. TEI converts each question into a prompt containing the state, instructions, and one `[MASK]` marker for every candidate option. The model produces one score at each marker; temperature scaling and softmax turn those scores into option probabilities.

The decision head also receives a learned embedding for the question type and applies a configurable stack of transformer layers. An action head combines the `[CLS]` representation with the maximum option probability, the gap between the top two probabilities, normalized entropy, and the number of options. It returns the recommended action and its probability alongside the question result. Confidence is derived from the normalized entropy of the option distribution.

See the [Laya repository](https://github.com/NandhaKishorM/laya) and its [architecture explanation](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me#:~:text=3.%20Model%20Architecture%3A%20421M%20Parameters) for more background.
