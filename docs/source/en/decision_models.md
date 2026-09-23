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
> Decision models use the `/v1/decide` HTTP endpoint. The HTTP server does not expose embedding, reranking, or prediction routes for them.

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

Send the input state and a map of named questions to `/v1/decide`:

```bash
curl http://localhost:8080/v1/decide \
  -X POST \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {
      "subject": "Duplicate charge on invoice #4411",
      "body": "We were billed twice for March. Please refund the duplicate today."
    },
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {
          "billing": "invoices, payments, refunds",
          "technical": "bugs, outages, system errors",
          "other": "everything else"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical"]
      },
      "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?"
      }
    }
  }'
```

`state` can be a string or a JSON value. Each question has an `instructions` field and a type-specific `criteria` field:

- `choice` accepts a JSON object whose keys are option labels and whose values describe the options.
- `score` accepts an array of ordered criteria. The response includes the expected score and a probability for each level.
- `noul` is a binary question. Its two options are `false` and `true`; criteria descriptions are optional.

A response contains an answer for each question:

```json
{
  "answers": {
    "department": {
      "type": "choice",
      "label": "billing",
      "probabilities": {
        "billing": 0.94,
        "technical": 0.02,
        "other": 0.04
      },
      "confidence": 0.79,
      "action": 0,
      "action_probability": 1.0
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.84,
      "confidence": 0.84,
      "action": 0,
      "action_probability": 1.0
    }
  }
}
```

The response order is not significant because `answers` is a map keyed by question name.

## Model architecture

Laya uses a ModernBERT encoder followed by a decision head. TEI converts each question into a prompt containing the state, instructions, and one `[MASK]` marker for every candidate option. The model produces one score at each marker; temperature scaling and softmax turn those scores into option probabilities.

The decision head also receives a learned embedding for the question type and applies a configurable stack of transformer layers. An action head combines the `[CLS]` representation with the maximum option probability, the gap between the top two probabilities, normalized entropy, and the number of options. It returns the recommended action and its probability alongside the question result. Confidence is derived from the normalized entropy of the option distribution.

See the [Laya repository](https://github.com/NandhaKishorM/laya) and its [architecture explanation](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me#:~:text=3.%20Model%20Architecture%3A%20421M%20Parameters) for more background.
