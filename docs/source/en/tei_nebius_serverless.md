<!--Copyright 2026 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contains specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Deploying TEI on Nebius Serverless AI Endpoints

[Nebius Serverless AI Endpoints](https://docs.nebius.com/serverless/overview) run a container behind a managed HTTPS URL. This guide configures one NVIDIA L40S GPU to serve `BAAI/bge-small-en-v1.5`, then covers embedding requests, batching, stopping, restarting, and deletion.

An Endpoint runs until you stop or delete it. Allocated resources are billed while it runs, including while idle; these instructions do not configure automatic scaling. See [Serverless pricing and quotas](https://docs.nebius.com/serverless/pricing-quotas).

> [!NOTE]
> Validated on September 11, 2026 with Nebius CLI `0.12.265`, one regular L40S in `eu-north1`, and the image/model pins below. Checks covered GPU startup, authenticated native/OpenAI embedding requests, input limits, small batching samples, stop/start recovery, and deletion of the Endpoint, VM and boot disk. These checks do not establish production throughput or memory sizing.

## Prerequisites

- A Nebius project with billing enabled and an identity authorized to manage Endpoints. The [Endpoint management guide](https://docs.nebius.com/serverless/endpoints/manage) describes the required permissions.
- The [Nebius CLI](https://docs.nebius.com/cli/install), configured for that project. These examples use the command interface in version `0.12.265`.
- Bash, curl 7.76 or newer, jq 1.6 or newer, and OpenSSL on your local machine.
- A subnet in `eu-north1` and quota/capacity for one `gpu-l40s-a` instance with preset `1gpu-8vcpu-32gb`. This supplies one L40S, 8 vCPUs, and 32 GiB of host RAM. Published platform support does not guarantee available capacity; consult the [platform and preset documentation](https://docs.nebius.com/compute/virtual-machines/types).
- Outbound access from the workload to `ghcr.io` and Hugging Face for the image and model download. A managed HTTPS URL does not establish that this outbound access is configured.

The model is public and ungated; this example needs no Hugging Face token or private-registry credentials. Your Nebius CLI identity and the Endpoint bearer token have different purposes: the CLI identity manages the resource, while the Endpoint token authenticates inference requests.

## Select the image and model

The example pins an official TEI Ada GPU image built from upstream commit [`e2e051a`](https://github.com/huggingface/text-embeddings-inference/commit/e2e051afda1dbc8993979feae5f061eba80900d3). It is a commit build, not a numbered release. Its CUDA entrypoint includes NVIDIA tool discovery and driver-version parsing fixes absent from TEI 1.9.3. Keep that entrypoint rather than overriding it with the router binary.

| Setting | Value |
|---|---|
| Image platform | `linux/amd64`, Ada GPU image (`89-sha-e2e051a`) |
| Image manifest | `sha256:3dfbc9b6af7087f8f7737a2a4bb5277f8086c45b21086911313b7efbbd1741c8` |
| Model | [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a` |
| Output | 384-dimensional embeddings, CLS pooling |
| Model input limit | 512 token positions, including special tokens |

Use the matching image from [supported hardware](supported_models) if you change GPU architecture. This image contains CUDA 12.9.1 and compatibility libraries; host driver/library compatibility still needs verification during startup. Changing an image or model pin requires rechecking startup and inference.

## Create the Endpoint

Run the examples in the same Bash session. Fill in the project and subnet IDs explicitly:

```bash
export PROJECT_ID='<project-id>'
export SUBNET_ID='<subnet-id>'
export ENDPOINT_NAME="tei-embeddings-$(date +%s)"
export TEI_IMAGE='ghcr.io/huggingface/text-embeddings-inference@sha256:3dfbc9b6af7087f8f7737a2a4bb5277f8086c45b21086911313b7efbbd1741c8'
export MODEL_ID='BAAI/bge-small-en-v1.5'
export MODEL_REVISION='5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'

# Keep the token locally without printing it. Do not enable shell tracing.
umask 077
export RUN_DIR
RUN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tei-nebius.XXXXXX")
export ENDPOINT_TOKEN
ENDPOINT_TOKEN=$(openssl rand -hex 32)
printf '%s' "$ENDPOINT_TOKEN" > "$RUN_DIR/endpoint-token"
```

The temporary directory contains a secret. Keep it out of source control and shared logs. The `--token` argument may be visible to other local processes; on a shared machine, use the CLI's `--token-secret` option with a properly configured SecretStash secret instead. See [Endpoint authentication options](https://docs.nebius.com/serverless/endpoints/manage).

Create one regular GPU Endpoint. TEI reads its configuration from environment variables, preserving the official image entrypoint:

```bash
nebius ai endpoint create \
  --parent-id "$PROJECT_ID" --subnet-id "$SUBNET_ID" \
  --name "$ENDPOINT_NAME" --image "$TEI_IMAGE" \
  --platform gpu-l40s-a --preset 1gpu-8vcpu-32gb \
  --container-port 8080 \
  --env "MODEL_ID=$MODEL_ID" --env "REVISION=$MODEL_REVISION" \
  --env 'HOSTNAME=0.0.0.0' --env 'PORT=8080' \
  --env 'DTYPE=float16' --env 'AUTO_TRUNCATE=false' \
  --env 'MAX_BATCH_TOKENS=4096' --env 'MAX_CLIENT_BATCH_SIZE=16' \
  --env 'MAX_CONCURRENT_REQUESTS=32' --env 'TOKENIZATION_WORKERS=4' \
  --auth token --token "$ENDPOINT_TOKEN" \
  --disk-size 250Gi --shm-size 16Gi \
  --public=false --preemptible=false --retries 1 --async
```

`--async` returns after submission, before model readiness. The CLI output can include the Endpoint token, including when `--format json` is requested; keep create/get/lifecycle output out of shared logs. Copy the returned Endpoint ID and save it with the project ID:

```bash
export ENDPOINT_ID='<endpoint-id>'
jq -n --arg project "$PROJECT_ID" --arg id "$ENDPOINT_ID" \
  --arg image "$TEI_IMAGE" --arg model "$MODEL_ID" --arg revision "$MODEL_REVISION" \
  '{project_id:$project, endpoint_id:$id, image:$image, model:$model, revision:$revision}' \
  > "$RUN_DIR/receipt.json"
```

If submission times out or its outcome is unclear, inspect `nebius ai endpoint list --parent-id "$PROJECT_ID"` or the Serverless console before creating another resource. Endpoint names are not unique: reconcile the ID, creation time and configuration, including any further result pages. `--retries 1` disables automatic CLI retries for the mutation; it does not cancel a submitted operation.

`--public=false` avoids a VM public address. HTTP traffic still uses the managed HTTPS URL. Do not construct a URL from a VM address or send the token over plaintext HTTP.

The disk and shared-memory sizes are conservative starting values, not TEI minimums. Allow space for image extraction, model downloads and logs. Files in the container, including the Hub cache at `/data`, are disposable. Store returned embeddings in your application's durable database.

## Wait for readiness

The following function checks the control-plane state and the health route through the managed URL. It stops after a 15-minute polling window; individual CLI/HTTP requests have their own timeouts. It does not cancel or delete the Endpoint when it returns an error.

```bash
wait_for_tei() {
  local deadline state endpoint_json urls count code
  deadline=$(( $(date +%s) + 900 ))
  ENDPOINT_URL=''
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! endpoint_json=$(nebius ai endpoint get "$ENDPOINT_ID" \
      --format json --retries 1 --timeout 15s --auth-timeout 15s --no-browser); then
      printf 'Cannot read Endpoint state; reconcile the resource before continuing.\n' >&2
      return 1
    fi
    if ! state=$(printf '%s' "$endpoint_json" | jq -er '.status.state'); then
      printf 'Endpoint response has no state.\n' >&2
      return 1
    fi
    case "$state" in
      PROVISIONING|STARTING|IMAGE_PULLING) ;;
      RUNNING)
        if ! urls=$(printf '%s' "$endpoint_json" | jq -c \
          '[.status.public_endpoints[]? | select(type == "string") | select(startswith("https://"))]'); then
          return 1
        fi
        count=$(printf '%s' "$urls" | jq 'length')
        if [ "$count" -gt 1 ]; then
          printf 'Multiple HTTPS URLs found; select the configured HTTP port explicitly.\n' >&2
          return 1
        fi
        if [ "$count" -eq 1 ]; then
          ENDPOINT_URL=$(printf '%s' "$urls" | jq -r '.[0]')
          ENDPOINT_URL=${ENDPOINT_URL%/}
          if code=$(curl --silent --show-error --connect-timeout 5 --max-time 15 \
            -o /dev/null -w '%{http_code}' \
            -H "Authorization: Bearer $ENDPOINT_TOKEN" "$ENDPOINT_URL/health"); then
            case "$code" in
              200) export ENDPOINT_URL; return 0 ;;
              401|403)
                printf 'Endpoint rejected the token.\n' >&2
                return 1 ;;
              404|408|429|500|502|503|504) ;;
              *) printf 'Unexpected health HTTP status: %s\n' "$code" >&2; return 1 ;;
            esac
          fi
        fi ;;
      *) printf 'Endpoint state is %s; inspect its status details and logs.\n' "$state" >&2; return 1 ;;
    esac
    sleep 5
  done
  printf 'Readiness deadline exceeded; reconcile and stop or delete the Endpoint.\n' >&2
  return 1
}

wait_for_tei
```

Proceed only if the function succeeds. Confirm that inference works as well; a control-plane `RUNNING` state or a health response alone is not the complete smoke test:

```bash
curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 30 \
  -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' "$ENDPOINT_URL/embed" \
  -d '{"inputs":"Readiness check","normalize":true,"truncate":false}' \
  > "$RUN_DIR/readiness.json" &&
jq -e 'type == "array" and length == 1 and
  all(.[]; type == "array" and length == 384 and
    all(.[]; type == "number" and . > -1e100 and . < 1e100))' \
  "$RUN_DIR/readiness.json"
```

Require both commands to succeed before sending application traffic. A failure leaves the Endpoint running; follow the cleanup section if you are abandoning the deployment. Capacity provisioning can take up to 30 minutes before the documented `NotEnoughResources` error, so this shorter readiness deadline does not prove a permanent capacity failure. See [lifecycle and statuses](https://docs.nebius.com/serverless/lifecycle).

## Request embeddings

The native `/embed` endpoint accepts `inputs`, either a string or an array. This example returns two vectors in input order:

```bash
curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 30 \
  -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' "$ENDPOINT_URL/embed" \
  -d '{"inputs":["What is deep learning?","Paris is in France."],"normalize":true,"truncate":false}'
```

TEI also exposes an OpenAI-compatible `/v1/embeddings` route, which uses `input`:

```bash
curl --fail-with-body --silent --show-error --connect-timeout 5 --max-time 30 \
  -H "Authorization: Bearer $ENDPOINT_TOKEN" \
  -H 'Content-Type: application/json' "$ENDPOINT_URL/v1/embeddings" \
  -d '{"model":"BAAI/bge-small-en-v1.5","input":["What is deep learning?","Paris is in France."],"encoding_format":"float"}'
```

The OpenAI-compatible response contains `data[].index` and `data[].embedding`. Its `model` field does not switch the model loaded by this Endpoint. An OpenAI SDK client would use `${ENDPOINT_URL}/v1` as its base URL, not the full `/v1/embeddings` path.

For retrieval, follow the [BGE model card's query/passage instructions](https://huggingface.co/BAAI/bge-small-en-v1.5). Do not apply a query-only instruction to every document. Keep the model revision and preprocessing consistent between stored vectors and queries. Chunk longer inputs with a tokenizer before the model limit; automatic truncation is disabled here to avoid silently dropping text.

Before accepting application traffic, verify that missing and incorrect Endpoint tokens are rejected by the managed URL and the correct token succeeds. This example relies on Nebius ingress authentication and does not set TEI's separate `API_KEY`. TEI's own API-key middleware does not protect its health and metrics routes; do not expose its native port as an alternative public access path. If you enable both authentication layers later, verify how the ingress forwards the authorization header.

## Batching and resource sizing

TEI performs dynamic batching within this single container. The controls have different meanings:

| Setting | Example value | Controls |
|---|---:|---|
| `MAX_BATCH_TOKENS` | 4096 | Total token work in a server batch |
| `MAX_CLIENT_BATCH_SIZE` | 16 | Maximum number of inputs in one client request |
| `MAX_CONCURRENT_REQUESTS` | 32 | Concurrent requests accepted by TEI |
| `TOKENIZATION_WORKERS` | 4 | CPU workers for tokenization |

These are starting points, not measured optimal values. Test 1, 4 and 16 inputs per request and concurrency 1, 2 and 4 with representative sequence lengths. Observe GPU memory, tokenization load, latency and error rate before increasing limits. The server token budget is not the per-input model limit. See [CLI arguments](cli_arguments).

TEI's default payload cap is 2,000,000 bytes; ingress may impose additional limits. Excessive client batches, oversized or overlong inputs, and malformed payloads should be corrected rather than retried unchanged. For transient overload/network errors, use bounded backoff and application-side deduplication when persisting results. A retry can repeat GPU computation.

## Stop, restart and delete

Stop sending traffic before stopping the Endpoint. In-flight requests may fail; there is no durable request queue in this example.

```bash
nebius ai endpoint stop --id "$ENDPOINT_ID" --retries 1 --timeout 15m
nebius ai endpoint get "$ENDPOINT_ID" --format json | jq -r '.status.state'
```

Require successful completion of the stop command and `STOPPED` before starting again. If the command times out, it has not cancelled the operation: inspect operation progress in the Serverless console before another mutation. A `STOPPED` observation alone is not a substitute for operation completion. Avoid overlapping stop/start/delete requests.

Stopping releases the associated VM and boot disk, so plan for a fresh image/model download on restart. Mounted buckets and shared filesystems are separate resources and remain billable. See [Endpoint management](https://docs.nebius.com/serverless/endpoints/manage).

```bash
nebius ai endpoint start --id "$ENDPOINT_ID" --retries 1 --timeout 15m
```

After the start operation completes, run `wait_for_tei` again to rediscover the managed URL, then repeat the embedding smoke test. Do not assume the old URL or cache survived. A timed-out start also needs reconciliation before further mutation.

When finished, delete the Endpoint and verify its removal:

```bash
nebius ai endpoint delete --id "$ENDPOINT_ID" --retries 1 --timeout 15m
nebius ai endpoint get "$ENDPOINT_ID" --format json
```

Require successful deletion and a `NotFound` result for the exact Endpoint ID. An authentication or network error is not proof of deletion. Verify the operation has released its resources in the console; keep the receipt if cleanup is still in progress or fails. Do not delete unrelated or shared volumes. Once cleanup is confirmed, remove the local token and unset it:

```bash
rm -- "$RUN_DIR/endpoint-token"
unset ENDPOINT_TOKEN
```

Closing the terminal, losing the local process, or reaching a polling deadline does not stop the Endpoint. Keep the receipt available and explicitly reconcile and delete abandoned deployments.

## Troubleshooting

| Symptom | What to check |
|---|---|
| Prolonged `PROVISIONING` | Project quota and capacity for the selected platform/preset; inspect status details before another create. |
| Image pull or model download failure | Registry/Hub reachability, the exact image/model revision, and sufficient boot-disk space. |
| Missing `nvidia-smi` or CUDA loader error | The pinned image, host driver and mounted NVIDIA tools/libraries. Preserve the official entrypoint. |
| `RUNNING` but health/inference fails | Container port 8080, the discovered managed URL, model-loading logs and the Endpoint token. |
| HTTP 401/403 | Use the token belonging to this Endpoint, not the CLI IAM credential or a Hub token. |
| Memory pressure or overload | Reduce batch-token and concurrency limits; retest with realistic input lengths. |
| Lifecycle timeout or operation conflict | Inspect the existing operation in the console and wait for completion before a new mutation. |

Use the [Serverless failure guide](https://docs.nebius.com/serverless/jobs/failure) for logs and diagnostics. Do not keep a failed container alive with an indefinite sleep just to hide a startup error.

## Reranking

Reranking is a separate deployment choice: it requires a suitable sequence-classification model, such as `BAAI/bge-reranker-base`, rather than the embedding model used above. See [supported models](supported_models) and the [reranker quick tour](quick_tour#re-rankers).

For a reranker Endpoint, pin its own model revision and validate `/rerank` with a `query` and `texts` payload, score/index mapping, combined query/document token limits and memory usage. A successful embedding deployment does not validate a reranker. Do not send `/rerank` requests to this BGE embedding Endpoint or claim reranking support from the embedding smoke test.
