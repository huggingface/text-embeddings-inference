#!/bin/bash

if [[ -z "${HF_MODEL_ID}" ]]; then
  echo "HF_MODEL_ID must be set"
  exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
  export REVISION="${HF_MODEL_REVISION}"
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

if [[ -z "${CUDA_COMPUTE_CAP}" ]]
then
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')
else
    compute_cap=$CUDA_COMPUTE_CAP
fi

if [[ ${compute_cap} -eq 75 ]]
then
    text-embeddings-router-75 --port 8080 --json-output
elif [[ ${compute_cap} -ge 80 && ${compute_cap} -lt 90 ]]
then
    text-embeddings-router-80 --port 8080 --json-output
elif [[ ${compute_cap} -eq 90 ]]
then
    text-embeddings-router-90 --port 8080 --json-output
else
    echo "cuda compute cap ${compute_cap} is not supported"; exit 1
fi
