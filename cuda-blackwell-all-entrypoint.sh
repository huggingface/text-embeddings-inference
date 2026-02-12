#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 100 ]; then
    exec text-embeddings-router-100 "$@"
elif [ ${compute_cap} -eq 120 ]; then
    exec text-embeddings-router-120 "$@"
else
    echo "cuda compute cap ${compute_cap} is not supported"
    exit 1
fi
