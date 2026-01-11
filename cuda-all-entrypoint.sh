#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 "$@"
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 87 ]; then
    exec text-embeddings-router-80 "$@"
elif [ ${compute_cap} -eq 87 ]; then
    exec text-embeddings-router-87 "$@"
elif [ ${compute_cap} -eq 89 ]; then
    exec text-embeddings-router-89 "$@"
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 "$@"
else
    echo "cuda compute cap ${compute_cap} is not supported"
    exit 1
fi
