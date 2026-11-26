#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

# On Jetson L4T, CUDA libraries are provided by the host via nvidia-container-runtime.
# Add compat path if it exists.
if [ -d /usr/local/cuda/compat ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

if [ ${compute_cap} -eq 87 ]; then
    exec text-embeddings-router-87 "$@"
else
    echo "cuda compute cap ${compute_cap} is not supported by the Jetson image (supported: 87)"
    exit 1
fi
