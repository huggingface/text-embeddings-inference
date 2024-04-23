#!/bin/bash

if ! command -v nvidia-smi &> /dev/null; then
    # It does not matter which router we exec, the model will be served on CPU
    exec text-embeddings-router-75 "$@"
else
    compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')

    if [ ${compute_cap} -eq 75 ]
    then
        exec text-embeddings-router-75 "$@"
    elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]
    then
        exec text-embeddings-router-80 "$@"
    elif [ ${compute_cap} -eq 90 ]
    then
        exec text-embeddings-router-90 "$@"
    else
        echo "cuda compute cap ${compute_cap} is not supported"; exit 1
    fi
fi
