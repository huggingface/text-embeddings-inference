#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

# Function to compare version numbers
verlte() {
    [ "$1" = "$2" ] && return 1 || [ "$2" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

# CUDA compat libs logic
if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d"." -f 3-)
    echo "CUDA compat package requires NVIDIA driver ≤ ${CUDA_COMPAT_MAX_DRIVER_VERSION}"
    cat /proc/driver/nvidia/version
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM version:.* \([0-9]\+\.[0-9]\+\.[0-9]\+\) .*/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    echo "Current installed NVIDIA driver version is ${NVIDIA_DRIVER_VERSION}"
    if [ $(verlte "$CUDA_COMPAT_MAX_DRIVER_VERSION" "$NVIDIA_DRIVER_VERSION") ]; then
        echo "Setup CUDA compatibility libs path to LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
        echo $LD_LIBRARY_PATH
    else
        echo "Skip CUDA compat libs setup as newer NVIDIA driver is installed"
    fi
else
    echo "Skip CUDA compat libs setup as package not found"
fi

if [[ -z "${HF_MODEL_ID}" ]]; then
    echo "HF_MODEL_ID must be set"
    exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"

if [[ -n "${HF_MODEL_REVISION}" ]]; then
    export REVISION="${HF_MODEL_REVISION}"
fi

compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv | sed -n '2p' | sed 's/\.//g')
if [ ${compute_cap} -eq 75 ]; then
    exec text-embeddings-router-75 --port 8080 --json-output
elif [ ${compute_cap} -ge 80 -a ${compute_cap} -lt 90 ]; then
    exec text-embeddings-router-80 --port 8080 --json-output
elif [ ${compute_cap} -eq 90 ]; then
    exec text-embeddings-router-90 --port 8080 --json-output
elif [ ${compute_cap} -eq 100 ]; then
    exec text-embeddings-router-100 --port 8080 --json-output
elif [ ${compute_cap} -eq 120 ]; then
    exec text-embeddings-router-120 --port 8080 --json-output
else
    echo "CUDA compute cap ${compute_cap} is not supported"
    exit 1
fi
