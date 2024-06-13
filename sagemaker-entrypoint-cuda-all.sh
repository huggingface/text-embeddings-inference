#!/bin/bash

verlte() {
    [ "$1" = "$2" ] && return 1 || [ "$2" = "$(echo -e "$1\n$2" | sort -V | head -n1)" ]
}

if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 | cut -d"." -f 3-)
    echo "CUDA compat package requires Nvidia driver ≤${CUDA_COMPAT_MAX_DRIVER_VERSION}"
    cat /proc/driver/nvidia/version
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    echo "Current installed Nvidia driver version is ${NVIDIA_DRIVER_VERSION}"
    if [ $(verlte "$CUDA_COMPAT_MAX_DRIVER_VERSION" "$NVIDIA_DRIVER_VERSION") ]; then
        echo "Setup CUDA compatibility libs path to LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
        echo $LD_LIBRARY_PATH
    else
        echo "Skip CUDA compat libs setup as newer Nvidia driver is installed"
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

if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

# Query GPU name using nvidia-smi
gpu_name=$(nvidia-smi --query-gpu=gpu_name --format=csv | awk 'NR==2')
if [ $? -ne 0 ]; then
    echo "Error: $gpu_name"
    echo "Query gpu_name failed"
else
    echo "Query gpu_name succeeded. Printing output: $gpu_name"
fi

# Function to get compute capability based on GPU name
get_compute_cap() {
    gpu_name="$1"

    # Check if the GPU name contains "A10G"
    if [[ "$gpu_name" == *"A10G"* ]]; then
        echo "86"
    # Check if the GPU name contains "A100"
    elif [[ "$gpu_name" == *"A100"* ]]; then
        echo "80"
    # Check if the GPU name contains "H100"
    elif [[ "$gpu_name" == *"H100"* ]]; then
        echo "90"
    # Cover Nvidia T4
    elif [[ "$gpu_name" == *"T4"* ]]; then
        echo "75"
    # Cover Nvidia L4
    elif [[ "$gpu_name" == *"L4"* ]]; then
        echo "89"
    else
        echo "80"  # Default compute capability
    fi
}

if [[ -z "${CUDA_COMPUTE_CAP}" ]]
then
    compute_cap=$(get_compute_cap "$gpu_name")
    echo "the compute_cap is $compute_cap"
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
