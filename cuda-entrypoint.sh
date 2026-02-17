#!/bin/bash

if ! command -v nvidia-smi &>/dev/null; then
    echo "Error: 'nvidia-smi' command not found."
    exit 1
fi

# NOTE: Given that we need to support CUDA versions earlier than CUDA 12.9.1, we
# need to include the `cuda-compat-12-9` in `LD_LIBRARY_PATH` when the host CUDA
# version is lower than that; whilst we shouldn't include that when CUDA is 13.0+
# as otherwise it will fail due to it.
if [ -d /usr/local/cuda/compat ]; then
    DRIVER_CUDA=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $3; exit}')

    IFS='.' read -r MAJ MIN PATCH <<EOF
${DRIVER_CUDA:-0.0.0}
EOF
    : "${MIN:=0}"
    : "${PATCH:=0}"

    DRIVER_INT=$((10#${MAJ} * 10000 + 10#${MIN} * 100 + 10#${PATCH}))
    TARGET_INT=$((12 * 10000 + 9 * 100 + 1))

    if [ "$DRIVER_INT" -lt "$TARGET_INT" ]; then
        export LD_LIBRARY_PATH="/usr/local/cuda/compat:${LD_LIBRARY_PATH}"
    fi
fi

exec text-embeddings-router "$@"
