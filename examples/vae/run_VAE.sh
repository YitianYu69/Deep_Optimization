#!/usr/bin/env bash
set -euo pipefail

export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export TORCHINDUCTOR_COMPILE_THREADS=1
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

echo "Using CC: $CC"
echo "Using CXX: $CXX"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAE_CONFIG="${VAE_CONFIG:-$SCRIPT_DIR/imagenet.toml}"
OUT_DIR="${OUT_DIR:-runs/vae}"

TRAIN_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
    TRAIN_ARGS+=(--resume "$RESUME")
else
    TRAIN_ARGS+=(--config "$VAE_CONFIG" --output "$OUT_DIR")
    if [[ -n "${INIT_WEIGHTS:-}" ]]; then
        TRAIN_ARGS+=(--init-weights "$INIT_WEIGHTS")
    fi
fi

torchrun --standalone --nproc-per-node="${NPROC_PER_NODE:-4}" \
    "$SCRIPT_DIR/train_vae.py" \
    "${TRAIN_ARGS[@]}" "$@"
