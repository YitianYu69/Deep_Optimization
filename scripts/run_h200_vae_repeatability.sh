#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_repeatability.sh /path/to/original_result.json [diagnostic options]" >&2
  exit 2
fi
script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
bash "$script_dir/run_h200_vae_diagnostics.sh" "$1" \
  --diagnostic-deterministic \
  --diagnostic-cases backward_repeatability dense_eager_repeat dense_native_repeat dense_eager_native \
  --diagnostic-trace-modules moments encoder.14 encoder.13 encoder.11.body.9 "${@:2}"
