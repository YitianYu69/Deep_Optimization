#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"
# Candidate sources/wheel are pinned in manifest.json. Device preflight runs
# before pip, and --target keeps the caller's environment package unchanged.
exec "${PYTHON:-python}" -B benchmarks/compiler/qualify_packed_offload_ddp.py \
  --install --require-h200 "$@"
