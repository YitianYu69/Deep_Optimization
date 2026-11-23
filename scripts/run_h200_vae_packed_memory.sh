#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_packed_memory.sh original_result.json --experiment combined|workspace|selection|arena --host-pinned-mib N --device-staging-mib N --d2h-inflight-mib N [--trials 6]" >&2
  exit 2
fi
previous=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")
shift
bundle_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$bundle_root"
exec python -B benchmarks/compiler/run_packed_memory_experiments.py "$previous" "$@"
