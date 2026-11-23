#!/usr/bin/env bash
# Keep the runner and oracle from this bundle together; preserve launch cwd.
set -euo pipefail
diagnostic_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
diagnostic_python="${BENCHMARK_PYTHON:-python}"
"$diagnostic_python" "$diagnostic_dir/diagnose_training_stability.py" --version
exec "$diagnostic_python" "$diagnostic_dir/diagnose_training_stability.py" "$@"
