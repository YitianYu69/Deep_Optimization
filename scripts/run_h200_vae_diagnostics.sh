#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_diagnostics.sh /path/to/previous_result.json [diagnostic options]" >&2
  exit 2
fi
previous=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")
shift
bundle_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$bundle_root"

python - <<'PY'
try:
    import torch
except ImportError as error:
    raise SystemExit("Activate your CUDA PyTorch environment before running this script") from error
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: activate your H200 environment and request a GPU")
PY

wheel=$(python - <<'PY'
from pathlib import Path
import hashlib, json
root = Path.cwd()
manifest = json.loads((root / "manifest.json").read_text())
wheel = root / "packages" / Path(manifest["wheel"]).name
if hashlib.sha256(wheel.read_bytes()).hexdigest() != manifest["wheel_sha256"]:
    raise SystemExit("Bundled wheel checksum differs from manifest.json")
print(wheel)
PY
)
python -m pip install --no-deps --force-reinstall "$wheel"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
output="work/h200_vae_diagnostics_${stamp}.json"
preflight="work/h200_vae_diagnostics_${stamp}-saved-abi.json"
preflight_log="work/h200_vae_diagnostics_${stamp}-preflight.log"
if [[ -e "$output" || -e "$preflight" || -e "$preflight_log" || -e "${output%.json}.zip" || -e "${output%.json}-artifacts" ]]; then
  echo "Diagnostic destination already exists; historical results are preserved" >&2
  exit 2
fi
mkdir -p work
set +e
env -u TORCHDYNAMO_REPRO_AFTER TORCHDYNAMO_REPRO_LEVEL=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python -B benchmarks/compiler/qualification_tests.py \
  --modules tests.compiler.test_native_activation_saved_abi.NativeActivationSavedABICudaTests \
    tests.opt.test_padding.MetadataPaddingStrictZeroCudaTests \
    tests.compiler.test_native_padding.NativePaddingStrictZeroCudaTests \
    tests.compiler.test_native_deterministic_padding_capture.NativeDeterministicPaddingCaptureCudaTests \
  --device cuda:0 --require-cuda --fail-on-skip --output "$preflight" 2>&1 | tee "$preflight_log"
status=$?
if [[ $status -eq 0 ]]; then
  python -B examples/vae/benchmark_vae.py --diagnose \
    --diagnostic-from-result "$previous" --output "$output" "$@"
  status=$?
else
  python - "$output" "$preflight" <<'PY'
from pathlib import Path
import json, sys
Path(sys.argv[1]).write_text(json.dumps({
    "status": "blocked", "scope": "untimed_numerical_diagnostics",
    "qualification_eligible": False, "timing_eligible": False,
    "error": "CUDA saved-activation/padding preflight failed; full-model cases were not started",
    "preflight": sys.argv[2], "runs": []}, indent=2) + "\n")
PY
fi
set -e
python - "$output" "$preflight" "$preflight_log" <<'PY'
from pathlib import Path
import sys, zipfile
result = Path(sys.argv[1])
artifacts = result.parent / (result.stem + "-artifacts")
if result.is_file():
    archive = result.with_suffix(".zip")
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(result, result.name)
        preflight = Path(sys.argv[2])
        if preflight.is_file():
            bundle.write(preflight, preflight.name)
        preflight_log = Path(sys.argv[3])
        if preflight_log.is_file():
            bundle.write(preflight_log, preflight_log.name)
        for path in sorted(artifacts.rglob("*")):
            if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".log", ".py"):
                bundle.write(path, path.relative_to(result.parent))
    print(f"Attach this metadata/code diagnostic archive: {archive.resolve()}")
PY
exit "$status"
