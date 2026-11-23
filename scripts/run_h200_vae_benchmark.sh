#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_benchmark.sh /path/to/original_result.json [benchmark options]" >&2
  exit 2
fi
for argument in "${@:2}"; do
  if [[ "$argument" == --output || "$argument" == --output=* ]]; then
    echo "This launcher archives its generated output; use the direct Python runner to set --output" >&2
    exit 2
  fi
  if [[ "$argument" == --arm-manifest || "$argument" == --arm-manifest=* || "$argument" == --expected-version || "$argument" == --expected-version=* ]]; then
    echo "This launcher pins the bundled candidate; use the direct Python runner for arm overrides" >&2
    exit 2
  fi
done
previous=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")
shift
bundle_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$bundle_root"

python - <<'PY'
try:
    import torch
except ImportError as error:
    raise SystemExit("Activate your CUDA PyTorch environment before running this benchmark") from error
if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
    raise SystemExit("The H200 benchmark requires a CUDA GPU with BF16 support")
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
output="work/h200_vae_eager_native_${stamp}.json"
log="work/h200_vae_eager_native_${stamp}.log"
if [[ -e "$output" || -e "$log" || -e "${output%.json}.zip" || -e "${output%.json}-artifacts" ]]; then
  echo "Benchmark destination already exists; historical results are preserved" >&2
  exit 2
fi
mkdir -p work
pin="${output%.json}-arm.json"
has_pin=$(python - "$wheel" "$pin" <<'PY'
from email.parser import Parser
from pathlib import Path
import hashlib, json, sys, zipfile

wheel, pin = map(Path, sys.argv[1:])
manifest = json.loads(Path("manifest.json").read_text())
selected = manifest.get("version")
if selected is None or selected == "0.1.0+nativeautodiff.20261119.vaememory.4":
    print("0")
else:
    with zipfile.ZipFile(wheel) as archive:
        metadata_name, = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != selected:
            raise SystemExit("Candidate wheel metadata differs from the bundle version")
        package = {name.removeprefix("deepopt/"): archive.read(name) for name in archive.namelist()
                   if name.startswith("deepopt/") and not name.endswith("/")}
    if "__init__.py" not in package:
        raise SystemExit("Candidate wheel has no DeepOpt package")
    digest = hashlib.sha256()
    for name in sorted(package, key=lambda value: Path(value).parts):
        digest.update(name.encode() + b"\0" + package[name] + b"\0")
    value = {"schema_version": 1, "arm": "paired_candidate", "expected_installed": {
        "version": selected, "source_sha256": digest.hexdigest(), "files_hashed": len(package)},
        "wheel_sha256": manifest["wheel_sha256"], "native_options_sha256": None}
    with pin.open("x") as stream:
        json.dump(value, stream, indent=2)
    print("1")
PY
)
if [[ "$has_pin" == 1 ]]; then
  set -- --arm-manifest "$pin" "$@"
fi
set +e
env -u TORCHDYNAMO_REPRO_AFTER TORCHDYNAMO_REPRO_LEVEL=0 \
  python -B examples/vae/benchmark_execution.py --from-result "$previous" \
    --device cuda:0 --batch-size 128 --warmup 5 --updates 10 --trials 1 --output "$output" "$@" 2>&1 | tee "$log"
status=$?
set -e
python - "$output" "$log" "$pin" <<'PY'
from pathlib import Path
import sys, zipfile
result, log, pin = map(Path, sys.argv[1:])
artifacts = result.parent / (result.stem + "-artifacts")
archive = result.with_suffix(".zip")
with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
    for path in (result, log, pin):
        if path.is_file():
            bundle.write(path, path.name)
    for path in sorted(artifacts.rglob("*")):
        if path.is_file() and not path.is_symlink() and path.suffix in (".json", ".log", ".py", ".txt", ".ttir", ".ttgir", ".llir", ".ptx", ".sass"):
            bundle.write(path, path.relative_to(result.parent))
print(f"Attach this eager/native benchmark archive: {archive.resolve()}")
PY
exit "$status"
