#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_packed_offload.sh original_result.json --host-pinned-mib N [--device-staging-mib N] [--d2h-inflight-mib N] [--max-host-pinned-mib N] [--host-headroom-mib N] [--trials 6] [profile options]" >&2
  exit 2
fi
for argument in "${@:2}"; do
  case "$argument" in
    --output|--output=*|--experiment-manifest|--experiment-manifest=*|--arm-manifest|--arm-manifest=*|--expected-version|--expected-version=*|--packed-offload-json|--packed-offload-json=*|--device|--device=*|--batch-size|--batch-size=*|--warmup|--warmup=*|--updates|--updates=*|--optimizer|--optimizer=*|--synthetic|--no-memory-audit)
      echo "This launcher preserves the original CUDA B128 recipe; use the direct Python supervisor for explicit diagnostics" >&2
      exit 2
      ;;
  esac
done
previous=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")
shift
bundle_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$bundle_root"
python - "$bundle_root" "$previous" "$@" <<'PY'
"""VAE_PACKED_OFFLOAD_LAUNCHER: same candidate, fresh workers, bounded evidence."""
import argparse
import ast
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile

root, previous = (Path(value).resolve() for value in sys.argv[1:3])
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--host-pinned-mib", type=int, nargs="+", required=True,
                    help="Explicit intermediate physical capacities; multiple values require a maximum ceiling")
for name in ("device-staging-mib", "d2h-inflight-mib"):
    parser.add_argument("--" + name, type=int, default=0,
                        help="Optional hard limit in MiB. Use zero for automatic sizing from the compiled plan.")
parser.add_argument("--host-pageable-mib", type=int, default=0)
parser.add_argument("--host-headroom-mib", type=int, default=8192,
                    help="Explicit physical/job RAM headroom; unavailable limits stop the capacity point")
parser.add_argument("--max-host-pinned-mib", type=int,
                    help="Explicit safe physical ceiling; enables intermediate and all-eligible capacity sweep")
parser.add_argument("--inventory-json", type=Path)
parser.add_argument("--inventory-output", type=Path)
parser.add_argument("--copy-chunk-mib", type=int, default=16)
parser.add_argument("--peak-budget-mib", type=int)
parser.add_argument("--admission", choices=("error", "resident_fallback"), default="resident_fallback")
parser.add_argument("--trials", type=int, default=6)
parser.add_argument("--arms", choices=("all", "demand-only"), default="all")
parser.add_argument("--policy-version", type=int, choices=(1, 2), default=2)
parser.add_argument("--native-options-json", type=Path)
parser.add_argument("--cold-diagnostics", action="store_true")
parser.add_argument("--project-root", type=Path)
parser.add_argument("--train-dir", type=Path)
parser.add_argument("--profile-after-measurement", action="store_true")
parser.add_argument("--profile-updates", type=int, default=1)
parser.add_argument("--profile-memory", action="store_true")
parser.add_argument("--profile-record-shapes", action="store_true")
parser.add_argument("--start-free-tolerance-mib", type=int, default=256)
args = parser.parse_args(sys.argv[3:])
if (any(value < 1 for value in args.host_pinned_mib)
        or min(args.device_staging_mib, args.d2h_inflight_mib) < 0
        or min(args.copy_chunk_mib, args.trials) < 1
        or args.host_pageable_mib < 0 or args.host_headroom_mib < 0 or args.start_free_tolerance_mib < 0
        or (args.max_host_pinned_mib is not None and args.max_host_pinned_mib < max(args.host_pinned_mib))
        or (args.max_host_pinned_mib is None and len(args.host_pinned_mib) != 1)
        or (args.max_host_pinned_mib is not None and args.policy_version != 2)
        or (args.peak_budget_mib is not None and args.peak_budget_mib < 1) or not 1 <= args.profile_updates <= 3):
    parser.error("Host pinned, chunk and block counts must be positive. Transport, pageable and tolerance values must be nonnegative. Profile updates must be 1-3.")
if not args.profile_after_measurement and (args.profile_memory or args.profile_record_shapes or args.profile_updates != 1):
    parser.error("Profile options require --profile-after-measurement")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = root / "work" / f"h200_vae_packed_offload_{stamp}.json"
artifacts = output.parent / (output.stem + "-artifacts")
setup = output.parent / (output.stem + "-launcher")
installs = output.parent / (output.stem + "-installs")
archive = output.with_suffix(".zip")
if any(path.exists() for path in (output, artifacts, setup, installs, archive)):
    raise FileExistsError("Choose a fresh output; saved runs are immutable")
setup.mkdir(parents=True)
receipt = {"schema_version": 1, "scope": "original_H200_B128_same_candidate_packed_offload",
    "status": "running", "stage": "setup", "original_receipt": str(previous),
    "input_shape": [128, 3, 224, 224], "input_requires_grad": False,
    "warmup": 5, "updates": 10, "blocks": args.trials, "commands": [],
    "objective": "minimum_whole_run_GPU_peak_then_latency_at_matched_peak",
    "performance_policy_promoted": False, "tensor_files_saved": 0, "install_trees_archived": False}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save():
    (setup / "launcher.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

def run(command, log, environment):
    receipt["commands"].append({"command": command, "stage": receipt["stage"], "log": str(log)})
    save()
    with log.open("x") as stream:
        with subprocess.Popen(command, cwd=root, env=environment, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)

def candidate_spec(manifest):
    supplied = root / "packages" / Path(manifest["wheel"]).name
    path = supplied.resolve()
    if (not path.is_relative_to(root / "packages") or supplied.is_symlink()
            or sha(path.read_bytes()) != manifest["wheel_sha256"]):
        raise ValueError("Candidate wheel path/checksum differs from the bundle manifest")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if (len(names) != len(set(names)) or wheel.testzip() is not None
                or any(Path(name).is_absolute() or ".." in Path(name).parts for name in names)):
            raise ValueError("Candidate wheel membership/CRC/path mismatch")
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != manifest["version"]:
            raise ValueError("Candidate distribution/version differs from its bundle pin")
        package = {name.removeprefix("deepopt/"): wheel.read(name) for name in names
            if name.startswith("deepopt/") and not name.endswith("/")}
    for name in ("__init__.py", "_packed_offload.py", "compiler/offload_plan.py", "compiler/runtime/packed_offload.py", "compiler/runtime/manager.py"):
        if name not in package:
            raise ValueError("Candidate wheel lacks required packed offload module: " + name)
    tree = ast.parse(package["_packed_offload.py"])
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PackedActivationOffloadPolicy")
    enabled = next(node.value for node in definition.body if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name) and node.target.id == "enabled")
    if ast.literal_eval(enabled) is not False:
        raise ValueError("Offload must remain disabled in the production defaults")
    digest = hashlib.sha256()
    for name in sorted(package, key=lambda item: Path(item).parts):
        digest.update(name.encode() + b"\0" + package[name] + b"\0")
    return {"wheel": {"path": str(path), "sha256": manifest["wheel_sha256"]},
        "install_root": str(installs / "candidate"), "expected_installed": {"version": manifest["version"],
            "source_sha256": digest.hexdigest(), "files_hashed": len(package)}}

environment = dict(os.environ)
environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
environment.pop("DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY", None)
environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
code = 1
try:
    receipt["stage"] = "packed_offload_profile"
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("packed_offload_profile") != "packed_offload_v1":
        raise ValueError("This bundle has no packed_offload_v1 release profile")
    receipt.update(bundle_manifest_sha256=sha((root / "manifest.json").read_bytes()),
        original_receipt_sha256=sha(previous.read_bytes()))
    candidate = candidate_spec(manifest)
    receipt["candidate"] = candidate
    receipt["stage"] = "original_recipe_validation"
    original = json.loads(previous.read_bytes())
    raw = original["settings"]
    runtime, training, application = raw["runtime"], raw["training"], raw["application"]
    compressed = runtime.get("activation_compression", {})
    if (application["model"] != dict(base_channels=32, max_channels=512, latent_channels=4, blocks_per_stage=2)
            or application["data"]["image_size"] != 224 or training.get("optimizer") != "custom"
            or training.get("accumulation_steps", 1) != 1 or runtime.get("precision") != "bf16"
            or runtime.get("bf16_policy") != "bf16_storage" or compressed.get("enabled") is not True
            or compressed.get("bits") != 2 or compressed.get("group_size") != 1024):
        raise ValueError("Require the original VAE224 custom-optimizer BF16-storage two-bit/group1024 receipt")
    if original.get("model_factory") not in (None, "mbconv_vae:MBConvVAE_V2", "examples.vae.mbconv_vae:MBConvVAE_V2"):
        raise ValueError("The target launcher requires the original MBConvVAE_V2 model factory")
    receipt["stage"] = "CUDA_environment_preflight"
    run([sys.executable, "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'H200 CUDA/BF16 required; no CPU substitution'; assert 'H200' in torch.cuda.get_device_name(0), 'Original H200 target required'"],
        setup / "environment.log", environment)
    receipt["stage"] = "install_candidate"
    installs.mkdir()
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", candidate["install_root"], candidate["wheel"]["path"]],
        setup / "candidate-install.log", environment)
    sys.path.insert(0, str(root))
    from examples.vae.benchmark_optimization import verify_arm
    from examples.vae.benchmark_packed_offload import POLICY_DEFAULTS, ARMS, load_manifest

    identity = verify_arm(candidate, setup / "experiment-manifest.json")
    base = {**POLICY_DEFAULTS, "host_pinned_bytes": min(args.host_pinned_mib) * 2**20,
        "device_staging_bytes": args.device_staging_mib * 2**20, "d2h_inflight_bytes": args.d2h_inflight_mib * 2**20,
        "host_pageable_bytes": args.host_pageable_mib * 2**20, "copy_chunk_bytes": args.copy_chunk_mib * 2**20,
        "peak_budget_bytes": None if args.peak_budget_mib is None else args.peak_budget_mib * 2**20,
        "admission": args.admission}
    if args.policy_version == 2:
        base["version"] = 2
    policies = {arm: {**base, "enabled": arm != "resident",
        "policy": "demand" if arm == "demand_offload" else "memory_first_overlap"} for arm in ARMS}
    experiment = setup / "experiment-manifest.json"
    experiment.write_text(json.dumps({"schema_version": 1, "experiment_profile": "packed_offload_v1",
        "candidate": candidate, "policies": policies}, indent=2) + "\n")
    load_manifest(experiment)
    receipt.update(experiment_manifest_sha256=sha(experiment.read_bytes()), policies=policies)
    native_controls = {}
    if args.native_options_json is not None:
        from examples.vae.native_execution import _json_input

        native_controls, source = _json_input(args.native_options_json)
        frozen_options = setup / "requested-native-options.json"
        with frozen_options.open("xb") as stream:
            stream.write(args.native_options_json.read_bytes())
        if sha(frozen_options.read_bytes()) != source["sha256"]:
            raise ValueError("Native options changed while freezing the launcher input")
        args.native_options_json = frozen_options
        receipt["native_options_input"] = {**source, "frozen_path": str(frozen_options)}
    receipt["stage"] = "all_arm_configuration_preflight"
    configuration_command = [sys.executable, "-B", "examples/vae/benchmark_packed_offload.py", "--from-result", str(previous),
        "--experiment-manifest", str(experiment), "--device", "cuda:0", "--batch-size", "128", "--warmup", "5", "--updates", "10",
        "--trials", str(args.trials), "--start-free-tolerance-mib", str(args.start_free_tolerance_mib),
        "--host-headroom-mib", str(args.host_headroom_mib),
        "--arms", args.arms,
        "--configuration-preflight-only", "--output", str(setup / "configuration-preflight.json")]
    for name in ("project_root", "train_dir"):
        if getattr(args, name) is not None:
            configuration_command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
    if args.native_options_json is not None:
        configuration_command += ["--native-options-json", str(args.native_options_json.resolve())]
    run(configuration_command, setup / "configuration-preflight.log", environment)
    # Use the resolved, hash-verified preflight settings, including controls
    # inherited from the original mode/options envelope rather than only the
    # optional override file. No model or CUDA allocation is performed here.
    configuration_receipt = json.loads((setup / "configuration-preflight.json").read_text())
    resolved_convolution = []
    for row in configuration_receipt.get("configuration_preflight", {}).get("workers", ()):
        reference = row.get("result")
        if not isinstance(reference, dict):
            raise ValueError("Successful configuration preflight lacks a raw worker receipt")
        path = (setup / reference["path"]).resolve()
        if not path.is_relative_to(setup) or not path.is_file():
            raise ValueError("Configuration worker receipt is outside its immutable artifact directory")
        data = path.read_bytes()
        if len(data) != reference["bytes"] or sha(data) != reference["sha256"]:
            raise ValueError("Configuration worker receipt checksum changed")
        options = json.loads(data)["settings"]["runtime"]["compile"]["options"]
        resolved_convolution.append({key: options.get(key, default) for key, default in
            (("native_conv_workspace_reuse", "off"), ("native_conv_dx_batch_size", 1))})
    if resolved_convolution:
        if any(value != resolved_convolution[0] for value in resolved_convolution):
            raise ValueError("Offload arms resolved different native convolution controls")
        native_controls = {**native_controls, **resolved_convolution[0]}
    convolution_original_environment = {"DEEPOPT_TEST_LARGE_CUDA_CONV_REUSE": "1"}
    if native_controls.get("native_conv_workspace_reuse", "off") == "tiled_dx":
        bound = native_controls.get("native_conv_dx_batch_size", 1)
        if type(bound) is not int or not 1 <= bound < 128:
            raise ValueError("Original B128 tiled-dX qualification requires the actual selected bound in [1,127]; no clamping")
        convolution_original_environment["DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE"] = str(bound)
        receipt["requested_convolution_dx_batch_size"] = bound
    receipt["stage"] = "focused_CUDA_contracts"
    candidate_environment = {**environment, "PYTHONPATH": os.pathsep.join((candidate["install_root"], str(root / "benchmarks"), str(root))),
        "DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD": "1",
        "DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD_PANELS": "1",
        "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY": json.dumps(identity, sort_keys=True),
        "DEEPOPT_TEST_CUDA_CONV_ARTIFACT_DIR": str(setup / "convolution-contract-artifacts")}
    modules = ["tests.compiler.test_compiled_packed_offload.CompiledPackedOffloadCudaTests",
        "tests.compiler.test_packed_offload_cuda.PackedOffloadCudaTests",
        "tests.compiler.test_packed_offload_cuda.PackedOffloadOriginalShapeCudaTests",
        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelCudaTests",
        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelOriginalShapeCudaTests"]
    receipt["focused_cuda_modules"] = modules
    convolution_requested = native_controls.get("native_conv_workspace_reuse", "off") == "tiled_dx"
    receipt["optional_convolution_contracts_requested"] = convolution_requested
    if convolution_requested:
        # Validate the original shape before the slower general suite. A tiny
        # passing operator cannot establish this device/backend capacity.
        receipt["stage"] = "focused_CUDA_convolution-reuse-original"
        run([sys.executable, "-B", "benchmarks/compiler/qualification_tests.py", "--modules",
            "tests.compiler.test_native_convolution_reuse.NativeConvolutionReuseOriginalShapeCudaTests",
            "--device", "cuda:0", "--require-cuda", "--require-triton", "--fail-on-skip",
            "--output", str(setup / "convolution-reuse-original-CUDA-tests.json")],
            setup / "convolution-reuse-original-CUDA-tests.log",
            {**candidate_environment, **convolution_original_environment})
        receipt["qualified_convolution_dx_batch_size"] = bound
    receipt["stage"] = "focused_CUDA_contracts"
    run([sys.executable, "-B", "benchmarks/compiler/qualification_tests.py", "--modules", *modules,
        "--device", "cuda:0", "--require-cuda", "--require-triton", "--fail-on-skip",
        "--output", str(setup / "focused-CUDA-tests.json")], setup / "focused-CUDA-tests.log", candidate_environment)
    # Physical arena allocator domains retain charged capacities. Separate
    # processes keep original-size and tiny budget contracts independent.
    extra_contracts = [
        ("arena", "tests.memory.test_pinned_offload_arena.PinnedArenaCudaTests", {}),
        ("arena-original", "tests.memory.test_pinned_offload_arena.PinnedArenaOriginalShapeCudaTests",
            {"DEEPOPT_TEST_LARGE_CUDA_PINNED_ARENA": "1"}),
        ("convolution-reuse", "tests.compiler.test_native_convolution_reuse.NativeConvolutionReuseCudaTests", {}),
        ("model-offload", "tests.compiler.test_model_offload_plan.ModelOffloadCudaTests", {})]
    if not convolution_requested:
        extra_contracts = [row for row in extra_contracts if not row[0].startswith("convolution-reuse")]
    for label, selector, extra_environment in extra_contracts:
        receipt["stage"] = "focused_CUDA_" + label
        run([sys.executable, "-B", "benchmarks/compiler/qualification_tests.py", "--modules", selector,
            "--device", "cuda:0", "--require-cuda", "--require-triton", "--fail-on-skip",
            "--output", str(setup / (label + "-CUDA-tests.json"))], setup / (label + "-CUDA-tests.log"),
            {**candidate_environment, **extra_environment})
    receipt["stage"] = "same_candidate_resident_demand_predictive"
    entrypoint = ("examples/vae/benchmark_packed_offload_frontier.py" if args.max_host_pinned_mib is not None
                  else "examples/vae/benchmark_packed_offload.py")
    receipt["capacity_mode"] = "frontier_with_explicit_ceiling" if args.max_host_pinned_mib is not None else "single_capacity_no_maximum_claim"
    command = [sys.executable, "-B", entrypoint, "--from-result", str(previous),
        "--experiment-manifest", str(experiment), "--device", "cuda:0", "--batch-size", "128", "--warmup", "5", "--updates", "10",
        "--trials", str(args.trials), "--start-free-tolerance-mib", str(args.start_free_tolerance_mib),
        "--host-headroom-mib", str(args.host_headroom_mib), "--output", str(output)]
    command += ["--arms", args.arms]
    if args.max_host_pinned_mib is not None:
        command += ["--max-host-pinned-mib", str(args.max_host_pinned_mib),
                    "--host-pinned-mib", *(str(value) for value in args.host_pinned_mib)]
        if args.inventory_json is not None:
            command += ["--inventory-json", str(args.inventory_json.resolve())]
    if args.inventory_output is not None:
        command += ["--inventory-output", str(args.inventory_output.resolve())]
    if args.native_options_json is not None:
        command += ["--native-options-json", str(args.native_options_json.resolve())]
    if args.cold_diagnostics:
        command += ["--cold-diagnostics"]
    for name in ("project_root", "train_dir"):
        if getattr(args, name) is not None:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
    if args.profile_after_measurement:
        command += ["--profile-after-measurement", "--profile-updates", str(args.profile_updates)]
        for name in ("profile_memory", "profile_record_shapes"):
            if getattr(args, name):
                command += ["--" + name.replace("_", "-")]
    run(command, setup / "experiment.log", environment)
    receipt.update(status="completed", stage="complete")
    code = 0
except Exception as error:
    receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
    if isinstance(error, subprocess.CalledProcessError):
        code = error.returncode if error.returncode > 0 else 128 - error.returncode
    if not output.exists():
        output.write_text(json.dumps({"status": "failed", "execution_completed": False,
            "scope": receipt["scope"], "failure_stage": receipt["stage"], "error": receipt["error"],
            "traceback": receipt["traceback"], "performance_policy_promoted": False}, indent=2) + "\n")
    print(receipt["traceback"], file=sys.stderr)
finally:
    receipt["exit_code"] = code
    save()
    extensions = {".json", ".log", ".py", ".txt", ".ttir", ".ttgir", ".llir", ".ptx", ".sass"}
    paths = [output, *[path for path in sorted([*setup.rglob("*"), *artifacts.rglob("*")])
        if path.is_file() and not path.is_symlink() and path.suffix in extensions]]
    inventory = {str(path.relative_to(output.parent)): {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
        for path in paths if path.is_file()}
    inventory_path = setup / "archive-manifest.json"
    inventory_path.write_text(json.dumps({"files": inventory, "tensor_files_saved": 0, "install_trees_archived": False}, indent=2) + "\n")
    paths.append(inventory_path)
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent))
    print(f"Attach this packed-save offload archive: {archive.resolve()}")
raise SystemExit(code)
PY
