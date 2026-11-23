#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/run_h200_vae_optimization.sh /path/to/original_result.json [--trials 3] [--passes fusion-both|norm-activation|swiglu (v3), conv-both|conv-order|conv-geometry (v2), both|norm-cat|donation (legacy)] [--fusion-stage forward|forward_backward] [profile options]" >&2
  exit 2
fi
for argument in "${@:2}"; do
  case "$argument" in
    --output|--output=*|--experiment-manifest|--experiment-manifest=*|--arm-manifest|--arm-manifest=*|--expected-version|--expected-version=*|--native-options-json|--native-options-json=*|--device|--device=*|--batch-size|--batch-size=*|--warmup|--warmup=*|--updates|--updates=*|--optimizer|--optimizer=*|--synthetic)
      echo "This launcher fixes and archives the original CUDA B128 experiment; use the direct Python supervisor for overrides" >&2
      exit 2
      ;;
  esac
done
previous=$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve(strict=True))' "$1")
shift
bundle_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$bundle_root"
python - "$bundle_root" "$previous" "$@" <<'PY'
"""VAE_OPTIMIZATION_LAUNCHER: local wheel installs and immutable run evidence."""
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
parser.add_argument("--trials", type=int, default=1)
parser.add_argument("--passes", choices=("fusion-both", "norm-activation", "swiglu", "conv-both", "conv-order", "conv-geometry", "both", "norm-cat", "donation"))
parser.add_argument("--fusion-stage", choices=("activation_pack", "forward", "forward_backward"))
parser.add_argument("--project-root", type=Path)
parser.add_argument("--train-dir", type=Path)
parser.add_argument("--profile-after-measurement", action="store_true")
parser.add_argument("--profile-updates", type=int, default=1)
parser.add_argument("--profile-memory", action="store_true")
parser.add_argument("--profile-record-shapes", action="store_true")
parser.add_argument("--start-free-tolerance-mib", type=int, default=256)
args = parser.parse_args(sys.argv[3:])
if args.trials < 1 or not 1 <= args.profile_updates <= 3 or args.start_free_tolerance_mib < 0:
    parser.error("trials must be positive, profile-updates must be 1-3 and start-free-tolerance-mib must be nonnegative")
if not args.profile_after_measurement and (args.profile_memory or args.profile_record_shapes or args.profile_updates != 1):
    parser.error("profile options require --profile-after-measurement")

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
output = root / "work" / f"h200_vae_optimization_{stamp}.json"
artifacts = output.parent / (output.stem + "-artifacts")
setup = output.parent / (output.stem + "-launcher")
installs = output.parent / (output.stem + "-installs")
archive = output.with_suffix(".zip")
if any(path.exists() for path in (output, artifacts, setup, installs, archive)):
    raise FileExistsError("Choose a fresh run; historical evidence is immutable")
setup.mkdir(parents=True)
installs.mkdir()
receipt = {"schema_version": 1, "scope": "original_H200_B128_native_optimization_launcher",
    "status": "running", "stage": "setup", "original_receipt": str(previous),
    "input_shape": [128, 3, 224, 224], "warmup": 5, "updates": 10, "blocks": args.trials,
    "passes": args.passes, "fusion_stage": args.fusion_stage, "commands": [], "install_trees_archived": False,
    "start_free_tolerance_bytes": args.start_free_tolerance_mib * 2**20,
    "tensor_files_saved": 0, "correctness_assessed": False, "performance_policy_promoted": False}

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save_receipt():
    (setup / "launcher.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

def run(command, log, environment):
    receipt["commands"].append({"command": command, "log": str(log), "stage": receipt["stage"]})
    save_receipt()
    with log.open("x") as stream:
        with subprocess.Popen(command, cwd=root, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as process:
            for line in process.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)

def wheel_spec(path, expected_sha, expected_version, *, required_modules=(), required_defaults=None):
    path = path.resolve()
    if not path.is_relative_to(root / "packages") or sha(path.read_bytes()) != expected_sha:
        raise ValueError("Bundled wheel path/checksum differs from its pinned package")
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        if len(names) != len(set(names)) or wheel.testzip() is not None:
            raise ValueError("Wheel membership/CRC mismatch")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("Wheel contains an unsafe path")
        metadata_name, = [name for name in names if name.endswith(".dist-info/METADATA")]
        metadata = Parser().parsestr(wheel.read(metadata_name).decode())
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != expected_version:
            raise ValueError("Wheel distribution/version differs from the bundle manifest")
        package = {name.removeprefix("deepopt/"): wheel.read(name) for name in names
                   if name.startswith("deepopt/") and not name.endswith("/")}
    if "__init__.py" not in package:
        raise ValueError("Wheel has no DeepOpt package")
    if any(name not in package for name in required_modules):
        raise ValueError("Candidate wheel lacks a required experiment module")
    if required_defaults is not None:
        backend = ast.parse(package.get("compiler/backend.py", b""))
        definition = next((node for node in backend.body if isinstance(node, ast.ClassDef) and node.name == "BackendConfig"), None)
        values = {node.target.id: ast.literal_eval(node.value) for node in definition.body
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                  and node.target.id in required_defaults} if definition is not None else {}
        if any(name not in values or type(values[name]) is not type(value) or values[name] != value
               for name, value in required_defaults.items()):
            raise ValueError("Candidate wheel lacks the typed training-fusion default contract")
    digest = hashlib.sha256()
    for name in sorted(package, key=lambda value: Path(value).parts):
        digest.update(name.encode() + b"\0" + package[name] + b"\0")
    return {"wheel": {"path": str(path), "sha256": expected_sha}, "expected_installed": {
        "version": expected_version, "source_sha256": digest.hexdigest(), "files_hashed": len(package)}}

environment = dict(os.environ)
environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
environment.pop("DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY", None)
environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"
code = 1
try:
    manifest = json.loads((root / "manifest.json").read_text())
    receipt["stage"] = "experiment_profile"
    profile = manifest.get("experiment_profile")
    if profile not in (None, "native_convolution_wave2", "native_training_fusion_v3"):
        raise ValueError("Unsupported bundle experiment_profile")
    convolution_profile = profile == "native_convolution_wave2"
    fusion_profile = profile == "native_training_fusion_v3"
    allowed = (("fusion-both", "norm-activation", "swiglu") if fusion_profile else
               ("conv-both", "conv-order", "conv-geometry") if convolution_profile else ("both", "norm-cat", "donation"))
    args.passes = args.passes or allowed[0]
    if args.passes not in allowed:
        raise ValueError(f"--passes {args.passes} does not belong to this bundle experiment profile")
    if args.fusion_stage is not None and not fusion_profile:
        raise ValueError("--fusion-stage requires the native_training_fusion_v3 bundle profile")
    if fusion_profile:
        args.fusion_stage = args.fusion_stage or "forward_backward"
        if args.fusion_stage == "activation_pack" and args.passes != "swiglu":
            raise ValueError("activation_pack is only a SwiGLU single-feature ablation")
    receipt.update(experiment_profile=profile or "legacy_memory_wave1", passes=args.passes, fusion_stage=args.fusion_stage)
    fusion_defaults = {"fuse_norm_cat_body": False, "donate_fresh_backward_add": False,
        "native_conv_backward_order": "source", "cuda_conv_dx_geometry": False,
        "fuse_norm_activation_residuals": False, "norm_activation_fusion_stage": "off",
        "fuse_swiglu_residuals": False, "swiglu_fusion_stage": "off"}
    control = manifest["control"]
    receipt.update(bundle_manifest_sha256=sha((root / "manifest.json").read_bytes()),
                   original_receipt_sha256=sha(previous.read_bytes()))
    a = wheel_spec(root / control["bundle_wheel"], control["wheel_sha256"], control["version"])
    b = wheel_spec(root / "packages" / Path(manifest["wheel"]).name, manifest["wheel_sha256"], manifest["version"],
        required_modules=(("compiler/autodiff/norm_activation_residuals.py", "compiler/autodiff/swiglu_fusion.py")
            if fusion_profile else ("compiler/autodiff/convolution_order.py", "compiler/autodiff/convolution_geometry.py")
            if convolution_profile else ()), required_defaults=fusion_defaults if fusion_profile else None)
    if (a["expected_installed"]["version"] != "0.1.0+nativeautodiff.20261119.vaememory.4"
            or a["wheel"]["sha256"] != "d2b7c417237211cbf8853c47d72028f1aa3cdf9d7782de715787dc345195ab94"
            or a["expected_installed"]["source_sha256"] != "fb5a619fad46e77831fd97ba00818af870db46feb6dc39bc3b32c6392cf09775"):
        raise ValueError("A must be the immutable original .4 wheel/package")
    if (a["expected_installed"]["source_sha256"] != control["package_source_sha256"]
            or a["expected_installed"]["files_hashed"] != control["files_hashed"]):
        raise ValueError("Control package differs from its historical freeze descriptor")
    if b["expected_installed"]["version"] == a["expected_installed"]["version"]:
        raise ValueError("Candidate requires a unique version; never replace the .4 wheel")
    receipt["stage"] = "CUDA_environment_preflight"
    run([sys.executable, "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'H200 CUDA/BF16 is required; no CPU substitution'"],
        setup / "environment.log", environment)
    for label, spec in (("control", a), ("candidate", b)):
        target = installs / label
        receipt["stage"] = "install_" + label
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(target), spec["wheel"]["path"]],
            setup / (label + "-install.log"), environment)
        spec["install_root"] = str(target)
    sys.path.insert(0, str(root))
    from examples.vae.benchmark_optimization import verify_arm

    for spec in (a, b):
        verify_arm(spec, setup / "experiment-manifest.json")
    off = {"fuse_norm_cat_body": False, "donate_fresh_backward_add": False}
    if fusion_profile:
        off = dict(fusion_defaults)
        norm, swiglu = args.passes in ("fusion-both", "norm-activation"), args.passes in ("fusion-both", "swiglu")
        on = {**off, "fuse_norm_activation_residuals": norm,
            "norm_activation_fusion_stage": args.fusion_stage if norm else "off",
            "fuse_swiglu_residuals": swiglu, "swiglu_fusion_stage": args.fusion_stage if swiglu else "off"}
    elif convolution_profile:
        off.update(native_conv_backward_order="source", cuda_conv_dx_geometry=False)
        on = {**off, "native_conv_backward_order": "dw_first" if args.passes in ("conv-both", "conv-order") else "source",
              "cuda_conv_dx_geometry": args.passes in ("conv-both", "conv-geometry")}
    else:
        on = {"fuse_norm_cat_body": args.passes in ("both", "norm-cat"),
              "donate_fresh_backward_add": args.passes in ("both", "donation")}
    arms = {"A": {**a, "native_options": {}}, "B": {**b, "native_options": off}, "C": {**b, "native_options": on}}
    experiment = setup / "experiment-manifest.json"
    experiment.write_text(json.dumps({"schema_version": 1, "arms": arms,
        **({"experiment_profile": profile} if profile is not None else {})}, indent=2) + "\n")
    receipt["experiment_manifest_sha256"] = sha(experiment.read_bytes())
    receipt["stage"] = "focused_CUDA_contracts"
    candidate_environment = dict(environment)
    candidate_environment["PYTHONPATH"] = os.pathsep.join((b["install_root"], str(root / "benchmarks"), str(root)))
    for name in ("DEEPOPT_RUN_H200_NORM_CAT", "DEEPOPT_TEST_LARGE_CUDA_DONATION",
                 "DEEPOPT_RUN_H200_CONV_ORDER", "DEEPOPT_TEST_LARGE_CUDA_CONV_GEOMETRY",
                 "DEEPOPT_RUN_H200_NORM_ACTIVATION", "DEEPOPT_TEST_LARGE_CUDA_SWIGLU"):
        candidate_environment.pop(name, None)
    if fusion_profile:
        candidate_environment["DEEPOPT_RUN_H200_NORM_ACTIVATION"] = "1"
        candidate_environment["DEEPOPT_TEST_LARGE_CUDA_SWIGLU"] = "1"
        diagnostic_directory = setup / "fusion-diagnostics"
        candidate_environment["DEEPOPT_FUSION_DIAGNOSTIC_DIR"] = str(diagnostic_directory)
        receipt["fusion_diagnostics"] = {"directory": str(diagnostic_directory),
            "scope": "focused_kernel_metadata_and_compiled_code_no_tensor_payloads"}
        modules = ["tests.compiler.test_native_norm_activation_residuals.NativeNormActivationResidualCudaTests",
                   "tests.compiler.test_native_norm_activation_residuals.NativeNormActivationResidualOriginalShapeCudaTests",
                   "tests.compiler.test_native_swiglu_fusion.NativeSwiGLUFusionCudaTests",
                   "tests.compiler.test_native_swiglu_fusion.NativeSwiGLUFusionOriginalShapeCudaTests",
                   "tests.compiler.test_native_training_fusion_combined.NativeTrainingFusionCombinedCudaTests",
                   "tests.compiler.test_native_norm_fusion_layout.NativeNormFusionLayoutCudaTests"]
    elif convolution_profile:
        candidate_environment["DEEPOPT_RUN_H200_CONV_ORDER"] = "1"
        candidate_environment["DEEPOPT_TEST_LARGE_CUDA_CONV_GEOMETRY"] = "1"
        modules = ["tests.compiler.test_native_convolution_order.NativeConvolutionOrderCudaTests",
                   "tests.compiler.test_native_convolution_geometry.NativeConvolutionGeometryCudaTests",
                   "tests.compiler.test_native_convolution_geometry.NativeConvolutionGeometryOriginalShapeCudaTests",
                   "tests.compiler.test_native_convolution_memory_combined.NativeConvolutionMemoryCombinedCudaTests.test_actual_cuda_combined_native_vjp_and_retained_backward"]
    else:
        candidate_environment["DEEPOPT_RUN_H200_NORM_CAT"] = "1"
        candidate_environment["DEEPOPT_TEST_LARGE_CUDA_DONATION"] = "1"
        modules = ["tests.compiler.test_norm_cat.NormCatBodyCUDATests.test_original_b128_encoder1_groupnorm_cat_geometry",
                   "tests.compiler.test_backward_add_donation.BackwardAddDonationCudaTests",
                   "tests.compiler.test_backward_add_donation.BackwardAddDonationOriginalShapeCudaTests"]
    candidate_environment["DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY"] = json.dumps(verify_arm(b, experiment), sort_keys=True)
    receipt["focused_cuda_modules"] = modules
    run([sys.executable, "-B", "benchmarks/compiler/qualification_tests.py", "--modules", *modules,
        "--device", "cuda:0", "--require-cuda", "--require-triton", "--fail-on-skip",
        "--output", str(setup / "focused-CUDA-tests.json")], setup / "focused-CUDA-tests.log", candidate_environment)
    receipt["stage"] = "native_A_B_C_experiment"
    command = [sys.executable, "-B", "examples/vae/benchmark_optimization.py", "--from-result", str(previous),
        "--experiment-manifest", str(experiment), "--device", "cuda:0", "--batch-size", "128", "--warmup", "5",
        "--updates", "10", "--trials", str(args.trials),
        "--start-free-tolerance-mib", str(args.start_free_tolerance_mib), "--output", str(output)]
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
    save_receipt()
    extensions = {".json", ".log", ".py", ".txt", ".ttir", ".ttgir", ".llir", ".ptx", ".sass"}
    paths = [output, *[path for path in sorted([*setup.rglob("*"), *artifacts.rglob("*")])
                      if path.is_file() and not path.is_symlink() and path.suffix in extensions]]
    inventory = {str(path.relative_to(output.parent)): {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                 for path in paths if path.is_file()}
    inventory_path = setup / "archive-manifest.json"
    inventory_path.write_text(json.dumps({"files": inventory, "tensor_files_saved": 0,
        "install_trees_archived": False}, indent=2) + "\n")
    paths.append(inventory_path)
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.is_file():
                bundle.write(path, path.relative_to(output.parent))
    print(f"Attach this native optimization archive: {archive.resolve()}")
raise SystemExit(code)
PY
