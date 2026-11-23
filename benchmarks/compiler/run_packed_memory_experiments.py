"""Prepare and qualify same-candidate H200 memory ablations from a frozen bundle.

This entry point is standard-library only. It does not replace the environment
package, alter a training loop, or use historical timings as a denominator.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zipfile


ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))


OPTIONS = dict(fuse_norm_cat_body=False, donate_fresh_backward_add=False,
    native_conv_backward_order="source", cuda_conv_dx_geometry=False,
    fuse_norm_activation_residuals=False, norm_activation_fusion_stage="off",
    fuse_swiglu_residuals=False, swiglu_fusion_stage="off",
    native_conv_workspace_reuse="off", native_conv_dx_batch_size=1,
    packed_offload_selection_mode="peak")
PROFILES = {"workspace": ("native_convolution_workspace_v2", ("D0", "D1")),
    "combined": ("native_memory_combined_v2", ("R0", "BEST")),
    "selection": ("native_offload_selection_v2", ("S0", "S1")),
    "arena": ("native_pinned_arena_v2", ("H0", "H1"))}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def candidate_spec(root, install):
    from benchmarks.compiler.export_qualification_bundle import _verified_wheel

    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if (manifest.get("packed_offload_profile") != "packed_offload_v1"
            or manifest.get("packed_memory_profile") != "packed_memory_v2"):
        raise ValueError("A frozen packed-memory bundle is required")
    wheel = root / "packages" / Path(manifest["wheel"]).name
    production = root / "release/production-freeze.json"
    if (wheel.is_symlink() or production.is_symlink()
            or sha(production.read_bytes()) != manifest["production_freeze_sha256"]):
        raise ValueError("Candidate production freeze changed")
    release, identity = _verified_wheel(wheel, manifest["wheel_sha256"], production, root, current_sources=False)
    if release["version"] != manifest["version"]:
        raise ValueError("Bundle version differs from candidate freeze")
    for name, expected in manifest["source_sha256"].items():
        path = root / name
        if (Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink()
                or not path.is_file() or sha(path.read_bytes()) != expected):
            raise ValueError("Bundle helper/source changed: " + name)
    return {"wheel": {"path": str(wheel), "sha256": manifest["wheel_sha256"]},
        "install_root": str(Path(install).resolve()),
        "expected_installed": {"version": release["version"],
            "source_sha256": identity["package_source_sha256"], "files_hashed": identity["files_hashed"]}}


def validate_original(previous):
    original = json.loads(Path(previous).read_bytes())
    raw = original["settings"]
    runtime, training, application = raw["runtime"], raw["training"], raw["application"]
    compression = runtime.get("activation_compression", {})
    if (application["model"] != dict(base_channels=32, max_channels=512, latent_channels=4, blocks_per_stage=2)
            or application["data"]["image_size"] != 224 or training.get("optimizer") != "custom"
            or training.get("accumulation_steps", 1) != 1 or runtime.get("precision") != "bf16"
            or runtime.get("bf16_policy") != "bf16_storage" or compression.get("enabled") is not True
            or compression.get("bits") != 2 or compression.get("group_size") != 1024
            or original.get("model_factory") not in (None, "mbconv_vae:MBConvVAE_V2", "examples.vae.mbconv_vae:MBConvVAE_V2")):
        raise ValueError("Require the original VAE224 custom-optimizer BF16-storage two-bit/group1024 receipt")


def experiment(candidate, args):
    from examples.vae.benchmark_packed_offload import POLICY_DEFAULTS

    profile, names = PROFILES[args.experiment]
    options = {**OPTIONS, "native_conv_dx_batch_size": args.dx_batch_size,
        "packed_offload_selection_mode": args.selection_mode}
    base = {**POLICY_DEFAULTS, "version": 2, "enabled": True, "policy": "demand",
        "host_pinned_bytes": args.host_pinned_mib * 2**20,
        "device_staging_bytes": args.device_staging_mib * 2**20,
        "d2h_inflight_bytes": args.d2h_inflight_mib * 2**20,
        "admission": "resident_fallback"}
    arms = {name: {**deepcopy(candidate), "native_options": dict(options)} for name in names}
    result = {"schema_version": 1, "experiment_profile": profile, "arms": arms}
    tiling = "off" if args.no_tiling else "tiled_dx"
    if args.experiment in ("workspace", "combined"):
        if args.experiment == "workspace" and args.no_tiling:
            raise ValueError("Workspace ablation requires tiled_dx in D1")
        arms[names[1]]["native_options"]["native_conv_workspace_reuse"] = tiling
    else:
        for arm in arms.values():
            arm["native_options"]["native_conv_workspace_reuse"] = tiling
    if args.experiment == "selection":
        if args.selection_mode != "peak":
            raise ValueError("Selection ablation fixes S0 regional_order and S1 peak")
        arms[names[0]]["native_options"]["packed_offload_selection_mode"] = "regional_order"
    if args.experiment in ("combined", "arena"):
        result["policies"] = {name: dict(base) for name in names}
        if args.experiment == "combined":
            result["policies"]["R0"]["enabled"] = False
        else:
            if args.selection_mode != "peak":
                raise ValueError("Legacy arena H0 supports only its historical peak selector")
            result["policies"]["H0"]["version"] = 1
    else:
        result["packed_activation_offload"] = base
    return result


def contract_groups(*, include_convolution=True):
    # Physical allocator domains cannot shrink cached arenas between tiny and
    # original-size budget tests. Each tuple is a fresh process.
    groups = [("transport", [
        "tests.compiler.test_compiled_packed_offload.CompiledPackedOffloadCudaTests",
        "tests.compiler.test_packed_offload_cuda.PackedOffloadCudaTests",
        "tests.compiler.test_packed_offload_cuda.PackedOffloadOriginalShapeCudaTests",
        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelCudaTests",
        "tests.compiler.test_packed_offload_panels_cuda.PackedOffloadPanelOriginalShapeCudaTests"],
        {"DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD": "1", "DEEPOPT_TEST_LARGE_CUDA_PACKED_OFFLOAD_PANELS": "1"}),
        ("arena", ["tests.memory.test_pinned_offload_arena.PinnedArenaCudaTests"], {}),
        ("arena-original", ["tests.memory.test_pinned_offload_arena.PinnedArenaOriginalShapeCudaTests"],
            {"DEEPOPT_TEST_LARGE_CUDA_PINNED_ARENA": "1"}),
        ("convolution", ["tests.compiler.test_native_convolution_reuse.NativeConvolutionReuseCudaTests"], {}),
        ("convolution-original", ["tests.compiler.test_native_convolution_reuse.NativeConvolutionReuseOriginalShapeCudaTests"],
            {"DEEPOPT_TEST_LARGE_CUDA_CONV_REUSE": "1"}),
        ("model-scope", ["tests.compiler.test_model_offload_plan.ModelOffloadCudaTests"], {})]
    if include_convolution:
        original = next(row for row in groups if row[0] == "convolution-original")
        return [original, *[row for row in groups if row is not original]]
    return [row for row in groups if not row[0].startswith("convolution")]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("from_result", type=Path)
    result.add_argument("--experiment", choices=PROFILES, default="combined")
    for name in ("host-pinned-mib", "device-staging-mib", "d2h-inflight-mib"):
        result.add_argument("--" + name, type=int, required=True)
    result.add_argument("--dx-batch-size", type=int, default=16)
    result.add_argument("--selection-mode", choices=("peak", "all_eligible"), default="peak")
    result.add_argument("--no-tiling", action="store_true")
    result.add_argument("--trials", type=int, default=6)
    result.add_argument("--start-free-tolerance-mib", type=int, default=256)
    result.add_argument("--host-headroom-mib", type=int, default=8192)
    result.add_argument("--project-root", type=Path)
    result.add_argument("--train-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if (any(getattr(args, key) < 1 for key in ("host_pinned_mib", "device_staging_mib", "d2h_inflight_mib", "dx_batch_size", "trials"))
            or args.dx_batch_size >= 128 or args.start_free_tolerance_mib < 0 or args.host_headroom_mib < 0):
        raise ValueError("Budgets, tile bound and trial count must be positive; headroom/tolerance nonnegative")
    previous = args.from_result.resolve(strict=True)
    validate_original(previous)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = ROOT / "work" / f"h200_vae_packed_memory_{args.experiment}_{stamp}.json"
    setup, artifacts = (output.parent / (output.stem + suffix) for suffix in ("-launcher", "-artifacts"))
    install = output.parent / (output.stem + "-installs") / "candidate"
    candidate = candidate_spec(ROOT, install)
    manifest = experiment(candidate, args)
    if args.dry_run:
        print(json.dumps({"experiment": manifest, "qualification_processes": contract_groups(include_convolution=not args.no_tiling),
            "installed_only": True, "CUDA_execution_proven": False, "performance_policy_promoted": False}, indent=2))
        return 0
    if any(path.exists() for path in (output, setup, artifacts, install.parent, output.with_suffix(".zip"))):
        raise FileExistsError("Use a fresh immutable run destination")
    setup.mkdir(parents=True)
    receipt = dict(status="running", stage="install", commands=[], tensor_files_written=0,
        install_trees_archived=False, original_receipt_sha256=sha(previous.read_bytes()), candidate=candidate)
    environment = dict(os.environ)
    environment.pop("TORCHDYNAMO_REPRO_AFTER", None)
    environment.pop("DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY", None)
    environment["TORCHDYNAMO_REPRO_LEVEL"] = "0"

    def save():
        (setup / "launcher.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")

    def run(command, label, env=None):
        receipt.update(stage=label)
        receipt["commands"].append(command)
        save()
        with (setup / (label + ".log")).open("x") as log:
            with subprocess.Popen(command, cwd=ROOT, env=environment if env is None else env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace") as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)

    code = 1
    try:
        install.parent.mkdir()
        run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-compile", "--target", str(install),
            candidate["wheel"]["path"]], "candidate-install")
        from examples.vae.benchmark_optimization import verify_arm, load_manifest

        experiment_path = setup / "experiment-manifest.json"
        experiment_path.write_text(json.dumps(manifest, indent=2) + "\n")
        load_manifest(experiment_path)
        identity = verify_arm(candidate, experiment_path)
        command = [sys.executable, "-B", "examples/vae/benchmark_optimization.py", "--from-result", str(previous),
            "--experiment-manifest", str(experiment_path), "--device", "cuda:0", "--batch-size", "128",
            "--warmup", "5", "--updates", "10", "--trials", str(args.trials),
            "--start-free-tolerance-mib", str(args.start_free_tolerance_mib),
            "--host-headroom-mib", str(args.host_headroom_mib)]
        for name in ("project_root", "train_dir"):
            if getattr(args, name) is not None:
                command += ["--" + name.replace("_", "-"), str(getattr(args, name).resolve())]
        run([*command, "--configuration-preflight-only", "--output", str(setup / "configuration-preflight.json")],
            "configuration-preflight")
        run([sys.executable, "-c", "import torch; assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(); assert 'H200' in torch.cuda.get_device_name(0), 'H200 CUDA/BF16 required'"],
            "CUDA-environment")
        installed_env = {**environment, "PYTHONPATH": os.pathsep.join((str(install), str(ROOT / "benchmarks"), str(ROOT))),
            "DEEPOPT_QUALIFICATION_INSTALLED_IDENTITY": json.dumps(identity, sort_keys=True),
            "DEEPOPT_TEST_CUDA_CONV_DX_BATCH_SIZE": str(1 if args.no_tiling else args.dx_batch_size),
            "DEEPOPT_TEST_CUDA_CONV_ARTIFACT_DIR": str(setup / "convolution-contract-artifacts")}
        for label, modules, extra in contract_groups(include_convolution=not args.no_tiling):
            run([sys.executable, "-B", "benchmarks/compiler/qualification_tests.py", "--modules", *modules,
                "--device", "cuda:0", "--require-cuda", "--require-triton", "--fail-on-skip",
                "--output", str(setup / (label + "-CUDA-tests.json"))], "contract-" + label, {**installed_env, **extra})
        run([*command, "--output", str(output)], "paired-benchmark")
        receipt.update(status="completed", stage="complete")
        code = 0
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(receipt["traceback"], file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            code = error.returncode if error.returncode > 0 else 128 - error.returncode
        if not output.exists():
            output.write_text(json.dumps({**receipt, "execution_completed": False, "performance_policy_promoted": False}, indent=2) + "\n")
    finally:
        receipt["exit_code"] = code
        save()
        suffixes = {".json", ".log", ".py", ".txt", ".ttir", ".ttgir", ".llir", ".ptx", ".sass"}
        paths = [output, *[path for base in (setup, artifacts) for path in sorted(base.rglob("*"))
            if path.is_file() and not path.is_symlink() and path.suffix in suffixes]]
        archive_manifest = setup / "archive-manifest.json"
        archive_manifest.write_text(json.dumps({"tensor_files_written": 0, "install_trees_archived": False,
            "files": {str(path.relative_to(output.parent)): {"bytes": path.stat().st_size, "sha256": sha(path.read_bytes())}
                for path in paths if path.is_file()}}, indent=2) + "\n")
        with output.with_suffix(".zip").open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in [*paths, archive_manifest]:
                if path.is_file():
                    archive.write(path, path.relative_to(output.parent))
        print(f"Attach this packed-memory ablation archive: {output.with_suffix('.zip')}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
