"""Prepare a source-identified installed-only qualification bundle after freeze.

--dry-run is read-only and needs no Torch imports. Actual export requires both
production and benchmark freeze manifests; historical destinations are refused.
"""

import argparse
import ast
import base64
import csv
from email.parser import Parser
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SUFFIXES = {".py", ".json", ".toml", ".md", ".txt", ".sh"}
CONVOLUTION_MODULES = ("src/deepopt/compiler/autodiff/convolution_order.py",
                       "src/deepopt/compiler/autodiff/convolution_geometry.py")
CONVOLUTION_GUIDES = ("docs/execution/compiler_codegen_scheduler_memory.md",)
FUSION_MODULES = ("src/deepopt/compiler/autodiff/norm_activation_residuals.py",
                  "src/deepopt/compiler/autodiff/swiglu_fusion.py")
FUSION_CONFIG_DEFAULTS = {"fuse_norm_cat_body": False, "donate_fresh_backward_add": False,
    "native_conv_backward_order": "source", "cuda_conv_dx_geometry": False,
    "fuse_norm_activation_residuals": False, "norm_activation_fusion_stage": "off",
    "fuse_swiglu_residuals": False, "swiglu_fusion_stage": "off"}
FUSION_GUIDES = ("docs/execution/compiler_fusion.md",)
OFFLOAD_MODULES = ("src/deepopt/_packed_offload.py", "src/deepopt/compiler/offload_plan.py",
                   "src/deepopt/compiler/runtime/packed_offload.py", "src/deepopt/compiler/runtime/manager.py")
MEMORY_V2_MODULES = ("src/deepopt/compiler/model_offload_plan.py", "src/deepopt/compiler/runtime/pinned_arena.py",
                     "src/deepopt/compiler/autodiff/convolution_reuse.py")
MEMORY_V2_GUIDES = ("scripts/run_h200_vae_packed_memory.sh",)
OFFLOAD_GUIDES = ("docs/execution/packed_activation_offload_ddp.md",)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def experiment_profile(root):
    """Old source fixtures/releases keep their original experiment profile."""
    fusion = tuple(root / name for name in FUSION_MODULES)
    if any(path.is_symlink() for path in fusion):
        raise ValueError("Training fusion modules must be regular source files")
    if all(path.is_file() for path in fusion):
        backend = root / "src/deepopt/compiler/backend.py"
        if backend.is_symlink():
            raise ValueError("Training fusion config must be a regular source file")
        if backend.is_file():
            tree = ast.parse(backend.read_bytes())
            definition = next((node for node in tree.body if isinstance(node, ast.ClassDef)
                               and node.name == "BackendConfig"), None)
            values = {node.target.id: ast.literal_eval(node.value) for node in definition.body
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                and node.target.id in FUSION_CONFIG_DEFAULTS} if definition is not None else {}
            if all(name in values and type(values[name]) is type(value) and values[name] == value
                   for name, value in FUSION_CONFIG_DEFAULTS.items()):
                return "native_training_fusion_v3"
    paths = tuple(root / name for name in CONVOLUTION_MODULES)
    if any(path.is_symlink() for path in paths):
        raise ValueError("Convolution experiment modules must be regular source files")
    return "native_convolution_wave2" if all(path.is_file() for path in paths) else None


def packed_offload_profile(root):
    """Independent transport profile; never expand the compiler-option whitelist."""
    paths = tuple(root / name for name in OFFLOAD_MODULES)
    if any(path.is_symlink() for path in paths):
        raise ValueError("Packed offload modules must be regular source files")
    if not all(path.is_file() for path in paths):
        return None
    policy = ast.parse(paths[0].read_bytes())
    definition = next((node for node in policy.body if isinstance(node, ast.ClassDef)
                       and node.name == "PackedActivationOffloadPolicy"), None)
    values = {node.target.id: ast.literal_eval(node.value) for node in definition.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        and node.target.id == "enabled"} if definition is not None else {}
    if values.get("enabled") is not False:
        raise ValueError("Packed offload export requires disabled production defaults")
    return "packed_offload_v1"


def sources(root):
    """Keep the source layout; never include src/, results, tensors or caches."""
    selected = {}
    for directory in ("benchmarks", "tests", "examples/vae", "docs/execution"):
        for path in sorted((root / directory).rglob("*")):
            relative = path.relative_to(root)
            if any(part == "__pycache__" or part.startswith((".", "inductor_vae_v2_b16_224_"))
                   for part in relative.parts):
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlink is not exportable: {path}")
            if path.suffix in SOURCE_SUFFIXES and path.is_file():
                if not stat.S_ISREG(path.stat().st_mode):
                    raise ValueError(f"Source is not a regular file: {path}")
                selected[relative.as_posix()] = path
    for name in ("LICENSE", "AGENTS.md", "compiler_references/inductor/README.md"):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required source or guide missing: {name}")
        selected[name] = path
    if (root / "examples/vae/diagnostics.py").is_file():
        launcher = root / "scripts/run_h200_vae_diagnostics.sh"
        if not launcher.is_file() or launcher.is_symlink():
            raise ValueError("VAE diagnostic export requires its regular-file H200 launcher")
        selected[launcher.relative_to(root).as_posix()] = launcher
    if (root / "examples/vae/diagnostic_repeatability.py").is_file():
        launcher = root / "scripts/run_h200_vae_repeatability.sh"
        if not launcher.is_file() or launcher.is_symlink():
            raise ValueError("VAE repeatability export requires its H200 launcher")
        selected[launcher.relative_to(root).as_posix()] = launcher
    if (root / "examples/vae/native_execution.py").is_file():
        launcher = root / "scripts/run_h200_vae_native128.sh"
        if not launcher.is_file() or launcher.is_symlink():
            raise ValueError("VAE native execution export requires its H200 launcher")
        selected[launcher.relative_to(root).as_posix()] = launcher
    if (root / "examples/vae/benchmark_execution.py").is_file():
        launcher = root / "scripts/run_h200_vae_benchmark.sh"
        if not launcher.is_file() or launcher.is_symlink():
            raise ValueError("VAE eager/native benchmark export requires its H200 launcher")
        selected[launcher.relative_to(root).as_posix()] = launcher
    if (root / "examples/vae/benchmark_optimization.py").is_file():
        launcher = root / "scripts/run_h200_vae_optimization.sh"
        if not launcher.is_file() or launcher.is_symlink():
            raise ValueError("VAE optimization benchmark export requires its H200 launcher")
        selected[launcher.relative_to(root).as_posix()] = launcher
    offload_benchmark = root / "examples/vae/benchmark_packed_offload.py"
    if offload_benchmark.is_file() or packed_offload_profile(root) is not None:
        if not offload_benchmark.is_file() or offload_benchmark.is_symlink():
            raise ValueError("Packed offload runtime export requires its benchmark supervisor")
        if packed_offload_profile(root) is None:
            raise ValueError("Packed offload benchmark requires all runtime/planner modules")
        for name in ("scripts/run_h200_vae_packed_offload.sh", *OFFLOAD_GUIDES):
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError("Packed offload export requires its regular-file launcher/guide: " + name)
            selected[name] = path
        memory_v2 = tuple(root / name for name in MEMORY_V2_MODULES)
        if any(path.exists() for path in memory_v2):
            if any(not path.is_file() or path.is_symlink() for path in memory_v2):
                raise ValueError("Packed memory v2 requires all arena/model-selection/workspace modules")
            for name in MEMORY_V2_GUIDES:
                path = root / name
                if not path.is_file() or path.is_symlink():
                    raise ValueError("Packed memory v2 requires matching guide: " + name)
                selected[name] = path
    profile = experiment_profile(root)
    if profile is not None:
        for name in CONVOLUTION_GUIDES:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Convolution experiment requires its regular-file guide: {name}")
            selected[name] = path
    if profile == "native_training_fusion_v3":
        for name in FUSION_GUIDES:
            path = root / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Training fusion experiment requires its regular-file guide: {name}")
            selected[name] = path
    return selected


def _verified_wheel(wheel, expected_sha, production, root, *, current_sources):
    """Verify a wheel against its own freeze, with an explicit source scope."""
    if not wheel.is_file() or wheel.is_symlink() or not production.is_file() or production.is_symlink():
        raise ValueError("Wheel and production freeze must be regular files")
    if digest(wheel.read_bytes()) != expected_sha:
        raise ValueError("Wheel SHA256 differs from the declared release")
    frozen = json.loads(production.read_text())
    hashes = frozen["source_sha256"]
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Production freeze has no source hashes")
    if any(not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
           or PurePosixPath(name).as_posix() != name for name in hashes):
        raise ValueError("Production freeze source paths must be canonical relative paths")
    expected_package = {name.removeprefix("src/"): value for name, value in hashes.items()
                        if name.startswith("src/deepopt/")}
    if not expected_package:
        raise ValueError("Production freeze has no DeepOpt package hashes")
    if current_sources:
        if any((root / name).is_symlink() or digest((root / name).read_bytes()) != value
               for name, value in hashes.items()):
            raise ValueError("Production sources changed since freeze; do not export")
        current_package = {}
        for path in sorted((root / "src/deepopt").rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("Production package source contains a symlink")
            current_package[path.relative_to(root / "src").as_posix()] = digest(path.read_bytes())
        if current_package != expected_package:
            raise ValueError("Production source package membership changed since freeze; do not export")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or archive.testzip() is not None:
            raise ValueError("Wheel membership/CRC is invalid")
        files = {info.filename for info in archive.infolist() if not info.is_dir()}
        for info in archive.infolist():
            name = info.filename.rstrip("/")
            modes = (0, stat.S_IFDIR) if info.is_dir() else (0, stat.S_IFREG)
            if (not name or PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
                    or PurePosixPath(name).as_posix() != name
                    or stat.S_IFMT(info.external_attr >> 16) not in modes):
                raise ValueError("Wheel contains a noncanonical or nonregular member")
        records = [name for name in files if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise ValueError("Wheel must contain exactly one RECORD")
        record = records[0]
        metadata_name = record.removesuffix("RECORD") + "METADATA"
        if [name for name in files if name.endswith(".dist-info/METADATA")] != [metadata_name]:
            raise ValueError("Wheel must contain its distribution's single METADATA")
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        if metadata["Name"] != "deep-optimization" or metadata["Version"] != frozen["version"]:
            raise ValueError("Wheel distribution identity differs from production freeze")
        recorded = set()
        for name, hashed, size in csv.reader(io.StringIO(archive.read(record).decode())):
            if name in recorded or name not in files:
                raise ValueError("Wheel RECORD membership is invalid")
            recorded.add(name)
            if not hashed:
                if name != record or size:
                    raise ValueError("Unexpected unhashed wheel member")
                continue
            algorithm, expected = hashed.split("=", 1)
            data = archive.read(name)
            actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if algorithm != "sha256" or actual != expected or len(data) != int(size):
                raise ValueError(f"Wheel RECORD mismatch: {name}")
        if recorded != files:
            raise ValueError("Wheel RECORD does not cover every file")
        package = {name: archive.read(name) for name in files if name.startswith("deepopt/")}
        if {name: digest(data) for name, data in package.items()} != expected_package:
            raise ValueError("Wheel package bytes/membership differ from production freeze")
        identity = hashlib.sha256()
        # Match installed_identity's sorted Path.rglob order, including a file
        # prefix such as a.py alongside a/value.txt.
        for name, data in sorted(package.items(), key=lambda pair: PurePosixPath(pair[0]).parts):
            identity.update(name.removeprefix("deepopt/").encode() + b"\0" + data + b"\0")
    return frozen, {"package_source_sha256": identity.hexdigest(), "files_hashed": len(package)}


def verify_wheel(wheel, expected_sha, production, root):
    """Legacy candidate API: its freeze must still match current sources."""
    return _verified_wheel(wheel, expected_sha, production, root, current_sources=True)[0]


def _control(root, wheel, expected_sha, production):
    values = (wheel, expected_sha, production)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise ValueError("Control wheel, SHA256 and production freeze must be supplied together")
    wheel, production = Path(wheel), Path(production)
    release, identity = _verified_wheel(wheel, expected_sha, production, root, current_sources=False)
    return {"version": release["version"], "wheel": str(wheel), "wheel_sha256": expected_sha,
            "production_freeze": str(production), "production_freeze_sha256": digest(production.read_bytes()),
            **identity, "verification": "own_production_freeze_not_current_sources",
            "bundle_wheel": "packages/" + wheel.name,
            "bundle_production_freeze": "release/control-production-freeze.json"}


def plan(root, output, wheel, expected_sha, production, benchmark=None, *,
         control_wheel=None, control_wheel_sha256=None, control_production_freeze=None):
    archive = output.with_name(output.name + ".zip")
    if any(output.resolve().is_relative_to(root / name) for name in
           ("benchmarks", "tests", "examples", "docs", "compiler_references")):
        raise ValueError("Export destination must be outside benchmark/test/guide/reference source trees")
    if output.exists() or archive.exists():
        raise FileExistsError("Use a fresh directory and ZIP; historical artifacts are immutable")
    selected = sources(root)
    hashes = {name: digest(path.read_bytes()) for name, path in selected.items()}
    release = verify_wheel(wheel, expected_sha, production, root)
    control = _control(root, control_wheel, control_wheel_sha256, control_production_freeze)
    if control is not None and Path(control["wheel"]).name == wheel.name:
        raise ValueError("Candidate and control wheel filenames must differ")
    if benchmark is not None:
        frozen = json.loads(benchmark.read_text())["source_sha256"]
        missing = [name for name, value in hashes.items() if frozen.get(name) != value]
        if missing:
            raise ValueError(f"Benchmark freeze missing/changed sources: {missing[:8]}")
    return {"version": release["version"], "output": str(output), "archive": str(archive),
            "wheel": str(wheel), "wheel_sha256": expected_sha, "source_sha256": hashes,
            "production_freeze_sha256": digest(production.read_bytes()),
            **({"benchmark_freeze_sha256": digest(benchmark.read_bytes())} if benchmark is not None else {}),
            **({"control": control} if control is not None else {}),
            **({"experiment_profile": experiment_profile(root)} if experiment_profile(root) is not None else {}),
            **({"packed_offload_profile": packed_offload_profile(root)} if packed_offload_profile(root) is not None else {}),
            **({"packed_memory_profile": "packed_memory_v2"}
               if all((root / name).is_file() for name in MEMORY_V2_MODULES) else {}),
            "benchmark_freeze_verified": benchmark is not None, "installed_only": True,
            "cuda_qualified": False, "tensor_files_included": 0}


def export(root, description, production, benchmark, *, control_wheel=None,
           control_wheel_sha256=None, control_production_freeze=None):
    if benchmark is None or not description["benchmark_freeze_verified"]:
        raise ValueError("Actual export requires --benchmark-freeze after aggregate benchmark/docs freeze")
    output, archive = Path(description["output"]), Path(description["archive"])
    # Revalidate immediately before creating a new destination.
    if not any(value is not None for value in (control_wheel, control_wheel_sha256, control_production_freeze)):
        control = description.get("control", {})
        control_wheel, control_wheel_sha256, control_production_freeze = (
            control.get("wheel"), control.get("wheel_sha256"), control.get("production_freeze"))
    fresh = plan(root, output, Path(description["wheel"]), description["wheel_sha256"], production, benchmark,
                 control_wheel=control_wheel, control_wheel_sha256=control_wheel_sha256,
                 control_production_freeze=control_production_freeze)
    if fresh != description:
        raise ValueError("Export sources changed after planning")
    inputs = [(Path(description["wheel"]), "packages/" + Path(description["wheel"]).name,
               description["wheel_sha256"]),
              (production, "release/production-freeze.json", description["production_freeze_sha256"]),
              (benchmark, "release/benchmark-freeze.json", description["benchmark_freeze_sha256"])]
    if "control" in description:
        control = description["control"]
        inputs.extend(((Path(control["wheel"]), control["bundle_wheel"], control["wheel_sha256"]),
                       (Path(control["production_freeze"]), control["bundle_production_freeze"],
                        control["production_freeze_sha256"])))
    payloads = []
    for path, name, expected in inputs:
        data = path.read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Release input changed before export: {path}")
        payloads.append((name, data, expected))
    verify_wheel(Path(description["wheel"]), description["wheel_sha256"], production, root)
    offload_readme = None
    if description.get("packed_offload_profile") == "packed_offload_v1":
        memory_v2 = description.get("packed_memory_profile") == "packed_memory_v2"
        offload_readme = (
            "# H200 same-candidate packed-save CPU offload\n\n"
            "The launcher installs the pinned candidate wheel into a fresh target directory, requires real CUDA contracts, "
            "then compares resident, demand and predictive workers without changing the training recipe. "
            "Host/staging budgets must be supplied explicitly.\n\n"
            "```sh\nbash scripts/run_h200_vae_packed_offload.sh original_result.json \\\n"
            "  --host-pinned-mib PINNED_MIB --device-staging-mib STAGING_MIB --d2h-inflight-mib D2H_MIB \\\n"
            "  --trials " + ("6" if memory_v2 else "3") + "\n```\n\n"
            "Choose positive integer budgets for the target host and the supported consumer operand or named panel floor; "
            "the placeholders above are deliberately not defaults. Use the separate profile-worker flags only for diagnostics. "
            "Attach the resulting metadata/code ZIP. No tensor files or installation trees are archived. "
            "CPU checks do not establish CUDA correctness, overlap, memory reduction or performance.\n\n"
            "See [the runbook](docs/execution/packed_activation_offload_ddp.md).\n")
        if memory_v2:
            offload_readme += (
                "\nThis candidate defaults to version2 physical pinned arenas and model-wide selection. "
                "Add `--max-host-pinned-mib CEILING_MIB` to measure intermediate and all-eligible capacity points. "
                "The ceiling is explicit and must fit the job's host RAM after headroom. "
                "A resource-limited point is not reported as the design maximum.\n\n"
                "See [the capacity, workspace, selection, arena and combined runbook]"
                "(docs/execution/compiler_codegen_scheduler_memory.md) for complete commands and their comparison scopes.\n")
    if {name: digest(path.read_bytes()) for name, path in sources(root).items()} != description["source_sha256"]:
        raise ValueError("Export sources changed immediately before writing")
    output.mkdir(parents=True, exist_ok=False)
    if offload_readme is not None:
        (output / "RUN_PACKED_ACTIVATION_OFFLOAD.md").write_text(offload_readme)
    if description.get("packed_memory_profile") == "packed_memory_v2":
        (output / "RUN_PACKED_MEMORY_V2.md").write_text(
            "# Packed activation memory qualification\n\n"
            "Pass a local run receipt and explicit memory budgets to "
            "scripts/run_h200_vae_packed_memory.sh. See "
            "[compiler storage and capacity](docs/execution/compiler_codegen_scheduler_memory.md).\n")
    for name, expected in description["source_sha256"].items():
        data = (root / name).read_bytes()
        if digest(data) != expected:
            raise ValueError(f"Source changed during export: {name}; incomplete output retained")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    for name, data, _ in payloads:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (output / "RUN_BF16_POLICIES.md").write_text(
        "# AMP and experimental BF16 storage qualification\n\n"
        "Run from this extracted directory. See [the run guide](docs/execution/bf16_storage_policy.md).\n"
        "For original torchvision classifiers, use [the vision run guide](docs/execution/backend_qualification.md).\n"
        "The bundled wheel is an experimental installed candidate; CUDA correctness, quality, performance and capacity require target evidence.\n")
    if (output / "examples/vae/diagnostics.py").is_file():
        (output / "RUN_VAE_DIAGNOSTICS.md").write_text(
            "# H200 VAE numerical diagnostics\n\n"
            "Activate the same CUDA environment and run from this extracted directory:\n\n"
            "```bash\n"
            "bash scripts/run_h200_vae_repeatability.sh \\\n"
            "  recorded-run.json\n"
            "```\n\n"
            "The script installs the matching bundled wheel and runs focused backward repeatability controls.\n"
            "It creates a metadata/code ZIP to attach; the full original matrix remains available via the diagnostic launcher.\n"
            "See [the diagnostic guide](docs/execution/backend_qualification.md) for paths, controls and limits.\n"
            "Numerical tolerances are unchanged. These untimed diagnostics do not qualify H200 capacity or performance.\n")
    if (output / "examples/vae/native_execution.py").is_file():
        (output / "RUN_VAE_NATIVE128.md").write_text(
            "# Native VAE V2 batch-128 execution\n\n"
            "From this extracted directory with your CUDA environment active:\n\n"
            "```bash\n"
            "bash scripts/run_h200_vae_native128.sh \\\n"
            "  recorded-run.json\n"
            "```\n\n"
            "The script installs the matching wheel, then runs only the original compiled VAE V2 at physical input [128,3,224,224].\n"
            "It performs one warmup and two subsequent training updates using the recorded BF16/compression/optimizer recipe.\n"
            "There is one model and no eager-reference or compression-error comparison.\n"
            "Completion requires actual forward, backward, optimizer updates, finite results and native AutoDiff ownership.\n"
            "The resulting metadata/code ZIP records GPU memory and compiler evidence; it does not claim numerical or training-quality qualification.\n")
    if (output / "examples/vae/benchmark_execution.py").is_file():
        (output / "RUN_VAE_BENCHMARK.md").write_text(
            "# DeepOpt eager versus compiled VAE V2 benchmark\n\n"
            "From this extracted directory with your CUDA environment active:\n\n"
            "```bash\n"
            "bash scripts/run_h200_vae_benchmark.sh \\\n"
            "  recorded-run.json\n"
            "```\n\n"
            "The script installs the matching bundled candidate wheel, then runs DeepOpt eager and native compiled modes in separate processes.\n"
            "Each mode uses physical input [128,3,224,224], five warmup and ten measured optimizer updates with the recorded recipe.\n"
            "The first real input batch stays on the GPU and is repeated. Loading, H2D, finite-state checks and evidence collection are outside timing.\n"
            "The report compares runtime, measured allocator peaks, and startup/whole-run allocator peaks; reductions may be negative.\n"
            "There is no numerical or activation-compression error comparison. Native ownership and finite-state execution guards remain.\n"
            "Append --trials 3 for three fresh pairs, alternating route order. Attach the generated metadata/code ZIP.\n"
            "See [the benchmark guide](examples/vae/README.md) for scopes and interpretation.\n")
    if (output / "examples/vae/benchmark_optimization.py").is_file() and "control" in description:
        convolution_profile = description.get("experiment_profile") == "native_convolution_wave2"
        fusion_profile = description.get("experiment_profile") == "native_training_fusion_v3"
        pass_option = (" --passes fusion-both --fusion-stage forward_backward" if fusion_profile else
                       " --passes conv-both" if convolution_profile else "")
        pass_description = (
            "The new bundle defaults to `--passes fusion-both --fusion-stage forward_backward`: normalization/SiLU and SwiGLU training fusions.\n"
            "Use `--passes norm-activation` or `--passes swiglu` for a declared feature ablation, and `--fusion-stage forward` for forward-only work.\n"
            "Every legacy memory/convolution option stays at its default/off/source value in B and C. "
            "Each requested forward/backward stage must appear as its exact selected executor; incomplete stage evidence invalidates ratios.\n"
            "Focused actual CUDA checks include small and original block geometries and fail on skipped tests.\n"
            "See [the training fusion run guide](docs/execution/compiler_fusion.md). "
            "The separate [common-recipe parity guide](compiler_references/inductor/README.md) defines the Inductor comparison.\n"
            if fusion_profile else
            "The new bundle defaults to `--passes conv-both`: guarded weight-first adjoints and the CUDA geometry-only input descriptor.\n"
            "Use `--passes conv-order` or `--passes conv-geometry` for a declared single-change experiment. "
            "The previous norm-cat and donation controls remain disabled in both B and C.\n"
            "Focused CUDA checks include small and original B128 convolution geometry and fail on skipped tests.\n"
            "See [the convolution memory run guide](docs/execution/compiler_codegen_scheduler_memory.md).\n"
            if convolution_profile else "")
        (output / "RUN_VAE_OPTIMIZATION.md").write_text(
            "# H200 native VAE optimization A/B/C experiment\n\n"
            "From this extracted directory with your CUDA environment active:\n\n"
            "```bash\n"
            "bash scripts/run_h200_vae_optimization.sh \\\n"
            "  recorded-run.json" + pass_option + "\n"
            "```\n\n"
            "The default is one A/B/C triage block: three fresh native workers.\n"
            "A uses the pinned historical control wheel. B and C use the same uniquely versioned candidate wheel, "
            "with the new passes disabled in B and the requested passes enabled in C.\n"
            + pass_description +
            "Both wheels and their separate production freezes are included; the historical control is verified "
            "against its own freeze rather than the changed candidate sources.\n"
            "The physical input is [128,3,224,224], using the original recorded model, optimizer, BF16 storage "
            "and group1024 compression recipe. Each worker repeats the resident first real data batch.\n"
            "For three A/B/C blocks, nine fresh workers, append `--trials 3`:\n\n"
            "```bash\n"
            "bash scripts/run_h200_vae_optimization.sh \\\n"
            "  recorded-run.json --trials 3" + pass_option + "\n"
            "```\n\n"
            "Keep profiler collection outside measured updates and report startup and measured allocator peaks separately.\n"
            "This experiment records execution, compiler ownership and descriptive runtime/memory. "
            "It does not run an activation-compression error matrix or assess training quality.\n"
            "A local CPU check is diagnostic only; CUDA optimization and performance remain pending until target evidence is collected.\n"
            "See [the qualification guide](docs/execution/backend_qualification.md).\n")
    if (output / "examples/vae/benchmark_parity.py").is_file():
        (output / "RUN_VAE_PARITY.md").write_text(
            "# Common-recipe VAE compiler comparison\n\n"
            "Follow [the parity run guide](compiler_references/inductor/README.md) to pin the installed candidate and run the matched routes.\n"
            "This separate comparison starts with dense native Torch BF16 AMP at a common executed batch, with the original model and optimizer.\n"
            "Inductor arithmetic presets, compiled scope, state/input/source hashes and target identity remain explicit.\n"
            "Record runtime and memory on the matching target.\n")
    if {name: digest(path.read_bytes()) for name, path in sources(root).items()} != description["source_sha256"]:
        raise ValueError("Source membership/bytes changed during export; incomplete output retained")
    verify_wheel(Path(description["wheel"]), description["wheel_sha256"], production, root)
    if (digest(production.read_bytes()) != description["production_freeze_sha256"]
            or digest(benchmark.read_bytes()) != description["benchmark_freeze_sha256"]):
        raise ValueError("Release freeze changed during export; incomplete output retained")
    if "control" in description and _control(root, control_wheel, control_wheel_sha256,
                                              control_production_freeze) != description["control"]:
        raise ValueError("Control release changed during export; incomplete output retained")
    for name, _, expected in payloads:
        if digest((output / name).read_bytes()) != expected:
            raise ValueError(f"Copied release input bytes changed: {name}")
    files = {path.relative_to(output).as_posix(): digest(path.read_bytes()) for path in output.rglob("*") if path.is_file()}
    manifest = {**description, "files": files, "historical_reference_artifacts_included": False,
                "historical_control_release_included": "control" in description,
                "reference_note": "Inductor runs and indexes are local development evidence."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with archive.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, output.name + "/" + path.relative_to(output).as_posix())
    return {"directory": str(output), "archive": str(archive), "archive_sha256": digest(archive.read_bytes()),
            "version": description["version"], "files": len(files) + 1, "cuda_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-wheel-sha256", required=True)
    parser.add_argument("--production-freeze", type=Path, required=True)
    parser.add_argument("--benchmark-freeze", type=Path)
    parser.add_argument("--control-wheel", type=Path)
    parser.add_argument("--control-wheel-sha256", "--expected-control-wheel-sha256", dest="control_wheel_sha256")
    parser.add_argument("--control-production-freeze", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run and args.benchmark_freeze is None:
        parser.error("Export is held until --benchmark-freeze is supplied; use --dry-run for read-only planning")
    try:
        description = plan(ROOT, args.output.absolute(), args.wheel.absolute(), args.expected_wheel_sha256,
                           args.production_freeze.absolute(), args.benchmark_freeze.absolute() if args.benchmark_freeze else None,
                           control_wheel=args.control_wheel.absolute() if args.control_wheel else None,
                           control_wheel_sha256=args.control_wheel_sha256,
                           control_production_freeze=args.control_production_freeze.absolute() if args.control_production_freeze else None)
        result = description if args.dry_run else export(ROOT, description, args.production_freeze, args.benchmark_freeze)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
