"""Export the current benchmark sources without selecting a DeepOpt source tree."""

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import zipfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

from analysis.benchmark_do_ops import DEFAULT_TOLERANCES
from analysis.catalog import workload_list
from analysis.matrix import DEFAULT_SETTINGS, EXTENDED_GRID, merge, variants


def export_kernel_backends(output, wheel=None, library_root=None):
    """Export a coherent small bundle, without unrelated model sweeps."""
    archive = output.with_name(output.name + ".zip")
    if output.exists() or archive.exists():
        raise ValueError("Focused benchmark exports require a new directory and ZIP path")
    runner = ROOT / "benchmarks/benchmark_kernel_backends.py"
    spec = importlib.util.spec_from_file_location("_kernel_benchmark_export", runner)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {"benchmark_kernel_backends.py", "kernel_backends.md", "_gpu_lease.py",
             "analysis/__init__.py", "analysis/_gpu_lease.py", "analysis/scheduler.py",
             "analysis/matrix.py", "analysis/reporting.py",
             "normalization/benchmark_tilelang_compression_quality.py"}
    files.update(item[0] for item in module.WORKLOADS.values())
    library_root = Path(library_root) if library_root is not None else ROOT / "src/deepopt"
    if not (library_root / "__init__.py").is_file():
        raise ValueError("--library-root must contain the selected DeepOpt package sources")
    expected = {"deepopt/" + path.relative_to(library_root).as_posix(): sha256(path.read_bytes()).hexdigest()
                for path in sorted(library_root.rglob("*.py"))}
    if wheel is not None:
        if not wheel.is_file() or not wheel.name.endswith("-py3-none-any.whl"):
            raise ValueError("--wheel must name a portable DeepOpt Python wheel")
        with zipfile.ZipFile(wheel) as package:
            actual = {name: sha256(package.read(name)).hexdigest() for name in package.namelist()
                      if name.startswith("deepopt/") and name.endswith(".py")}
        if actual != expected:
            raise ValueError("The supplied wheel does not match the selected DeepOpt Python sources; rebuild it first")
    output.mkdir(parents=True)
    for name in sorted(files):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / "benchmarks" / name, destination)
    install = "Install the matching updated DeepOpt build in this environment before running.\n"
    if wheel is not None:
        name = "packages/" + wheel.name
        (output / "packages").mkdir()
        shutil.copyfile(wheel, output / name)
        files.add(name)
        install = f"python -m pip install --no-deps --force-reinstall {name}\n"
    instructions = (
        "Run from this extracted directory using the intended CUDA environment.\n\n"
        + install + "\nThe wheel installation does not replace Torch, Triton or TileLang.\n"
        "The runner checks matching worker options and selected DeepOpt helper APIs before GPU admission.\n"
        "Do not copy only the coordinator: keep all worker scripts and analysis helpers together.\n\n"
        "python benchmark_kernel_backends.py --device cuda:0 --preset representative "
        "--workloads linear layer_norm pooling --backends triton tilelang "
        "--time-budget 1 --trials 1 --timeout 180 --output work/kernel-backend-fixed\n\n"
        "Use a fresh output directory after replacing old scripts/builds; old failed runs have no timings to reuse.\n"
        "Add --resume to the identical command to reuse completed workers from this new run.\n"
        "Budget targets exclude cold compilation/setup; they are not hard per-launch deadlines.\n"
        "This bundle has no H100 performance qualification; GPU execution still needs to be checked.\n"
        "\nOpt-in focused numerical qualification (legacy commands above remain unchanged):\n"
        "python benchmark_kernel_backends.py --qualify-tilelang --device cuda:0 --preset smoke "
        "--workloads layer_norm pooling --backends triton tilelang --inputs bf16 --outputs bf16 "
        "--compression off --max-worker-runs 4 --time-budget 1 --trials 1 --timeout 180 "
        "--output work/tilelang-qualified-smoke\n"
        "Then change only --preset to representative and use a fresh output directory.\n"
        "Qualification is independent per role, not complete operator parity or convergence evidence.\n"
        "\nCPU-only compression quality screen (no GPU allocation or backend performance claims):\n"
        "python normalization/benchmark_tilelang_compression_quality.py --bits 2 4 8 "
        "--group-sizes 128 256 512 --distributions normal large_offset outliers "
        "--output work/tilelang-compression-quality.json\n"
    )
    (output / "RUN_KERNEL_BACKENDS.txt").write_text(instructions, encoding="utf-8")
    files.add("RUN_KERNEL_BACKENDS.txt")
    manifest = {"schema_version": 1, "kind": "kernel_backend_benchmarks",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "deepopt_python_sources_sha256": sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
                "matching_wheel_included": wheel is not None,
                "files": {name: sha256((output / name).read_bytes()).hexdigest() for name in sorted(files)}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    files.add("manifest.json")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(files):
            bundle.write(output / name, output.name + "/" + name)
    return {"directory": str(output), "archive": str(archive), "files": len(files),
            "matching_wheel_included": wheel is not None}


def main(argv=None):
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--output", type=Path, default=ROOT / "work/full-comparison-benchmarks",
                           help="Destination for separate scripts, configs, guide and a sibling ZIP")
    arguments.add_argument("--kernel-backends-only", action="store_true", help="Export only the matched custom-backend benchmark suite")
    arguments.add_argument("--wheel", type=Path, help="Include a matching portable DeepOpt wheel (focused mode only)")
    arguments.add_argument("--library-root", type=Path,
                           help="Verify a focused wheel against this staged deepopt package instead of checkout sources")
    options = arguments.parse_args(argv)
    OUTPUT = options.output.resolve()
    benchmark_root = (ROOT / "benchmarks").resolve()
    if OUTPUT == benchmark_root or benchmark_root in OUTPUT.parents:
        arguments.error("Choose an export directory outside benchmarks so canonical sources cannot be overwritten")
    if options.kernel_backends_only:
        try:
            result = export_kernel_backends(OUTPUT, options.wheel, options.library_root)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            arguments.error(str(error))
        print(json.dumps(result, indent=2))
        return 0
    if options.wheel is not None:
        arguments.error("--wheel requires --kernel-backends-only")
    if options.library_root is not None:
        arguments.error("--library-root requires --kernel-backends-only")
    EXPORTED_FILES = set()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    library_digest = sha256()
    library_root = ROOT / "src/deepopt"
    for source in sorted(library_root.rglob("*.py")):
        library_digest.update(source.relative_to(library_root).as_posix().encode() + source.read_bytes())
    library_source_sha256 = library_digest.hexdigest()


    def write_json(name, data):
        (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        EXPORTED_FILES.add(name)


    header = '''#!/usr/bin/env python3
"""Portable full DeepOpt / Torch BF16 benchmark.

This file contains all benchmark helper sources below. It materializes only
those helpers into a temporary directory while running, including isolated
worker processes. It imports the installed DeepOpt package; no library source
tree is added to Python's search path. See RUN_BENCHMARKS.txt for comparisons.
"""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


_BENCHMARK_SOURCES = {
'''
    sources = []
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        # Keep the embedded code readable, including its original docstrings.
        literal = "'''" + source.read_text(encoding="utf-8").replace("\\", "\\\\").replace("'''", "\\'\\'\\'") + "'''"
        sources.append(f"    {source.name!r}: {literal},\n")

    footer = '''}


def main(argv=None):
    with TemporaryDirectory(prefix="deepopt-benchmark-") as temporary:
        package = Path(temporary) / "analysis"
        package.mkdir()
        for name, source in _BENCHMARK_SOURCES.items():
            (package / name).write_text(source, encoding="utf-8")
        sys.path.insert(0, temporary)
        try:
            from analysis.benchmark_do_ops import main as run_benchmark
            return run_benchmark(argv)
        finally:
            sys.path.remove(temporary)


if __name__ == "__main__":
    raise SystemExit(main())
'''
    (OUTPUT / "benchmark_do_ops.py").write_text(header + "".join(sources) + footer, encoding="utf-8")
    EXPORTED_FILES.add("benchmark_do_ops.py")
    for name,needs_directory,guarded in (("benchmark_suite",True,False),("benchmark_select_candidates",False,False),("gpu_preflight",False,True),
                                        ("benchmark_cifar10_epoch",True,True)):
        ending=footer.replace("from analysis.benchmark_do_ops import main as run_benchmark",
                              f"from analysis.{name} import main as run_benchmark")
        if needs_directory:
            ending=ending.replace("            return run_benchmark(argv)",
                "            from functools import partial\n"
                "            run_benchmark = partial(run_benchmark, benchmark_dir=Path(__file__).resolve().parent)\n"
                "            return run_benchmark(argv)")
        if guarded:
            ending=ending.replace("            return run_benchmark(argv)",
                "            from analysis.scheduler import guarded_main\n"
                "            return guarded_main(run_benchmark, list(argv) if argv is not None else sys.argv[1:], __file__)")
        (OUTPUT/(name+".py")).write_text(header+"".join(sources)+ending,encoding="utf-8")
        EXPORTED_FILES.add(name+".py")

    for relative in ("benchmarks/activation_compression/benchmark_activation_unpack.py",
                     "benchmarks/benchmark_kernel_backends.py",
                     "benchmarks/kernel_backends.md",
                     "benchmarks/fp8/benchmark_fp8_eager.py",
                     "benchmarks/activation/benchmark_activations.py",
                     "benchmarks/pooling/benchmark_pooling.py",
                     "benchmarks/convolution/benchmark_convolution.py",
                     "benchmarks/compiler/benchmark_compiler.py",
                     "benchmarks/activation_compression/benchmark_activation_pack.py",
                     "benchmarks/activation_compression/benchmark_attention_pack.py",
                     "benchmarks/activation_compression/benchmark_spatial_layout.py",
                     "benchmarks/activation_compression/benchmark_convolution_codec.py",
                     "benchmarks/activation_compression/toy_quest_saved_linear.py",
                     "benchmarks/_gpu_lease.py",
                     "benchmarks/analysis/benchmark_allocator_compare.py",
                     "benchmarks/analysis/diagnose_training_stability.py",
                     "benchmarks/analysis/benchmark_attention_alignment.py",
                     "benchmarks/analysis/attention_comparison.py",
                     "benchmarks/analysis/benchmark_vit_attention_stability.py",
                     "benchmarks/analysis/benchmark_attention_preflight.py",
                     "benchmarks/analysis/run_attention_recovery.sh",
                     "benchmarks/analysis/run_optimization_recovery.sh",
                     "benchmarks/analysis/run_runtime_speed.sh",
                     "benchmarks/analysis/stability_oracle.py",
                     "benchmarks/analysis/run_stability_replay.sh",
                     "docs/activation_compression/flash_attention_2.md",
                     "benchmarks/normalization/benchmark_normalization.py",
                     "benchmarks/fp8/benchmark_fp8_accumulation.py",
                     "benchmarks/fp8/benchmark_fp8_linear_compare.py"):
        shutil.copyfile(ROOT / relative, OUTPUT / Path(relative).name)
        EXPORTED_FILES.add(Path(relative).name)

    shutil.copyfile(ROOT / "tests/cuda/test_flash_attention_backend.py", OUTPUT / "test_flash_attention_backend_cuda.py")
    EXPORTED_FILES.add("test_flash_attention_backend_cuda.py")
    shutil.copyfile(ROOT / "tests/cuda/test_runtime_speed.py", OUTPUT / "test_runtime_speed_cuda.py")
    EXPORTED_FILES.add("test_runtime_speed_cuda.py")
    for name in ("test_convolution_tuned.py", "test_speed_activation_candidates.py", "test_fp8_preparation_tuning.py"):
        shutil.copyfile(ROOT / "tests/cuda" / name, OUTPUT / name)
        EXPORTED_FILES.add(name)
    # Candidate gates share hardware/operand helpers. Ship a minimal test
    # package so standalone exports never depend on the repository's tests.
    for relative in ("tests/__init__.py", "tests/cuda/__init__.py", "tests/cuda/fp8_support.py"):
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if relative.endswith("fp8_support.py"):
            shutil.copyfile(ROOT / relative, target)
        else:
            target.write_text("")
        EXPORTED_FILES.add(relative)

    # Keep independently editable originals as well as the single-file launcher.
    source_directory = OUTPUT / "analysis"
    source_directory.mkdir(exist_ok=True)
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        shutil.copyfile(source, source_directory / source.name)
        EXPORTED_FILES.add("analysis/" + source.name)

    for config in sorted((ROOT / "benchmarks/analysis/configs").glob("*.json")):
        data = json.loads(config.read_text())
        for name, policy in DEFAULT_TOLERANCES.items():
            data.setdefault("tolerances", {}).setdefault(name, deepcopy(policy))
        write_json(config.name, data)

    all_operators = {"settings": deepcopy(DEFAULT_SETTINGS), "workloads": workload_list(),
                     "tolerances": deepcopy(DEFAULT_TOLERANCES)}
    all_operators["settings"]["runtime"]["timing_fill_uninitialized_memory"] = False
    write_json("all_operators.json", all_operators)

    all_settings = deepcopy(all_operators)
    all_settings["grid"] = deepcopy(EXTENDED_GRID)
    all_settings["grid"]["compression.group_size"] = [32, 128, 512, 1024, 2048, 32768]
    write_json("all_settings.json", all_settings)

    vit = json.loads((OUTPUT / "h100_vit_recovery.json").read_text())
    vit_ablation = deepcopy(vit)
    vit_ablation["settings"]["runtime"]["execution"] = "eager"
    vit_ablation["grid"] = {"runtime.compression_families": [None, [], ["linear"], ["attention"],
        ["normalization"], ["activation"], ["convolution"], ["pooling"],
        ["linear", "attention", "normalization", "activation", "convolution", "pooling"]]}
    write_json("h100_vit_ablation.json", vit_ablation)

    # Cross only the primary regression factors; the broad all-settings grid is OAT.
    vit_interactions = deepcopy(vit)
    vit_interactions["grid"] = {"runtime.execution": ["eager", "graph"],
        "fp8.backward_backend": ["inherit", "native_fast", "native_precise"],
        "compression.group_size": [512, 1024], "compression.bits": [2, 4],
        "compression.norm_xhat_recenter": [False, True]}
    write_json("h100_vit_interactions.json", vit_interactions)

    settings_variants = len(list(variants(all_settings["settings"], all_settings["grid"], "one-at-a-time")))
    vit_interaction_tasks = 5 * len(list(variants(
        merge(DEFAULT_SETTINGS, vit_interactions["settings"]),
        vit_interactions["grid"], "cartesian")))
    operators = sum(not item["op"].endswith("codec") for item in all_operators["workloads"])
    codecs = len(all_operators["workloads"]) - operators
    counts = {"operator_workloads": operators, "codec_workloads": codecs,
              "all_operator_tasks_with_gradient_probes": (operators * 5 + codecs) * 5,
              "settings_axes": len(all_settings["grid"]), "settings_variants": settings_variants,
              "all_settings_tasks": settings_variants * (operators * 5 + codecs),
              "vit_cases": 5, "vit_family_ablation_tasks": 45, "vit_crossed_interaction_tasks": vit_interaction_tasks}
    counts["random_init_torchvision_model_cases"] = 10
    counts["v5_model_cases"] = 15
    v6_counts={}
    for name,sweep in (("hopper_v6_models","cartesian"),("hopper_v6_preflight","cartesian"),
                       ("hopper_v6_vit_hot","one-at-a-time"),("hopper_v6_cnn_hot","one-at-a-time"),
                       ("hopper_v6_resnet_bridge","cartesian"),("hopper_v6_precision_2","cartesian"),
                       ("hopper_v6_precision_4","cartesian"),("hopper_v6_precision_8","cartesian")):
        configuration=json.loads((OUTPUT/(name+".json")).read_text())
        settings=merge(DEFAULT_SETTINGS,configuration["settings"])
        v6_counts[name]=len(list(variants(settings,configuration.get("grid",{}),sweep)))*len(configuration["workloads"])*5
    counts["v6_requested_model_tasks_by_matrix"]=v6_counts
    counts["v6_requested_model_tasks_excluding_preflight"]=sum(value for name,value in v6_counts.items() if name!="hopper_v6_preflight")
    speed_counts = {}
    for path in sorted((ROOT / "benchmarks/analysis/configs").glob("h100_runtime_speed_*.json")):
        configuration = json.loads(path.read_text())
        sweep = "cartesian" if path.stem.endswith(("comparison", "native", "profiles", "convolution")) else "one-at-a-time"
        settings = merge(DEFAULT_SETTINGS, configuration["settings"])
        speed_counts[path.stem] = len(list(variants(settings, configuration.get("grid", {}), sweep))) * len(configuration["workloads"]) * 5
    counts["runtime_speed_requested_tasks_by_matrix"] = speed_counts

    runner = r'''#!/usr/bin/env bash
# Run on one otherwise idle NVIDIA GPU with the UPDATED DeepOpt build installed.
set -uo pipefail
benchmark_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
benchmark_python="${BENCHMARK_PYTHON:-python}"
# V6 owns stage scheduling and physical-device admission in one coordinator.
for benchmark_argument in "$@"; do
  if [[ "$benchmark_argument" == --recovery-v6 ]]; then
    exec "$benchmark_python" "$benchmark_dir/benchmark_suite.py" "$@"
  fi
done
benchmark_output=full-comparison-results
benchmark_output_seen=0
benchmark_resume=1
benchmark_focused=0
benchmark_recovery_v3=0
benchmark_recovery_v4=0
benchmark_recovery_v5=0
for benchmark_argument in "$@"; do
  case "$benchmark_argument" in
    --resume) benchmark_resume=1 ;;
    --no-resume) benchmark_resume=0 ;;
    --focused) benchmark_focused=1 ;;
    --recovery-v3) benchmark_recovery_v3=1 ;;
    --recovery-v4) benchmark_recovery_v4=1 ;;
    --recovery-v5) benchmark_recovery_v5=1 ;;
    --help) echo "Usage: bash run_full_comparisons.sh [output-directory] [--resume|--no-resume] [--focused|--recovery-v3|--recovery-v4|--recovery-v5|--recovery-v6]"; echo "V6 scheduler options: python benchmark_suite.py --help"; exit 0 ;;
    --*) echo "Unknown option: $benchmark_argument" >&2; exit 2 ;;
    *) if [[ "$benchmark_output_seen" == 1 ]]; then echo "Only one output directory is accepted" >&2; exit 2; fi
       benchmark_output="$benchmark_argument"; benchmark_output_seen=1 ;;
  esac
done
benchmark_profile_steps="${BENCHMARK_PROFILE_STEPS:-3}"
benchmark_status=0
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

run_benchmark() {
  if [[ "$benchmark_resume" == 1 ]]; then
    "$@" --resume || benchmark_status=1
  else
    "$@" || benchmark_status=1
  fi
}

if [[ "$benchmark_recovery_v5" == 1 ]]; then
  benchmark_trace_flags=()
  if [[ -n "${BENCHMARK_TRACE_DIRECTORY:-}" ]]; then benchmark_trace_flags=(--trace-directory "$BENCHMARK_TRACE_DIRECTORY"); fi
  for benchmark_config in hopper_v5_models hopper_v5_model_ablation hopper_v5_integrated hopper_v5_norm_boundary hopper_v5_gelu_boundary hopper_v5_codec_pipeline hopper_v5_precision_2 hopper_v5_precision_4 hopper_v5_precision_8; do
    benchmark_sweep=cartesian
    if [[ "$benchmark_config" == hopper_v5_model_ablation ]]; then benchmark_sweep=one-at-a-time; fi
    benchmark_plan_flags=()
    if [[ "$benchmark_config" != hopper_v5_gelu_boundary && "$benchmark_config" != hopper_v5_codec_pipeline ]]; then
      benchmark_plan_flags=(--require-fp8-plan)
    fi
    run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
      --config "$benchmark_dir/$benchmark_config.json" --sweep "$benchmark_sweep" ${benchmark_plan_flags[@]+"${benchmark_plan_flags[@]}"} \
      --deterministic --fill-uninitialized-memory --warmup 10 --iterations 30 --repeats 5 \
      --max-numerical-elements "${BENCHMARK_MAX_NUMERICAL_ELEMENTS:-16384}" \
      --profile-steps "$benchmark_profile_steps" --profile-replay-steps 1 --profile-detail trace --compress-traces \
      --trace-workloads vit_b512 resnet18_b128 efficientnet_b0_b128 norm_fp32_fp8_edge vit_gelu_fp8_edge \
      --trace-cases torch_bf16 fp8_compression ${benchmark_trace_flags[@]+"${benchmark_trace_flags[@]}"} \
      --output "$benchmark_output/$benchmark_config"
  done
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_allocator_compare.py" \
    --config "$benchmark_dir/hopper_v5_allocator.json" --variable-config "$benchmark_dir/hopper_v4_allocator_variable.json" \
    --profile-steps 0 --max-numerical-elements "${BENCHMARK_MAX_NUMERICAL_ELEMENTS:-16384}" --output "$benchmark_output/allocator"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_attention_pack.py" \
    --bits 4 8 --group-sizes 128 512 1024 --num-warps 1 2 4 --indexing wide bounded \
    --output "$benchmark_output/attention-pack.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_spatial_layout.py" \
    --channels 1 3 4 16 --output "$benchmark_output/spatial-layout.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_pack.py" \
    --bits 2 --group-sizes 512 1024 --variants batched serial --groups-per-program 2 4 \
    --num-warps 1 2 4 auto --output "$benchmark_output/pack.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_fp8_accumulation.py" \
    --bits 2 4 8 --group-sizes 128 512 1024 --output "$benchmark_output/accumulation.json"
  exit "$benchmark_status"
fi

if [[ "$benchmark_recovery_v4" == 1 ]]; then
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_allocator_compare.py" \
    --config "$benchmark_dir/hopper_v4_vit.json" --variable-config "$benchmark_dir/hopper_v4_allocator_variable.json" \
    --profile-steps "$benchmark_profile_steps" --output "$benchmark_output/allocator"
  for benchmark_config in hopper_v4_vit hopper_v4_vit_layout hopper_v4_norm_boundary hopper_v4_activation_tiles hopper_v4_codec_pipeline hopper_v4_precision_boundaries; do
    benchmark_plan_flags=()
    if [[ "$benchmark_config" == hopper_v4_vit* || "$benchmark_config" == hopper_v4_norm_boundary ]]; then
      benchmark_plan_flags=(--require-fp8-plan)
    fi
    run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
      --config "$benchmark_dir/$benchmark_config.json" --sweep cartesian ${benchmark_plan_flags[@]+"${benchmark_plan_flags[@]}"} \
      --deterministic --fill-uninitialized-memory --warmup 10 --iterations 30 --repeats 5 \
      --profile-steps "$benchmark_profile_steps" --profile-replay-steps 1 \
      --output "$benchmark_output/$benchmark_config"
  done
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_attention_pack.py" \
    --bits 4 8 --group-sizes 128 512 1024 --output "$benchmark_output/attention-pack.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_spatial_layout.py" \
    --channels 1 3 4 16 --output "$benchmark_output/spatial-layout.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_pack.py" \
    --bits 2 --group-sizes 512 1024 --variants batched serial --groups-per-program 2 4 \
    --num-warps 1 2 4 --output "$benchmark_output/pack-v4.json"
  exit "$benchmark_status"
fi

if [[ "$benchmark_recovery_v3" == 1 ]]; then
  # Recover the previously failed model first. Correctness uses allocation fills;
  # the fill matrix separately controls filling in performance passes.
  for benchmark_config in hopper_vit_validation hopper_vit_fill; do
    run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
      --config "$benchmark_dir/$benchmark_config.json" --sweep cartesian --require-fp8-plan \
      --deterministic --fill-uninitialized-memory --warmup 10 --iterations 30 --repeats 5 \
      --profile-steps "$benchmark_profile_steps" --profile-replay-steps 1 \
      --output "$benchmark_output/$benchmark_config"
  done
  for benchmark_config in hopper_layer_tuning hopper_batch_group_variants hopper_pipeline_recovery hopper_codec_pipeline hopper_precision_calibration hopper_selective_calibration; do
    run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
      --config "$benchmark_dir/$benchmark_config.json" --sweep cartesian \
      --profile-steps "$benchmark_profile_steps" --profile-replay-steps 1 \
      --output "$benchmark_output/$benchmark_config"
  done
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_unpack.py" \
    --shapes 100864x3072 100864x768 --bits 2 --group-sizes 1024 \
    --tiles 32x32 32x64 64x64 --warps 4 8 --execution both \
    --output "$benchmark_output/unpack-focused.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_pack.py" \
    --shapes 100864x3072 100864x768 --source-dtypes bf16 e4m3fn \
    --bits 2 --group-sizes 512 1024 --groups-per-program 4 8 --execution both \
    --output "$benchmark_output/pack-focused.json"
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_dir/hopper_attention_latency.json" \
    --latency-steps "${BENCHMARK_LATENCY_STEPS:-160}" --profile-steps "$benchmark_profile_steps" \
    --output "$benchmark_output/hopper_attention_latency"
  exit "$benchmark_status"
fi

# Focused measurements come first, so an interrupted broad sweep still leaves
# evidence for runtime, norm fusion and the original reconstruction bottleneck.
for benchmark_config in h100_runtime_recovery h100_norm_recovery h100_precision_recovery; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_dir/$benchmark_config.json" --sweep cartesian \
    --profile-steps "$benchmark_profile_steps" --output "$benchmark_output/$benchmark_config"
done
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_unpack.py" \
  --shapes 100864x3072 100864x768 --bits 2 --group-sizes 1024 \
  --tiles 32x32 32x64 64x64 --warps 4 8 --execution both \
  --output "$benchmark_output/unpack-focused.json"
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/h100_vit_recovery.json" --require-fp8-plan \
  --deterministic --fill-uninitialized-memory --warmup 10 --iterations 30 --repeats 5 \
  --profile-steps "$benchmark_profile_steps" --output "$benchmark_output/h100_vit_recovery"
if [[ "$benchmark_focused" == 1 ]]; then exit "$benchmark_status"; fi

# Every supported operator, full backward and individual requested gradients.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/all_operators.json" --gradient-modes all dx dw db none \
  --profile-steps "$benchmark_profile_steps" \
  --output "$benchmark_output/operators"

# Every declared setting axis for all operator/pipeline/codec workloads.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/all_settings.json" --sweep one-at-a-time \
  --max-experiments 100000 --profile-steps "$benchmark_profile_steps" \
  --output "$benchmark_output/settings"

# Fully cross the finite input/output/scale/block precision-boundary grid.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/precision_boundaries.json" --sweep cartesian \
  --profile-steps "$benchmark_profile_steps" --output "$benchmark_output/precision-boundaries"

# Random-init ViT, torchvision ResNet18 and EfficientNet-B0; all five feature cases.
run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
  --config "$benchmark_dir/hopper_v5_models.json" --require-fp8-plan --deterministic --fill-uninitialized-memory \
  --warmup 10 --iterations 30 --repeats 5 --profile-steps "$benchmark_profile_steps" \
  --profile-detail trace --compress-traces --output "$benchmark_output/models"

# ViT family ablations, then the crossed hot factors.
for benchmark_vit_config in h100_vit_ablation h100_vit_interactions h100_vit_backends; do
  benchmark_sweep=one-at-a-time
  if [[ "$benchmark_vit_config" != h100_vit_ablation ]]; then benchmark_sweep=cartesian; fi
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_do_ops.py" \
    --config "$benchmark_dir/$benchmark_vit_config.json" --sweep "$benchmark_sweep" \
    --require-fp8-plan --deterministic --fill-uninitialized-memory --capture-workloads \
    --warmup 10 --iterations 30 --repeats 5 --profile-steps "$benchmark_profile_steps" \
    --output "$benchmark_output/$benchmark_vit_config"
done

# Shared-operand GEMMs plus complete BF16 / current / delayed FP8 Linear updates.
# Add torchao to --implementations separately if it is already installed.
for benchmark_backward_backend in inherit native_fast native_precise; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_fp8_linear_compare.py" \
    --implementations bf16 deepopt-current deepopt-delayed \
    --backward-backend "$benchmark_backward_backend" --prepared-gemms \
    --profile-steps "$benchmark_profile_steps" \
    --output "$benchmark_output/linear-$benchmark_backward_backend"
done

# Same packed payload: original grouped, tiled, decode/copy and production auto.
for benchmark_source_dtype in bf16 e4m3fn e5m2; do
  run_benchmark "$benchmark_python" "$benchmark_dir/benchmark_activation_unpack.py" \
    --source-dtype "$benchmark_source_dtype" --dtypes bf16 e4m3fn e5m2 \
    --bits 1 2 4 8 --group-sizes 128 512 1024 2048 --execution both \
    --output "$benchmark_output/unpack-$benchmark_source_dtype.json"
done
exit "$benchmark_status"
'''
    (OUTPUT / "run_full_comparisons.sh").write_text(runner, encoding="utf-8")
    EXPORTED_FILES.add("run_full_comparisons.sh")

    guide = f'''FULL DEEPOPT / TORCH BF16 COMPARISONS

Run this directory on the target H100/H200 with the updated DeepOpt build installed.
Recommended library Python-source SHA256: {library_source_sha256}
Compare this with each result's environment.deepopt_source_sha256.
V6 parallel suite (all five feature modes; batch512/image224 and historical controls):
  bash run_full_comparisons.sh ./results-v6 --recovery-v6 --scheduler auto --devices auto --cpu-workers 4 --gpu-concurrency auto --timing-isolation exclusive --deduplicate-effective-settings --profile-policy selected --resume
Use --preflight-only first to compile the failure paths and run all model modes.
Use --timing-isolation shared for memory-admitted screening; write a different
output directory and follow with an exclusive qualification run. Unknown memory
footprints calibrate exclusively. One busy GPU need not speed up with concurrency.
Use --full-coverage to additionally enumerate all public operators, requested
gradients and declared setting axes; old full-coverage launcher modes remain.
Optional actual CIFAR10 epochs: append --cifar10-root.
The default CIFAR10 root is ./data. Set --cifar10-root or the CIFAR10_ROOT
environment variable to use another directory.
The default model matrices use resident synthetic inputs and do not measure a dataset epoch.
CNN weights=None; neither model weights nor CIFAR10 data are downloaded.
Keep _gpu_lease.py and analysis/ beside standalone codec/allocator programs.
Direct programs and the v6 suite honor the same per-user physical GPU broker.
Local scheduling has no Dask dependency. --scheduler dask uses local Dask
workers with the same broker; install dask[distributed] in that environment if desired.
No MPS, RMM, spilling or system GPU-mode changes are made.
The benchmark files import the installed library. They do not install packages,
select a repository source tree, or download pretrained models or datasets.
The unpack controls need the optimized build that supports the implementation
argument; they cannot run against the unmodified older codec.
The v3 packer and independent norm controls require the updated v3 build.
The direct-stride, serial/warp, compact-layout, and new runtime controls require
the updated v4 build. Optional GEMM diagnostics are labeled unavailable on an
older compatible build; this does not replace failed kernel measurements.

Compressed LayerNorm always restores decoded xhat mean and its forward second
moment in the updated build, even with norm_xhat_recenter=false. That flag still
controls BatchNorm/GroupNorm mean correction. Normalization diagnostics report
the effective correction, second-moment error, saved gain bytes and merge costs.
Use the matching updated library; old compression checkpoints cannot exact-resume
across this numerical recipe change. Start a new run from model weights or keep
the original library to continue an old run.

FILES
benchmark_do_ops.py: self-contained full five-case operator/model suite.
benchmark_fp8_linear_compare.py: BF16/current FP8/delayed FP8 Linear updates,
  plus shared-operand Triton/native-fast/native-precise GEMM probes.
benchmark_activation_unpack.py: original grouped/tiled/copy/auto reconstruction.
benchmark_activation_pack.py: byte-exact grouped/batched independent-group packing;
  keep benchmark_activation_unpack.py beside it for shared measurement helpers.
benchmark_attention_pack.py: exact materialized/direct-strided Q/K/V packing controls.
benchmark_spatial_layout.py: equal-operand stride-1/16-alignment spatial convolution probes.
benchmark_fp8_accumulation.py: shared-operand forward/dx/dw references, codec error,
  precise/fast accumulation and transpose-preparation cost at the full model shapes.
  Both layout probes use the adjacent pack/unpack scripts for measurement/identity.
benchmark_allocator_compare.py: fresh-process native allocator off/on experiments.
toy_quest_saved_linear.py: standalone quantizer-only saved-input comparison;
  native BF16, min/max and Hadamard/RMS Gaussian codecs, real packing and toy MLP training.
benchmark_normalization.py: direct norm timing, phases, replay, and variant controls.
diagnose_training_stability.py + stability_oracle.py: matched real-batch gradients,
  actual custom-optimizer updates, codec audits and graph/fill comparisons.
  These diagnostics require a captured snapshot. Use diagnose_training_stability.py
  --help to see the replay options.
all_operators.json: {operators} operator/pipeline workloads and {codecs} codec controls.
all_settings.json: all workloads over {counts['settings_axes']} setting axes / {settings_variants} OAT variants.
h100_vit_recovery.json: ViT base patch16/224, batch512, ten classes, AdamW/EMA.
h100_vit_ablation.json: all/none/each compression family, including hook controls.
h100_vit_interactions.json: crossed eager/graph, backward backend, group,
  bit width and recentering factors ({vit_interaction_tasks} requested tasks).
precision_boundaries.json: crossed input/output types, scales, FP8 formats/blocks.
interaction.json: original six regression workloads with new accuracy policies.
settings_sweep.json / model_ablation.json: smaller diagnostic configs.
analysis/: editable module sources matching the embedded runner. To use edits,
  run python analysis/benchmark_do_ops.py with the same config/CLI arguments.
manifest.json: file/source hashes, settings and workload counts.

The five cases are torch_bf16, do_bf16, compression, fp8, fp8_compression.
Codecs use separately labeled storage/roundtrip controls, not training cases.

FULL RUN
  bash run_full_comparisons.sh /path/to/results
  bash run_full_comparisons.sh /path/to/results --focused
  bash run_full_comparisons.sh /path/to/results-v3 --recovery-v3
  bash run_full_comparisons.sh /path/to/results-v4 --recovery-v4
  bash run_full_comparisons.sh /path/to/results-v5 --recovery-v5

V5 includes torchvision resnet18 and efficientnet_b0 with weights=None, plus the
existing random-init timm ViT. CNNs default to batch128, 224px, 10 classes and EMA;
edit factory_kwargs to match your workload. No pretrained weights are downloaded.
Every model/setting retains all five feature cases. V5 timings explicitly disable
allocation filling; correctness keeps it enabled. Compact auto/norm auto changes
are measured-domain rollouts; producer fusion, recomputation, warp/index choices,
precise dW and operand preparation remain independently benchmarked experiments.
V5 samples up to 16384 values per floating-point tensor to bound snapshot size;
all tensors still receive full finiteness checks. Set BENCHMARK_MAX_NUMERICAL_ELEMENTS=65536
to match v4's sample cap or 0 for unsampled numerical snapshots. The cap is part
of the resume identity. Compare error metrics using matching sample policies.
Use --profile-detail stacks for full Python stacks, summary for counters only,
--trace-workloads/--trace-cases for selection, and --trace-directory for raw traces
on another filesystem. Gzip is supported. Timing/snapshots are saved atomically
before optional profiles. Set BENCHMARK_TRACE_DIRECTORY to forward a separate
trace directory to all v5 model stages. A partial diagnostic can resume without
repeating timing.

--recovery-v4 runs ten targeted stages: fresh-process allocator conditions with
performance filling disabled and correctness filling enabled, full ViT control
and spatial layout comparison, actual FP32-to-FP8 norm boundary, activation tile
and pack variants, eight-bit precision exclusions/backends, direct-strided
attention packing, equal-operand convolution layouts, and warp/serial packing.
The allocator launcher preserves existing options, requires the native backend,
and records both allocator environment aliases in child identity. Run it from a
fresh process; allocation settings do not change with runtime execution modes.
All five training cases remain present. These measurements still use synthetic
inputs; use a real-data validation factory before selecting a new lossy recipe.

--recovery-v3 runs eleven targeted stages and exits: full-model launch recovery,
correctness-fill/performance-fill comparison, independent norm candidates,
eager/graph pipelines, batched packing, warmed optimizer precision/recipe probes,
large decoder/packer measurements, and serialized attention latency/GC diagnostics.
All five training cases remain present in the operator/model stages. The Hopper
config names describe an architecture family; actual GPU identity is recorded.
hopper_vit_packing.json and hopper_vit_selective.json are optional full-model
follow-ups after the targeted stages. Selective policies use eager hook controls;
they are experiments, not a promoted production recipe.

For real training-quality curves, use a custom data factory with validate(model),
then run --quality-steps 200 --quality-interval 20 at several --seed values.
The factory callback returns finite scalar task metrics; reset() owns stateful
data/validation reset. quality-curves.csv and summary.json retain the observations.
Without a callback, validation is explicitly unavailable. Synthetic loss curves
and the first-update checks do not establish task accuracy or convergence.

Resume is enabled by default for every benchmark stage. Completed matching
cases are retained; failed/incomplete cases retry. Use --no-resume for a fresh
measurement. Build, script, configuration and environment changes invalidate
old checkpoints. Keep snapshots and requested traces with results.
Focused runtime/norm/precision/large-unpack/ViT measurements precede the broad
sweep. Candidate normalization fusion is opt-in in the norm configuration;
production auto uses guarded v3-measured H100 norm domains; other policies
retain the general path until target-GPU acceptance.

This is a large run. It continues subsequent benchmark sections if one fails,
preserves the failure reports, and exits nonzero if any section failed.
The all-operator gradient probes request {counts['all_operator_tasks_with_gradient_probes']} tasks; the
all-settings run requests {counts['all_settings_tasks']} tasks. Unsupported combinations remain
explicit skipped/error rows. They are not silently counted as measured coverage.
Runs use separate workers, sequentially on one GPU; no concurrent measurements.
Set BENCHMARK_PYTHON=/path/to/python to select the environment.
Set BENCHMARK_PROFILE_STEPS=0 to omit the separate profiler passes if needed.
Default full-run profile steps are 3; traces from this large sweep need disk space.

INDIVIDUAL RUNS (commands below assume the current directory contains these files)

1. Validate the grid without Torch/CUDA or executing kernels:
  python benchmark_do_ops.py --config all_settings.json --max-experiments 100000 \\
    --dry-run --output plan-settings

2. Every operator, all five cases, full and requested-gradient probes:
  python benchmark_do_ops.py --config all_operators.json \\
    --gradient-modes all dx dw db none --profile-steps 3 --output results/operators

3. All workloads and all declared setting axes, changing one factor at a time:
  python benchmark_do_ops.py --config all_settings.json --sweep one-at-a-time \\
    --max-experiments 100000 --profile-steps 3 --output results/settings

4. Fully crossed finite precision-boundary comparisons:
  python benchmark_do_ops.py --config precision_boundaries.json \\
    --sweep cartesian --profile-steps 3 --output results/boundaries

5. Full ViT five-case comparison:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_recovery.json --require-fp8-plan --deterministic \\
    --fill-uninitialized-memory --capture-workloads --warmup 10 \\
    --iterations 30 --repeats 5 --profile-steps 3 --output results/vit

6. Compression-family attribution with the same ViT factory:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_ablation.json --require-fp8-plan --deterministic \\
    --fill-uninitialized-memory --profile-steps 3 --output results/vit-families

7. Cross the principal ViT regression factors:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_do_ops.py \\
    --config h100_vit_interactions.json --sweep cartesian --require-fp8-plan \\
    --deterministic --fill-uninitialized-memory --profile-steps 3 \\
    --output results/vit-interactions

8. Complete Linear and prepared shared-operand GEMM controls:
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python benchmark_fp8_linear_compare.py \\
    --implementations bf16 deepopt-current deepopt-delayed --prepared-gemms \\
    --profile-steps 3 --output results/linear
  Optional: append torchao to --implementations if already installed.
  Repeat --backward-backend native_fast and native_precise to change the full
  training executor. Prepared probes compare those GEMMs in the same invocation.

9. Reconstruction controls at the dominant original H100 shapes:
  python benchmark_activation_unpack.py --bits 1 2 4 8 \\
    --group-sizes 128 512 1024 2048 --dtypes bf16 e4m3fn e5m2 \\
    --execution both --output results/unpack.json
  Repeat --source-dtype bf16 / e4m3fn / e5m2 and optional --block 32x32.

RESULTS AND INTERPRETATION
The full operator/model suite writes report.md, results.csv, summary.json,
plan.json, per-case result.json, worker logs, numerical snapshots and traces.
It retains synchronized wall and CUDA stream samples, forward/loss/backward
phases, optimizer/update timing, memory/saved storage, output/input-gradient/
parameter-gradient/update accuracy, CPU operators, GPU kernel names/launches,
and the FP8/compression timing interaction:
  combined - fp8 - compression + do_bf16
Separate profiler passes do not instrument the throughput timing loops. For graph
cases the profiler diagnoses the eager path and is labeled accordingly; graph
throughput measures replay. Stream event times include host submission gaps.
Raw timings remain visible when an accuracy policy rejects a case. Qualified
speedups additionally require matching initial state/build and passing policies.
Independent gradient/update relative-L2 and cosine gates are enabled in configs.
Use --strict-accuracy to make numerical rejection fail the operator/model CLI.
Use --max-numerical-elements 0 for full rather than sampled numerical diagnostics.
Gradient modes dx/dw/db request separate complete executions; their times cannot
be added to reconstruct the backward time. Parameterless probes are skipped.

OAT tests all listed values individually; it does not test every interaction or
every possible tensor size/integer option. The explicit Cartesian configs test
their declared cross-products. To add a suspected interaction, narrow a JSON
grid and use --sweep cartesian; do not cross the entire broad settings grid.
Operator shapes/kwargs are editable. Captured native module workloads appear
in recorded-workloads.json; replay them with --config for model-specific shapes.
This replay is representative and cannot reconstruct every functional call,
arbitrary stride or saved tensor. Use the real-model result as the authority.

DEPENDENCIES AND BEFORE/AFTER
Use the original compatible CUDA PyTorch/Triton environment plus updated DeepOpt.
Full ViT needs timm and its dependencies; hf_conv1d additionally needs transformers.
Missing optional dependencies and unsupported operations remain visible as skips.
No automatic dependency installation occurs. --help and --dry-run need only Python.
The full ViT factory uses fresh untrained weights and resident data. It includes
AdamW and EMA, but does not reproduce the original trainer's metrics, gradient
clones or process-group orchestration. Rerun that original diagnostic for an
exact before/after comparison against the supplied trainer traces.
For old/new library comparisons, keep this benchmark code/config/device/seed/
environment policy fixed, run in separate installed-library environments, and
write different output directories. The old build's known failures are evidence;
do not disable accuracy/planner checks or replace failed rows with new timings.
CPU smoke tests validate portability/logic only; these exports have not been
measured on the H100 here and do not establish a GPU speedup.
'''
    guide += '''

Installed FlashAttention-2 / attention stability investigation
----------------------------------------------------------
See flash_attention_2.md for backend setup, supported inputs, alignment metrics,
actual optimizer-state replay, graph tests, and the real CIFAR10 training recipe.

Focused replay/configuration/layout recovery (run after installing the matching library):
  bash run_attention_recovery.sh ./attention-recovery --resume
The CUDA smoke test runs first and must pass before timing starts. Packing auto
selection covers measured H100 layouts; forced separate/joint controls remain.
Matrix model updates use the declared AdamW performance proxy, not the custom
optimizer or an estimate of CIFAR10 epoch convergence.
These files are separate from the full operator matrix:
  benchmark_attention_alignment.py --snapshot SNAPSHOT --output attention-real
  benchmark_vit_attention_stability.py --optimizer test_optimizer:SGD_NS_Overshoot \\
    --config CONFIG --backend flash_attn_2 --attention-bits 4 --epochs 32 --output vit-fa2-4
The training runner now inherits the supplied recipe, including its LR and
group size. Use --learning-rate YOUR_VERIFIED_LR for an explicit LR override;
no optimizer LR is guessed when neither CLI nor configuration supplies one.
For graph runs set loader.drop_last=true (or --drop-last) and cuda_graph.autograd=true.
Both support --resume. Keep attention_comparison.py and
diagnose_training_stability.py beside them. GPU correctness and 32-epoch stability
are target-H100 checks, not established by local CPU smoke tests.
\nOptimization candidate comparisons
----------------------------------
Compare automatic and explicit packing, convolution policy, attention boundaries,
and producer fusion using the recovery launcher. The quality stage runs the
CIFAR10 training replay.
  bash run_optimization_recovery.sh ./results-optimization --resume
  RECOVERY_STAGE=convolution bash run_optimization_recovery.sh ./results-optimization --resume
  RECOVERY_STAGE=quality bash run_optimization_recovery.sh ./results-quality --resume
The default suite is a short candidate screen; confirm winners with
BENCHMARK_WARMUP=10 BENCHMARK_ITERATIONS=30 BENCHMARK_REPEATS=5.
GPU throughput remains exclusive; preparation and independent GPUs can run in
parallel. Native spatial fallback is reported as mixed compute. The depthwise
stencil uses FP8 storage and FP32 arithmetic, not FP8 tensor-core GEMM.
Model qualification now reports numerical rejections separately from successful
execution. No numerical tolerance was relaxed.
\nRuntime speed optimization
--------------------------
The full-model comparison requests 40 rows across four models, five modes, and
eager and graph execution.
  bash run_runtime_speed.sh ./results-speed --resume
  bash run_runtime_speed.sh ./results-speed-final --resume --final
  SPEED_STAGE=cnn bash run_runtime_speed.sh ./results-speed --resume
  SPEED_STAGE=convolution bash run_runtime_speed.sh ./results-speed --resume
  SPEED_STAGE=vit bash run_runtime_speed.sh ./results-speed --resume
  SPEED_STAGE=attention bash run_runtime_speed.sh ./results-speed --resume
The launcher first runs 16 implementation/replay GPU tests. It disables large
numerical snapshots and approximation comparisons with --no-numerical-comparison;
finite checks, execution errors, matching controls and timing variability remain.
Preflight revision 3 checks convolution indexing/scales exactly with ternary FP8
codes, verifies random native eager/graph equality and unit-scale identities,
and records ideal-FP32 deviation separately as advisory precision diagnostics.
  python benchmark_attention_preflight.py --convolution-only --output ./results-speed/convolution-check.json
This fast focused check does not replace the full 16-test launcher preflight.
Its JSON links synthetic operand/output snapshots for diagnosis. Production
FP8 accumulation and training kernels are unchanged by this preflight repair.
eager-vs-graph.csv reports complete update ratios and memory observations.
Default attention is installed FlashAttention-2; SPEED_ATTENTION_BACKEND=native
selects the native 40-row comparison instead. Final throughput defaults to
exclusive GPU timing, with parallel CPU preparation and independent GPUs.
New kernel compilation and throughput must be validated on the target H100.
'''
    (OUTPUT / "RUN_BENCHMARKS.txt").write_text(guide, encoding="utf-8")
    EXPORTED_FILES.add("RUN_BENCHMARKS.txt")

    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(),
        "recommended_library_source_sha256": library_source_sha256, "cases": [
        "torch_bf16", "do_bf16", "compression", "fp8", "fp8_compression"], "coverage": counts,
        "files": {}, "canonical_analysis_sources": {}}
    for source in sorted((ROOT / "benchmarks/analysis").glob("*.py")):
        manifest["canonical_analysis_sources"][source.name] = sha256(source.read_bytes()).hexdigest()
    for name in sorted(EXPORTED_FILES):
        manifest["files"][name] = sha256((OUTPUT / name).read_bytes()).hexdigest()
    write_json("manifest.json", manifest)

    archive = OUTPUT.parent / (OUTPUT.name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(EXPORTED_FILES):
            bundle.write(OUTPUT / name, OUTPUT.name + "/" + name)
    print(json.dumps({"directory": str(OUTPUT), "archive": str(archive), "coverage": counts}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
